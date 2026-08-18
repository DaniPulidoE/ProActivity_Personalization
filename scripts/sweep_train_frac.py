# Sweep the amount of per-driver training data used to fine-tune the xLSTM head
# and plot validation accuracy vs. number of training segments.
#
# For every prefix size k (chronologically first k segments outside the held-out
# validation tail) a fresh copy of the population head is fine-tuned with the
# L2-SP penalty and evaluated on the SAME validation tail, giving the
# "personalization quality vs. data collection time" learning curve.
#
# The frozen backbone runs exactly once: all segments are embedded up front and
# every sweep point trains a Linear(64->5) on the cached embeddings (full-batch,
# deterministic — no seed variance).
#
# The adaptation itself lives in ProVoice.models.head_adapt, which fine_tune_XLSTM
# also calls, so the curve measures exactly the procedure that gets served. Two
# properties of that module are what make points on this curve comparable to each
# other: the step budget does not depend on k, and the anchor is specified as a
# prior PRECISION tau (lambda = tau/2k) so its strength does not either. See the
# head_adapt docstring — both used to drift with k, in opposite directions.
#
# Usage:
#   python -m scripts.sweep_train_frac \
#       --in-data data/labeled_data.jsonl \
#       --in-model trained_models/state_xlstm.pt \
#       --out results/train_frac_sweep.png
import argparse
import csv
import pathlib

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ProVoice.decision_engine import truncate_frames_by_seconds
from ProVoice.models.xlstm_model import (
    encode_and_resample,
    load_checkpoint,
    logits_to_label,
)
from ProVoice.models.head_adapt import (
    augment_z,
    adapt_head,
    DEFAULT_ADAPT_LR,
    DEFAULT_ADAPT_STEPS,
    DEFAULT_TAU,
)
from ProVoice.models.train_XLSTM import (
    set_seed,
    read_jsonl,
    iter_jsonl,
    normalize_row,
    make_collate,
    set_accuracy,
    set_macro_f1,
    set_mae,
    set_qwk,
)

LEVELS = [f"Level_{i}" for i in range(1, 6)]
SEGMENT_SECONDS = 20.0  # driver labels arrive every 20 s -> one segment per label


def build_segments(df: pd.DataFrame, window_seconds: float | None = None,
                   resample_hz: float | None = None):
    """Encode one (X, levels) pair per segment, in chronological order.

    Chronology = first appearance in the JSONL (groupby(sort=False)), NOT the
    lexicographic order of segment_id strings. Segments with missing or
    all-zero Level_* labels are skipped and reported, never collapsed into a
    bogus class-0 label. ``window_seconds`` truncates each segment to its
    last k seconds and ``resample_hz`` puts it on a fixed time grid — both must
    match the checkpoint's training contract, or the frozen backbone is fed
    sequences whose step count means something different than it did at
    training.

    ``levels`` is the multi-hot mark vector — the whole label. It is both the
    training target and the metric target, so a driver who marked several
    acceptable LoAs is never collapsed to one of them.
    """
    if not all(k in df.columns for k in LEVELS):
        raise ValueError(f"Input data has no {LEVELS} columns; labels are required.")
    Xs, vs, gids, skipped = [], [], [], []
    for gid, g in df.groupby("segment_id", sort=False):
        g = g.reset_index(drop=True)
        lv = pd.to_numeric(g[LEVELS].iloc[0], errors="coerce").astype(float).values
        if np.isnan(lv).any() or lv.sum() <= 0:
            skipped.append(gid)
            continue
        rows = g.to_dict("records")
        rows = truncate_frames_by_seconds(rows, window_seconds)
        X = encode_and_resample(rows, resample_hz, window_seconds)
        gids.append(gid); Xs.append(X)
        vs.append((lv > 0).astype(np.float32))
    if skipped:
        print(f"[warn] skipped {len(skipped)} segment(s) with missing/empty Level_* labels: "
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    return gids, Xs, vs


@torch.no_grad()
def embed_segments(model, Xs, vs, context_length: int, device: str, chunk: int = 32,
                   embed_fcd: Optional[bool] = None):
    """Frozen in_proj+backbone -> pooled last-step embeddings, (N, 64) or (N, 76).

    With ``embed_fcd`` the 12 static FCD dims are appended, giving the adapted
    head direct access to the task. See ``head_adapt`` for why that matters here
    and why the appended block is anchored at zero.
    """
    # None = infer from the head's width, which is the safest default: the
    # embeddings then cannot be a different width from the head consuming them.
    if embed_fcd is None:
        embed_fcd = model.head_uses_fcd()
    collate = make_collate(context_length)
    zs = []
    for i in range(0, len(Xs), chunk):
        xb, lb, _ = collate(list(zip(Xs[i:i + chunk], vs[i:i + chunk])))
        h = model.backbone(model.in_proj(xb.to(device)))
        # RIGHT-padded batches: read the hidden state at the last REAL frame.
        idx = (lb.to(h.device).long() - 1).clamp(min=0)
        z = h[torch.arange(h.size(0), device=h.device), idx]
        zs.append(augment_z(z, xb.to(device).to(torch.float32), embed_fcd).cpu())
    return torch.cat(zs, dim=0)


@torch.no_grad()
def evaluate(head: nn.Linear, Z: torch.Tensor, V: torch.Tensor, head_type: str):
    """Returns dict of metrics, all set-aware: acc, f1 (nominal); mae, qwk (ordinal).

    Every one credits any level the driver marked acceptable and reduces
    exactly to its single-label form when every row marks one level, so these
    are directly comparable with results produced before multi-label windows
    existed.
    """
    pred = logits_to_label(head(Z), head_type).cpu().numpy()
    lv = V.cpu().numpy()
    return {
        "acc": set_accuracy(lv, pred),
        "f1":  set_macro_f1(lv, pred, 5),
        "mae": set_mae(lv, pred),
        "qwk": set_qwk(lv, pred, 5),
    }


def pick_sweep_points(n_pool: int, max_points: int):
    """All prefix sizes 1..n_pool, thinned evenly if there are too many."""
    if n_pool <= max_points:
        return list(range(1, n_pool + 1))
    ks = np.unique(np.linspace(1, n_pool, max_points).round().astype(int))
    return ks.tolist()


def plot_curve(ks, maes, accs, base_mae, base_acc, n_val, out_png: pathlib.Path):
    """Learning curve: set-MAE (primary) and set-accuracy vs. training segments.

    Two stacked panels rather than one panel with two y-axes. set-MAE lives on
    [0, K-1] where LOWER is better and set-accuracy on [0, 1] where HIGHER is
    better; overlaying them on a shared scale would be a dual-axis chart, which
    makes the crossing point of the two lines an artifact of the axis limits
    rather than a fact about the data. Separate panels also let each carry its
    own population baseline and its own "better" direction.

    Both metrics are set-aware (see train_XLSTM.resolve_targets) and reduce
    exactly to MAE/accuracy on single-label rows. macro-F1 is deliberately NOT
    plotted: it averages only over LoA levels present in the tail, so its
    denominator varies per driver and curves are not comparable across
    participants. It stays in the CSV.
    """
    surface, grid, axis_ink, muted, sec_ink = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781", "#52514e"
    blue, aqua = "#2a78d6", "#1baf7a"   # categorical slots 1 and 3 (validated pair)

    fig, (ax_m, ax_a) = plt.subplots(
        2, 1, figsize=(8, 7), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18})
    fig.patch.set_facecolor(surface)

    panels = (
        (ax_m, maes, base_mae, blue, f"set-MAE (levels)  ↓ lower is better",
         max(0.05, (max(list(maes) + [base_mae]) or 1.0)) * 1.25),
        (ax_a, accs, base_acc, aqua, "set-accuracy  ↑ higher is better", 1.0),
    )
    for ax, ys, base, color, ylab, ytop in panels:
        ax.set_facecolor(surface)
        # Dashed ONLY for the reference line — gridlines stay solid hairlines, so
        # dashing unambiguously means "threshold", never "grid".
        ax.axhline(base, color=muted, lw=1.5, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"population baseline  {base:.2f}", xy=(0, base),
                    xytext=(2, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=sec_ink)
        ax.plot(ks, ys, color=color, lw=2, marker="o", ms=5,
                markeredgecolor=surface, markeredgewidth=1, zorder=3)
        # Selective direct label: the endpoint only. Text wears an ink token,
        # not the series color; the mark beside it carries identity. This is
        # also the visible-label relief the aqua/surface contrast check asks for.
        #
        # Nudged AWAY from the baseline rather than centred on the point: a curve
        # that ends level with its baseline (the common "no improvement" case)
        # otherwise renders the value on top of the dashed rule, which reads as
        # struck-through text.
        if len(ks):
            above = ys[-1] >= base
            ax.annotate(f"{ys[-1]:.2f}", xy=(ks[-1], ys[-1]),
                        xytext=(7, 9 if above else -9), textcoords="offset points",
                        ha="left", va="bottom" if above else "top",
                        fontsize=9, color=sec_ink)
        ax.set_ylabel(ylab, color=sec_ink, fontsize=10)
        ax.set_ylim(0, ytop)
        ax.grid(axis="y", color=grid, lw=0.75)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis_ink)
        ax.tick_params(colors=muted, labelcolor=sec_ink)

    ax_m.set_xlim(left=0)
    # Segments are a COUNT — a tick at 2.5 segments is not a thing that exists.
    ax_a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax_a.set_xlabel(f"Training segments (chronological prefix)   —   "
                    f"held-out validation tail n={n_val}", color=sec_ink)

    # one segment = one driver label every 20 s
    sec = ax_m.secondary_xaxis("top", functions=(lambda k: k * SEGMENT_SECONDS / 60.0,
                                                 lambda m: m * 60.0 / SEGMENT_SECONDS))
    sec.set_xlabel("Driving data collected (minutes)", color=muted, fontsize=9)
    sec.spines["top"].set_visible(False)
    sec.tick_params(colors=muted, labelcolor=muted, labelsize=8)

    ax_m.set_title("Head fine-tuning: personalization vs. data collected",
                   color="#0b0b0b", fontsize=12, pad=26)

    fig.savefig(out_png, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Sweep training-prefix size for xLSTM head fine-tuning and plot "
                    "validation accuracy vs. number of segments.")
    ap.add_argument("--in-data",  dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--in-model", dest="in_model", default="trained_models/state_xlstm.pt")
    ap.add_argument("--out",      dest="out_png",  default="results/train_frac_sweep.png",
                    help="Output plot path; a CSV with the raw numbers is written next to it.")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.3,
                    help="Chronologically-last fraction of segments held out as the fixed "
                         "validation tail (same convention as fine_tune_XLSTM.py).")
    ap.add_argument("--lr",     type=float, default=DEFAULT_ADAPT_LR)
    ap.add_argument("--tau",    type=float, default=DEFAULT_TAU,
                    help="PRIOR PRECISION of the L2-SP anchor, held constant across the "
                         "whole sweep. The per-point lambda is derived as tau/(2k), which "
                         "is what makes points COMPARABLE along k: the objective is a "
                         "batch mean, so a fixed lambda would realise tau=2*k*lambda and "
                         "the anchor would grow stronger as data accumulates — and vanish "
                         "at the small-k end where the design most wants it. Must match "
                         "the tau used by fine_tune_XLSTM.py, or the curve describes a "
                         "different estimator than the one that gets served.")
    ap.add_argument("--steps", type=int, default=DEFAULT_ADAPT_STEPS,
                    help="Full-batch gradient steps per sweep point. Fixed and "
                         "k-INDEPENDENT on purpose (was --epochs).")
    ap.add_argument("--max-points", dest="max_points", type=int, default=60,
                    help="Cap on the number of sweep points (thinned evenly if exceeded).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [normalize_row(r) for r in iter_jsonl(pathlib.Path(args.in_jsonl))]
    if not rows:
        raise ValueError("JSONL is empty or contains no valid rows.")
    df = pd.DataFrame(rows)
    if "segment_id" not in df.columns or df["segment_id"].eq("").all():
        raise ValueError("Missing segment_id; cannot build sequences.")

    model, arch = load_checkpoint(args.in_model)
    model.to(device).eval()
    context_length = arch["context_length"]
    head_type = arch.get("head_type", "softmax")
    window_seconds = arch.get("window_seconds")
    resample_hz = arch.get("resample_hz")
    print(f"[model] head_type={head_type} window_seconds={window_seconds} "
          f"resample_hz={resample_hz} (from checkpoint)")

    gids, Xs, vs = build_segments(df, window_seconds=window_seconds, resample_hz=resample_hz)
    n_seg = len(gids)
    if n_seg < 3:
        raise ValueError(f"Need at least 3 labeled segments to sweep, got {n_seg}.")

    print(f"[embed] {n_seg} segments through frozen backbone (context_length={context_length}, device={device})")
    Z = embed_segments(model, Xs, vs, context_length, device)
    V = torch.from_numpy(np.stack(vs, axis=0))
    n_multi = int((V.sum(dim=-1) > 1).sum())
    if n_multi:
        print(f"[info] {n_multi}/{n_seg} segment(s) mark several acceptable LoAs; "
              f"all metrics below are set-aware (scored against the marked level "
              f"nearest the prediction).")

    n_val = max(1, round(args.val_frac * n_seg))
    if n_val >= n_seg:
        raise ValueError(f"--val-frac {args.val_frac} leaves no training segments (total={n_seg}).")
    Zpool, Vpool = Z[: n_seg - n_val], V[: n_seg - n_val]
    Zval,  Vval  = Z[n_seg - n_val:], V[n_seg - n_val:]
    print(f"[split] temporal: pool={len(Vpool)} earliest segments, val={n_val} latest segments")

    base = evaluate(model.head, Zval, Vval, head_type)
    print(f"[baseline] population head: set-acc={base['acc']:.3f} "
          f"macro-F1={base['f1']:.3f} set-MAE={base['mae']:.3f} QWK={base['qwk']:.3f}")

    ks = pick_sweep_points(len(Vpool), args.max_points)
    results, infos = [], []
    worst_grad = 0.0
    for k in ks:
        # Same call the deployed fine-tuner makes: full-batch, fixed steps, no
        # epoch selection, lambda derived from tau. The curve and the served
        # head are the same estimator by construction, not by convention.
        head, info = adapt_head(model.head, Zpool[:k], Vpool[:k],
                                tau=args.tau, head_type=head_type,
                                steps=args.steps, lr=args.lr)
        m = evaluate(head, Zval, Vval, head_type)
        results.append(m); infos.append(info)
        worst_grad = max(worst_grad, info["grad_norm"])
        print(f"[k={k:3d}] ({k * SEGMENT_SECONDS / 60.0:5.1f} min) set-acc={m['acc']:.3f} "
              f"macro-F1={m['f1']:.3f} set-MAE={m['mae']:.3f} QWK={m['qwk']:.3f} "
              f"(lam={info['l2sp']:.2e}, |grad|={info['grad_norm']:.2e})")
    # One number to check before trusting the curve: every point must have
    # reached its MAP, or differences along k are partly differences in how far
    # each point got rather than in what the data supports.
    print(f"[converge] worst |grad| over the sweep = {worst_grad:.2e} "
          f"({'OK' if worst_grad < 1e-3 else 'HIGH — raise --steps or --lr'})")

    out_png = pathlib.Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_png.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Every metric is set-aware and reduces exactly to its single-label form
        # on single-label rows, so the separate val_*/val_set_* pairs the CSV
        # used to carry were the same number whenever they were comparable.
        w.writerow(["n_train_segments", "minutes",
                    "val_set_accuracy", "val_set_macro_f1", "val_set_mae", "val_set_qwk",
                    "baseline_set_accuracy", "baseline_set_macro_f1",
                    "baseline_set_mae", "baseline_set_qwk",
                    "head_type", "n_val", "n_multi_label", "lr", "tau",
                    # l2sp VARIES along the sweep (= tau/2k) and is the number the
                    # Laplace layer needs, so it is recorded per row rather than
                    # once in a footer. grad_norm certifies the point converged.
                    "l2sp", "steps", "grad_norm"])
        for k, m, info in zip(ks, results, infos):
            w.writerow([k, round(k * SEGMENT_SECONDS / 60.0, 2),
                        m["acc"], m["f1"], m["mae"], m["qwk"],
                        base["acc"], base["f1"], base["mae"], base["qwk"],
                        head_type, n_val, n_multi, args.lr, args.tau,
                        info["l2sp"], args.steps, info["grad_norm"]])

    plot_curve(ks, [m["mae"] for m in results], [m["acc"] for m in results],
               base["mae"], base["acc"], n_val, out_png)
    print(f"[OK] plot  -> {out_png}")
    print(f"[OK] table -> {out_csv}")


if __name__ == "__main__":
    main()
