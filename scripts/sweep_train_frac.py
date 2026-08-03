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
# Usage:
#   python -m scripts.sweep_train_frac \
#       --in-data data/labeled_data.jsonl \
#       --in-model trained_models/state_xlstm.pt \
#       --out results/train_frac_sweep.png
import argparse
import copy
import csv
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ProVoice.decision_engine import truncate_frames_by_seconds
from ProVoice.models.xlstm_model import (
    encode_and_resample,
    load_checkpoint,
    logits_to_probs,
    levels_to_distribution,
    soft_corn_loss,
)
from ProVoice.models.train_XLSTM import (
    set_seed,
    read_jsonl,
    normalize_row,
    make_collate,
    accuracy,
    macro_f1,
    mae,
    qwk,
    set_accuracy,
    set_mae,
)

LEVELS = [f"Level_{i}" for i in range(1, 6)]
SEGMENT_SECONDS = 20.0  # driver labels arrive every 20 s -> one segment per label


def build_segments(df: pd.DataFrame, window_seconds: float | None = None,
                   resample_hz: float | None = None):
    """Encode one (X, y, levels) triple per segment, in chronological order.

    Chronology = first appearance in the JSONL (groupby(sort=False)), NOT the
    lexicographic order of segment_id strings. Segments with missing or
    all-zero Level_* labels are skipped and reported, never argmax'd into a
    bogus class-0 label. ``window_seconds`` truncates each segment to its
    last k seconds and ``resample_hz`` puts it on a fixed time grid — both must
    match the checkpoint's training contract, or the frozen backbone is fed
    sequences whose step count means something different than it did at
    training.

    ``levels`` is the multi-hot mark vector (the training target, so a driver
    who marked several acceptable LoAs is not collapsed); ``y`` is its argmax,
    kept for the single-label metrics.
    """
    if not all(k in df.columns for k in LEVELS):
        raise ValueError(f"Input data has no {LEVELS} columns; labels are required.")
    Xs, ys, vs, gids, skipped = [], [], [], [], []
    for gid, g in df.groupby("segment_id", sort=False):
        g = g.reset_index(drop=True)
        lv = pd.to_numeric(g[LEVELS].iloc[0], errors="coerce").astype(float).values
        if np.isnan(lv).any() or lv.sum() <= 0:
            skipped.append(gid)
            continue
        y = int(np.argmax(lv))
        rows = [g.iloc[i].to_dict() for i in range(len(g))]
        rows = truncate_frames_by_seconds(rows, window_seconds)
        X = encode_and_resample(rows, resample_hz, window_seconds)
        gids.append(gid); Xs.append(X); ys.append(y)
        vs.append((lv > 0).astype(np.float32))
    if skipped:
        print(f"[warn] skipped {len(skipped)} segment(s) with missing/empty Level_* labels: "
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    return gids, Xs, ys, vs


@torch.no_grad()
def embed_segments(model, Xs, ys, context_length: int, device: str, chunk: int = 32):
    """Frozen in_proj+backbone -> pooled last-step embeddings, shape (N, 64)."""
    collate = make_collate(context_length)
    zs = []
    for i in range(0, len(Xs), chunk):
        xb, _, lb, _ = collate(list(zip(Xs[i:i + chunk], ys[i:i + chunk])))
        h = model.backbone(model.in_proj(xb.to(device)))
        # RIGHT-padded batches: read the hidden state at the last REAL frame.
        idx = (lb.to(h.device).long() - 1).clamp(min=0)
        zs.append(h[torch.arange(h.size(0), device=h.device), idx].cpu())
    return torch.cat(zs, dim=0)


def fine_tune_head(pop_head: nn.Linear, Z: torch.Tensor, V: torch.Tensor,
                   lr: float, l2sp: float, epochs: int, head_type: str) -> nn.Linear:
    """Fine-tune a fresh copy of the population head on cached embeddings.

    Full-batch AdamW with the L2-SP anchor lambda*||theta - theta_pop||^2, so a
    given (k, lr, l2sp, epochs) always yields the same head — no seed variance.
    The loss matches the checkpoint's head type (CE for softmax, soft-CORN for
    corn); ``V`` is the multi-hot marked-level target both losses consume.
    """
    head = copy.deepcopy(pop_head)
    theta_pop = {n: p.detach().clone() for n, p in pop_head.named_parameters()}
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    if head_type == "corn":
        loss_fn = lambda logits, lvl: soft_corn_loss(logits, lvl)
    else:
        _ce = nn.CrossEntropyLoss()
        loss_fn = lambda logits, lvl: _ce(logits, levels_to_distribution(lvl))
    for _ in range(epochs):
        logits = head(Z)
        pen = sum(((p - theta_pop[n]) ** 2).sum() for n, p in head.named_parameters())
        loss = loss_fn(logits, V) + l2sp * pen
        opt.zero_grad(); loss.backward(); opt.step()
    return head


@torch.no_grad()
def evaluate(head: nn.Linear, Z: torch.Tensor, y: torch.Tensor, V: torch.Tensor,
             head_type: str):
    """Returns dict of metrics: acc, f1 (nominal); mae, qwk (ordinal).

    ``set_acc``/``set_mae`` credit any level the driver marked acceptable and
    reduce exactly to ``acc``/``mae`` when every row marks one level — report
    them whenever the data has multi-label windows.
    """
    pred = logits_to_probs(head(Z), head_type).argmax(dim=-1).cpu().numpy()
    true = y.cpu().numpy()
    lv = V.cpu().numpy()
    return {
        "acc": accuracy(true, pred),
        "f1":  macro_f1(true, pred, 5),
        "mae": mae(true, pred),
        "qwk": qwk(true, pred, 5),
        "set_acc": set_accuracy(lv, pred),
        "set_mae": set_mae(lv, pred),
    }


def pick_sweep_points(n_pool: int, max_points: int):
    """All prefix sizes 1..n_pool, thinned evenly if there are too many."""
    if n_pool <= max_points:
        return list(range(1, n_pool + 1))
    ks = np.unique(np.linspace(1, n_pool, max_points).round().astype(int))
    return ks.tolist()


def plot_curve(ks, accs, f1s, base_acc, n_val, out_png: pathlib.Path):
    surface, grid, axis_ink, muted, sec_ink = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781", "#52514e"
    blue, aqua = "#2a78d6", "#1baf7a"  # categorical slots 1-2 (validated adjacent pair)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(surface); ax.set_facecolor(surface)

    ax.axhline(base_acc, color=muted, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"population baseline  {base_acc:.2f}", xy=(1, base_acc),
                xytext=(0, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, color=sec_ink)

    ax.plot(ks, accs, color=blue, lw=2, marker="o", ms=5,
            markeredgecolor=surface, markeredgewidth=1, label="accuracy", zorder=3)
    ax.plot(ks, f1s, color=aqua, lw=2, marker="o", ms=5,
            markeredgecolor=surface, markeredgewidth=1, label="macro-F1", zorder=3)

    ax.set_xlabel("Training segments (chronological prefix)", color=sec_ink)
    ax.set_ylabel(f"Score on held-out validation tail (n={n_val})", color=sec_ink)
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.grid(axis="y", color=grid, lw=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis_ink)
    ax.tick_params(colors=muted, labelcolor=sec_ink)

    # one segment = one driver label every 20 s
    sec = ax.secondary_xaxis("top", functions=(lambda k: k * SEGMENT_SECONDS / 60.0,
                                               lambda m: m * 60.0 / SEGMENT_SECONDS))
    sec.set_xlabel("Driving data collected (minutes)", color=muted, fontsize=9)
    sec.spines["top"].set_visible(False)
    sec.tick_params(colors=muted, labelcolor=muted, labelsize=8)

    ax.set_title("Head fine-tuning: personalization vs. data collected",
                 color="#0b0b0b", fontsize=12, pad=24)
    ax.legend(loc="lower right", frameon=False, labelcolor=sec_ink, fontsize=9)

    fig.tight_layout()
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
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.2,
                    help="Chronologically-last fraction of segments held out as the fixed "
                         "validation tail (same convention as fine_tune_XLSTM.py).")
    ap.add_argument("--lr",     type=float, default=5e-4)
    ap.add_argument("--l2sp",   type=float, default=0.01,
                    help="Strength of the L2-SP anchor to the population head.")
    ap.add_argument("--epochs", type=int, default=300,
                    help="Full-batch gradient steps per sweep point.")
    ap.add_argument("--max-points", dest="max_points", type=int, default=60,
                    help="Cap on the number of sweep points (thinned evenly if exceeded).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [normalize_row(r) for r in read_jsonl(pathlib.Path(args.in_jsonl))]
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

    gids, Xs, ys, vs = build_segments(df, window_seconds=window_seconds, resample_hz=resample_hz)
    n_seg = len(gids)
    if n_seg < 3:
        raise ValueError(f"Need at least 3 labeled segments to sweep, got {n_seg}.")

    print(f"[embed] {n_seg} segments through frozen backbone (context_length={context_length}, device={device})")
    Z = embed_segments(model, Xs, ys, context_length, device)
    y = torch.tensor(ys, dtype=torch.long)
    V = torch.from_numpy(np.stack(vs, axis=0))
    n_multi = int((V.sum(dim=-1) > 1).sum())
    if n_multi:
        print(f"[info] {n_multi}/{n_seg} segment(s) mark several acceptable LoAs; "
              f"read set-acc/set-MAE rather than acc/MAE.")

    n_val = max(1, round(args.val_frac * n_seg))
    if n_val >= n_seg:
        raise ValueError(f"--val-frac {args.val_frac} leaves no training segments (total={n_seg}).")
    Zpool, ypool, Vpool = Z[: n_seg - n_val], y[: n_seg - n_val], V[: n_seg - n_val]
    Zval,  yval,  Vval  = Z[n_seg - n_val:], y[n_seg - n_val:], V[n_seg - n_val:]
    print(f"[split] temporal: pool={len(ypool)} earliest segments, val={n_val} latest segments")

    base = evaluate(model.head, Zval, yval, Vval, head_type)
    print(f"[baseline] population head: acc={base['acc']:.3f} set-acc={base['set_acc']:.3f} "
          f"macro-F1={base['f1']:.3f} MAE={base['mae']:.3f} set-MAE={base['set_mae']:.3f} "
          f"QWK={base['qwk']:.3f}")

    ks = pick_sweep_points(len(ypool), args.max_points)
    results = []
    for k in ks:
        head = fine_tune_head(model.head, Zpool[:k], Vpool[:k],
                              lr=args.lr, l2sp=args.l2sp, epochs=args.epochs,
                              head_type=head_type)
        m = evaluate(head, Zval, yval, Vval, head_type)
        results.append(m)
        print(f"[k={k:3d}] ({k * SEGMENT_SECONDS / 60.0:5.1f} min) acc={m['acc']:.3f} "
              f"set-acc={m['set_acc']:.3f} macro-F1={m['f1']:.3f} MAE={m['mae']:.3f} "
              f"set-MAE={m['set_mae']:.3f} QWK={m['qwk']:.3f}")

    out_png = pathlib.Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_png.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_train_segments", "minutes",
                    "val_accuracy", "val_macro_f1", "val_mae", "val_qwk",
                    "val_set_accuracy", "val_set_mae",
                    "baseline_accuracy", "baseline_macro_f1", "baseline_mae", "baseline_qwk",
                    "baseline_set_accuracy", "baseline_set_mae",
                    "head_type", "n_val", "n_multi_label", "lr", "l2sp", "epochs"])
        for k, m in zip(ks, results):
            w.writerow([k, round(k * SEGMENT_SECONDS / 60.0, 2),
                        m["acc"], m["f1"], m["mae"], m["qwk"],
                        m["set_acc"], m["set_mae"],
                        base["acc"], base["f1"], base["mae"], base["qwk"],
                        base["set_acc"], base["set_mae"],
                        head_type, n_val, n_multi, args.lr, args.l2sp, args.epochs])

    plot_curve(ks, [m["acc"] for m in results], [m["f1"] for m in results],
               base["acc"], n_val, out_png)
    print(f"[OK] plot  -> {out_png}")
    print(f"[OK] table -> {out_csv}")


if __name__ == "__main__":
    main()
