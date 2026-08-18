"""Stage 3 — L2-SP prior-precision sweep: tau x K over all 12 drivers.

Produces ``selected_tau.json`` (the single tau every downstream L2-SP and ANIL
run uses) and the per-driver personalization learning curves.

THE L2-SP ARM HAS EXACTLY ONE HYPERPARAMETER
--------------------------------------------
At convergence the adapted head theta* depends only on (Z, V, theta_pop, lambda)
— ``steps`` and ``lr`` decide only WHETHER the MAP is reached, not where it is,
and ``head_adapt`` reaches it (|grad| ~ 1e-7, re-checked every call). So the
whole search space is tau. That is worth stating in the write-up next to the
ANIL arm's five knobs: "equal tuning budget" cannot mean equal config counts
when one arm's space is one-dimensional by construction.

WHY THE SWEEP IS 2-D AND NOT "TUNE tau AT ONE K"
------------------------------------------------
The empirically best tau drifts with K (ordinary ridge behaviour). Anchoring at
one K0 therefore makes the SHAPE of the quality-vs-K curve partly an artifact of
where the anchor was placed — and that shape is what research question (b)
reads. Since one tau is committed to for comparability, the right tau is the one
that makes the whole curve good, so tau is selected on the curve, not at a point.

WHY EACH DRIVER USES ITS OWN FOLD CHECKPOINT
--------------------------------------------
Driver d is adapted from ``pop_heldout_d.pt`` — the stage-2 model trained on the
OTHER 11 drivers. Using one model trained on all 12 would start every curve from
a backbone that had already seen that driver: the curves would be optimistic AND
tau would be tuned for a leaky regime. That is a representation-level leak, far
worse than the (accepted, disclosed) one-scalar leak from selecting tau on
aggregate across drivers.

AGGREGATION
-----------
Average over K WITHIN each driver first, then across drivers. Drivers contribute
94-136 segments, so a flat mean over all (driver, K) points would silently weight
the long sessions. Only K <= ``--k-cap`` (default 60, ~25 min of labelling) enters
the average: beyond that is not a deployable condition and must not be allowed to
choose tau. Curves are still plotted over the full range.

TIE-BREAK, fixed in advance so it is not a post-hoc choice: among tau values
within one between-driver SE of the best mean, take the LARGEST tau. It degrades
more gracefully at low K and is the more conservative claim.

Usage::

    python -m ProVoice.training_scripts.sweep_l2sp_tau \\
        --in data/labeled_data.jsonl --ckpt-dir trained_models/lodo
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ProVoice.models.xlstm_model import load_checkpoint
from ProVoice.models.head_adapt import install_fcd_head
from ProVoice.models.head_adapt import (
    adapt_head, DEFAULT_ADAPT_LR, DEFAULT_ADAPT_STEPS,
)
from ProVoice.models.train_XLSTM import iter_jsonl, normalize_row
from ProVoice.training_scripts.folds import ALL_PIDS

# Imported, not re-implemented: the curve must be drawn by the same code that
# draws the single-driver curves in scripts/sweep_train_frac.py.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "scripts"))
from sweep_train_frac import (  # noqa: E402
    build_segments, embed_segments, evaluate, pick_sweep_points,
)

TAU_GRID = (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0)
GRAD_NORM_WARN = 1e-3


def sweep_driver(model, arch, df: pd.DataFrame, taus: List[float], val_frac: float,
                 max_points: int, steps: int, lr: float, device: str,
                 k_cap: int = 60) -> List[dict]:
    """Full tau x K grid for ONE driver. The backbone runs once, not once per tau."""
    head_type = arch.get("head_type", "softmax")
    gids, Xs, vs = build_segments(df, window_seconds=arch.get("window_seconds"),
                                  resample_hz=arch.get("resample_hz"))
    n_seg = len(gids)
    if n_seg < 3:
        return []
    # The BACKBONE runs on `device` — that is the expensive part and the only
    # thing worth a GPU here. `embed_segments` then returns the embeddings on the
    # CPU (it ends in `.cpu()`), so from this point on everything is CPU-side.
    Z = embed_segments(model, Xs, vs, arch["context_length"], device)
    V = torch.from_numpy(np.stack(vs, axis=0))

    # The head must follow the embeddings, or every adapt/evaluate below fails
    # with a device mismatch the moment `device` is not "cpu". This bit was
    # invisible until the first GPU run: on a CPU-only box `device` IS "cpu",
    # so the head and the embeddings agreed by accident.
    #
    # CPU is also the right place for the adaptation itself, not a concession:
    # it is 2000 steps on a (K x 64) tensor with a ~260-parameter head, entirely
    # kernel-launch-bound — measured at ~1.05 s whether K is 5 or 99, so CUDA
    # would add launch overhead and remove nothing.
    pop_head = model.head.to(Z.device)

    n_val = max(1, round(val_frac * n_seg))
    Zpool, Vpool = Z[: n_seg - n_val], V[: n_seg - n_val]
    Zval, Vval = Z[n_seg - n_val:], V[n_seg - n_val:]

    base = evaluate(pop_head, Zval, Vval, head_type)   # K=0 floor, same tail
    out = []
    for tau in taus:
        for k in pick_sweep_points(len(Vpool), max_points, k_cap):
            head, info = adapt_head(pop_head, Zpool[:k], Vpool[:k], tau=tau,
                                    head_type=head_type, steps=steps, lr=lr)
            m = evaluate(head, Zval, Vval, head_type)
            out.append({"tau": tau, "k": k, "l2sp": info["l2sp"],
                        # PROVENANCE. --embed-fcd only widens the head in memory;
                        # the on-disk checkpoints stay narrow, so without this
                        # column an FCD run and a plain one are indistinguishable
                        # after the fact. Read off the head itself rather than the
                        # flag, so the row cannot disagree with what was adapted.
                        "head_in": int(pop_head.in_features),
                        "embed_fcd": int(pop_head.in_features
                                         > model.embedding_dim),
                        "grad_norm": info["grad_norm"], "n_val": n_val,
                        "set_mae": m["mae"], "set_acc": m["acc"],
                        "set_qwk": m["qwk"], "set_macro_f1": m["f1"],
                        "base_set_mae": base["mae"], "base_set_acc": base["acc"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="trained_models/lodo",
                    help="Per-fold checkpoint directory; expects <prefix><pid>.pt.")
    ap.add_argument("--ckpt-prefix", dest="ckpt_prefix", default="pop_heldout_",
                    help="Filename prefix for the per-driver checkpoints. 'pop_heldout_' is "
                         "run_lodo_population's output (the L2-SP arm); pass 'anil_heldout_' "
                         "to draw the same curve from run_lodo_anil's meta-inits. The two "
                         "arms differ ONLY in which directory/prefix is read — same tau, "
                         "same adaptation, same tail — which is what makes the curves "
                         "comparable.")
    ap.add_argument("--outdir", default="results/l2sp_sweep")
    ap.add_argument("--taus", default=",".join(str(t) for t in TAU_GRID))
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.3,
                    help="MUST match stage 2's --val-frac: the K=0 floor and these "
                         "curves have to be scored on the same segments.")
    ap.add_argument("--embed-fcd", dest="embed_fcd", action="store_true",
                    help="Give the adapted head direct access to the task: it sees "
                         "[z_64 | FCD_12] instead of z_64 alone. The backbone is untouched, "
                         "so no retraining is implied; the appended block is initialized AND "
                         "L2-SP-anchored at zero, so K=0 reproduces the population head "
                         "exactly (checked at startup). MUST match the other arm: both arms "
                         "have to adapt the identical object or the comparison is confounded.")
    ap.add_argument("--k-cap", dest="k_cap", type=int, default=60,
                    help="Only K <= this enters the tau selection average (~25 min of "
                         "labelling). Curves are still recorded and plotted in full.")
    ap.add_argument("--max-points", dest="max_points", type=int, default=0,
                    help="Ceiling on the number of K points, NOT a target. 0 (default) "
                         "uses the whole fixed grid (sweep_train_frac.K_GRID_BASE): every "
                         "driver is evaluated at the SAME K values below --k-cap, which is "
                         "what makes the per-driver curves averageable at a given K rather "
                         "than only interpolatable. A smaller value thins the grid evenly.")
    ap.add_argument("--steps", type=int, default=DEFAULT_ADAPT_STEPS)
    ap.add_argument("--lr", type=float, default=DEFAULT_ADAPT_LR)
    ap.add_argument("--pids", default="")
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    want = set(p.strip() for p in args.pids.split(",") if p.strip()) or None
    ckpt_dir = pathlib.Path(args.ckpt_dir)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MEMORY: never hold the whole file. data/labeled_data.jsonl parses to a
    # MEASURED 4.0 GB as raw dicts; one driver's normalized rows are ~1/12th of
    # that with ~25 of the 73 keys kept, and are dropped before the next driver.
    src = pathlib.Path(args.in_jsonl)
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    all_rows: List[dict] = []
    worst_grad = 0.0
    for pid in ALL_PIDS:
        if want and pid not in want:
            continue
        ckpt = ckpt_dir / f"{args.ckpt_prefix}{pid}.pt"
        if not ckpt.exists():
            print(f"[skip] {pid}: no {ckpt} — run the matching LODO stage first")
            continue
        drows = [normalize_row(r) for r in iter_jsonl(src)
                 if str(r.get("participantid", "")) == pid]
        if not drows:
            print(f"[skip] {pid}: no rows in {src}")
            continue
        model, arch = load_checkpoint(str(ckpt))
        # Widen BEFORE any embedding or adaptation: embed_segments infers the
        # augmentation from the head's width, so this one call switches the whole
        # driver's sweep over consistently.
        install_fcd_head(model, args.embed_fcd)
        model.to(device).eval()
        df = pd.DataFrame(drows)
        del drows
        print(f"[{pid}] {len(taus)} tau x K grid from {ckpt.name}", flush=True)
        res = sweep_driver(model, arch, df, taus, args.val_frac,
                           args.max_points, args.steps, args.lr, device,
                           k_cap=args.k_cap)
        del df, model
        for r in res:
            r["pid"] = pid
            worst_grad = max(worst_grad, r["grad_norm"])
        all_rows.extend(res)
        if res:
            print(f"      floor(K=0) set-MAE={res[0]['base_set_mae']:.3f} "
                  f"set-acc={res[0]['base_set_acc']:.3f} (n_val={res[0]['n_val']})")

    if not all_rows:
        raise SystemExit("nothing swept — is stage 2 done?")
    print(f"[converge] worst |grad| over the whole sweep = {worst_grad:.2e} "
          f"({'OK' if worst_grad < GRAD_NORM_WARN else 'HIGH — raise --steps or --lr'})")

    out_csv = outdir / "l2sp_tau_sweep.csv"
    cols = ["pid", "tau", "k", "l2sp", "n_val", "set_mae", "set_acc", "set_qwk",
            "set_macro_f1", "base_set_mae", "base_set_acc", "grad_norm",
            "head_in", "embed_fcd"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r[c] for c in cols} for r in all_rows])

    summarize(pd.DataFrame(all_rows), taus, args.k_cap, outdir)


def summarize(df: pd.DataFrame, taus: List[float], k_cap: int, outdir: pathlib.Path) -> None:
    """Within-driver mean over K <= k_cap, then across drivers; pick tau."""
    sub = df[df["k"] <= k_cap]
    if sub.empty:
        raise SystemExit(f"no sweep points with K <= {k_cap}")
    per_driver = sub.groupby(["tau", "pid"])["set_mae"].mean().reset_index()
    agg = per_driver.groupby("tau")["set_mae"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    floor = df.groupby("pid")["base_set_mae"].first().mean()

    print(f"\n{'tau':>8} {'mean set-MAE':>13} {'sd':>7} {'se':>7} {'drivers':>8}"
          f"   (K <= {k_cap}, averaged within driver first)")
    for _, r in agg.iterrows():
        print(f"{r['tau']:8.3g} {r['mean']:13.3f} {r['std']:7.3f} {r['se']:7.3f} "
              f"{int(r['count']):8d}")
    print(f"{'floor':>8} {floor:13.3f}    (unadapted population model, K=0)")

    best = agg.loc[agg["mean"].idxmin()]
    within = agg[agg["mean"] <= best["mean"] + best["se"]]
    pick = within.loc[within["tau"].idxmax()]
    if float(pick["tau"]) != float(best["tau"]):
        print(f"[tie-break] {len(within)} tau within 1 SE of the minimum; "
              f"taking the largest (most regularized): {pick['tau']:g} over {best['tau']:g}")

    sel = {
        "tau": float(pick["tau"]),
        "mean_set_mae": float(pick["mean"]),
        "between_driver_sd": float(pick["std"]),
        "se": float(pick["se"]),
        "n_drivers": int(pick["count"]),
        "k_cap": k_cap,
        "unadapted_floor_set_mae": float(floor),
        "beats_floor": bool(pick["mean"] < floor),
        # Everything that changes what tau MEANS. val_frac decides which
        # segments the curves were scored on; embed_fcd decides what object was
        # adapted. A tau is only transferable to another run that matches both.
        "val_frac": args.val_frac,
        "embed_fcd": int(args.embed_fcd),
        "adapt_steps": DEFAULT_ADAPT_STEPS, "adapt_lr": DEFAULT_ADAPT_LR,
        "note": ("Single prior precision for ALL K, all drivers and BOTH study arms. "
                 "lambda = tau/(2K) is derived per adaptation by head_adapt. Selected on "
                 "mean set-MAE over K <= k_cap, averaged within driver then across "
                 "drivers; ties broken toward the larger tau."),
    }
    (outdir / "selected_tau.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(f"\n[selected] tau={sel['tau']:g}  mean set-MAE={sel['mean_set_mae']:.3f} "
          f"vs floor {floor:.3f}  -> {'BEATS' if sel['beats_floor'] else 'DOES NOT BEAT'} the floor")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    curves = df.groupby(["tau", "k"])["set_mae"].mean().reset_index()
    cmap = plt.get_cmap("viridis")
    for i, tau in enumerate(sorted(curves["tau"].unique())):
        c = curves[curves["tau"] == tau].sort_values("k")
        ax.plot(c["k"], c["set_mae"], marker="o", ms=3, color=cmap(i / max(1, len(taus) - 1)),
                lw=2.0 if float(tau) == sel["tau"] else 1.0,
                label=f"tau={tau:g}" + ("  (selected)" if float(tau) == sel["tau"] else ""))
    ax.axhline(floor, ls="--", c="0.35", lw=1.2, label="unadapted population (K=0)")
    ax.axvline(k_cap, ls=":", c="0.6", lw=1.0)
    ax.annotate(f"tau chosen on K <= {k_cap}", xy=(k_cap, ax.get_ylim()[1]),
                xytext=(-6, -12), textcoords="offset points", ha="right", fontsize=8, color="0.4")
    ax.set_xlabel("personalization segments K  (20 s each)")
    ax.set_ylabel("set-MAE (lower is better)")
    ax.set_title(f"L2-SP personalization vs. K, averaged over "
                 f"{int(agg['count'].max())} drivers (LODO)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    png = outdir / "l2sp_tau_sweep.png"
    fig.savefig(png, dpi=150)
    print(f"[OK] table -> {outdir / 'l2sp_tau_sweep.csv'}\n[OK] plot  -> {png}")


if __name__ == "__main__":
    main()
