r"""Stage 1 — population-model hyperparameter sweep.

    dropout in {0.10, 0.15, 0.20}  x  lr in {1e-3, 2e-3}
    x  6 fixed validation folds of 2 drivers  x  5 seeds   =  180 runs

Writes ``selected_population.json``: the winning ``(dropout, lr)`` and **E\***,
the epoch count stage 2 then trains for with no validation set at all.

WHAT THIS IS FOR
----------------
The population model is the initialization BOTH study arms adapt from (the ANIL
arm warm-starts from it), so it is tuned once, here, and then frozen. Tuning it
per arm would make the arms differ by more than their meta-objective, which is
the one thing the comparison is supposed to isolate.

WHY 5 SEEDS RATHER THAN A BIGGER GRID
-------------------------------------
Validation set-MAE swings by roughly +/-0.3 between adjacent epochs on ~250
validation segments (measured), while the differences between neighbouring
configurations are far smaller. A single run per configuration cannot rank them.
Compute is better spent averaging that noise away than on grid points: with 12
drivers the validation signal supports separating a handful of configurations,
not dozens.

WHY THE FOLDS ROTATE
--------------------
Every driver serves as validation in exactly one fold (see ``folds.py``), so the
chosen configuration rests on all 12 drivers rather than on whichever 2 were
picked. The cost is that hyperparameters have then seen every driver, including
each LODO test driver — a deliberate, disclosed trade: it is one configuration
choice, shared identically by both arms, so it shifts the absolute numbers but
cancels in the arm comparison. State it in the write-up; do not quietly rely on
it.

HOW E* IS EXTRACTED
-------------------
NOT the argmin epoch — with SE ~0.05 per epoch, the argmin over ~100 epochs is
mostly noise, and taking it would also make E* systematically late (a lucky
epoch is more likely to be found the longer you look). Instead: smooth each
run's validation curve with a centred moving average, take that curve's minimum,
then the MEDIAN across the 30 runs of the winning configuration. Median rather
than mean because a run that early-stops early truncates its curve.

RESUMABILITY
------------
180 trainings is long enough that the process will be interrupted. Every
completed run appends one row to the results CSV, and a restart skips any
(dropout, lr, fold, seed) already present. Delete the CSV to start over.

Usage::

    python -m ProVoice.training_scripts.sweep_population_hparams \\
        --in data/labeled_data.jsonl --outdir results/pop_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from ProVoice.training_scripts.folds import VALIDATION_FOLDS, train_pids_for_validation_fold

DROPOUTS = (0.10, 0.15, 0.20)
LRS = (1e-3, 2e-3)
SEEDS = (0, 1, 2, 3, 4)
SMOOTH_WINDOW = 5          # epochs, centred — see "HOW E* IS EXTRACTED"

RESULTS_COLUMNS = [
    "dropout", "lr", "fold", "val_pids", "seed",
    "best_set_mae",          # the run's raw minimum (what train_XLSTM selects on)
    "smoothed_best_set_mae", # minimum of the smoothed curve — the ranking quantity
    "best_epoch_smoothed",   # argmin of the smoothed curve — diagnostic only
    "best_epoch_1se",        # EARLIEST epoch within 1 SE of that minimum — E* comes from this
    "epochs_run",            # < --epochs when early stopping fired
    "set_acc_at_best", "qwk_at_best", "val_n",
]


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, shrinking the window at the edges.

    Edge-shrinking (rather than padding) matters: padding with the endpoint
    value would flatten the curve exactly where early-stopped runs end, biasing
    their smoothed minimum toward the final epoch.
    """
    if len(y) == 0:
        return y
    w = max(1, min(int(window), len(y)))
    half = w // 2
    out = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = float(np.mean(y[lo:hi]))
    return out


def read_done(path: pathlib.Path) -> set:
    """Already-completed ``(dropout, lr, fold, seed)`` keys, for resuming."""
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((float(row["dropout"]), float(row["lr"]),
                          int(row["fold"]), int(row["seed"])))
            except (KeyError, ValueError):
                continue        # a torn final line from an interrupted run
    return done


def run_one(in_jsonl: str, dropout: float, lr: float, val_pids: List[str], seed: int,
            epochs: int, patience: int, min_delta: float, workdir: pathlib.Path,
            extra: List[str]) -> Optional[Dict[str, float]]:
    """One training run. Returns its curve summary, or None if it failed.

    Runs train_XLSTM as a SUBPROCESS rather than importing it: each run needs a
    fresh process-global torch RNG state and a clean CUDA allocator, and a crash
    in run 97 of 180 should cost that run, not the sweep.
    """
    tag = f"d{dropout}_lr{lr}_f{'-'.join(val_pids)}_s{seed}"
    ckpt = workdir / f"ckpt_{tag}.pt"          # written, then discarded — stage 2 retrains
    mcsv = workdir / f"metrics_{tag}.csv"
    cmd = [sys.executable, "-m", "ProVoice.models.train_XLSTM",
           "--in", in_jsonl, "--out", str(ckpt), "--loss", "corn",
           "--val-pids", ",".join(val_pids), "--metrics-csv", str(mcsv),
           "--dropout", str(dropout), "--lr", str(lr), "--seed", str(seed),
           "--epochs", str(epochs), "--patience", str(patience),
           "--min-delta", str(min_delta)] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [FAIL] {tag} (exit {proc.returncode})")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
        return None
    if not mcsv.exists():
        print(f"  [FAIL] {tag}: no metrics CSV written")
        return None

    rows = list(csv.DictReader(mcsv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print(f"  [FAIL] {tag}: empty metrics CSV")
        return None
    mae = np.array([float(r["set_mae"]) for r in rows])
    sm = smooth(mae, SMOOTH_WINDOW)
    j = int(np.argmin(sm))
    # ONE-SE RULE. The argmin of a smoothed curve sits wherever a flat basin
    # happens to dip lowest, which on a noisy curve is late as often as not —
    # and E* is applied WITHOUT early stopping in the LODO run, where nothing
    # would catch an epoch count past the overfitting knee. So take the EARLIEST
    # epoch that is statistically indistinguishable from the minimum instead.
    #
    # The noise scale is estimated from the run's own residuals around its
    # smoothed curve; the smoothed value averages ~SMOOTH_WINDOW points, so its
    # standard error is sigma/sqrt(window). Erring early costs a little
    # under-training; erring late costs overfitting that no later stage detects.
    resid = mae - sm
    sigma = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0
    se_sm = sigma / np.sqrt(min(SMOOTH_WINDOW, len(mae)))
    within = np.flatnonzero(sm <= sm[j] + se_sm)
    j1 = int(within[0]) if within.size else j
    ckpt.unlink(missing_ok=True)               # only the curve is needed downstream
    return {
        "best_set_mae": float(mae.min()),
        "smoothed_best_set_mae": float(sm[j]),
        "best_epoch_smoothed": int(rows[j]["epoch"]),
        "best_epoch_1se": int(rows[j1]["epoch"]),
        "epochs_run": len(rows),
        "set_acc_at_best": float(rows[j]["set_acc"]),
        "qwk_at_best": float(rows[j]["qwk"]),
        "val_n": int(rows[j]["val_n"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--outdir", default="results/pop_sweep",
                    help="Results CSV, per-run metric curves and selected_population.json.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", dest="min_delta", type=float, default=0.0)
    ap.add_argument("--dropouts", default=",".join(str(d) for d in DROPOUTS))
    ap.add_argument("--lrs", default=",".join(str(l) for l in LRS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--folds", default="",
                    help="Comma-separated fold indices to run (default: all 6). For "
                         "splitting the sweep across machines.")
    ap.add_argument("--trainer-arg", dest="trainer_args", action="append", default=[],
                    help="Extra flag passed through to train_XLSTM, repeatable, e.g. "
                         "--trainer-arg=--grad-clip --trainer-arg=0.5")
    args = ap.parse_args()

    dropouts = [float(x) for x in args.dropouts.split(",") if x.strip()]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    fold_idx = ([int(x) for x in args.folds.split(",") if x.strip()]
                if args.folds else list(range(len(VALIDATION_FOLDS))))

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "runs"; workdir.mkdir(exist_ok=True)
    results_csv = outdir / "sweep_results.csv"
    done = read_done(results_csv)
    if done:
        print(f"[resume] {len(done)} run(s) already in {results_csv}; they will be skipped")

    total = len(dropouts) * len(lrs) * len(fold_idx) * len(seeds)
    print(f"[plan] {len(dropouts)} dropout x {len(lrs)} lr x {len(fold_idx)} fold x "
          f"{len(seeds)} seed = {total} runs (<= {args.epochs} epochs each)")
    for i in fold_idx:
        print(f"  fold {i}: val={list(VALIDATION_FOLDS[i])} "
              f"train={train_pids_for_validation_fold(VALIDATION_FOLDS[i])}")

    new = not results_csv.exists()
    fh = results_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    if new:
        writer.writerow(RESULTS_COLUMNS)
        fh.flush()

    n = 0
    for dropout in dropouts:
        for lr in lrs:
            for i in fold_idx:
                val_pids = list(VALIDATION_FOLDS[i])
                for seed in seeds:
                    n += 1
                    key = (dropout, lr, i, seed)
                    if key in done:
                        continue
                    print(f"[{n}/{total}] dropout={dropout} lr={lr:g} "
                          f"fold={i}{val_pids} seed={seed}", flush=True)
                    r = run_one(args.in_jsonl, dropout, lr, val_pids, seed,
                                args.epochs, args.patience, args.min_delta,
                                workdir, args.trainer_args)
                    if r is None:
                        continue
                    writer.writerow([dropout, lr, i, "|".join(val_pids), seed] +
                                    [r[c] for c in RESULTS_COLUMNS[5:]])
                    fh.flush()
                    print(f"      set-MAE raw={r['best_set_mae']:.3f} "
                          f"smoothed={r['smoothed_best_set_mae']:.3f} "
                          f"@epoch {r['best_epoch_smoothed']} "
                          f"({r['epochs_run']} epochs run)")
    fh.close()
    summarize(results_csv, outdir)


def summarize(results_csv: pathlib.Path, outdir: pathlib.Path) -> None:
    """Rank configurations, pick the winner and E*, write selected_population.json."""
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("[summary] no completed runs yet")
        return
    by_cfg: Dict[Tuple[float, float], List[dict]] = {}
    for r in rows:
        by_cfg.setdefault((float(r["dropout"]), float(r["lr"])), []).append(r)

    print(f"\n{'dropout':>8} {'lr':>7} {'n':>4} {'mean':>7} {'sd':>6} {'se':>6} "
          f"{'E*(1se)':>8} {'argmin':>7} {'IQR':>11}   (smoothed val set-MAE)")
    table = []
    for (dropout, lr), rs in sorted(by_cfg.items()):
        v = np.array([float(r["smoothed_best_set_mae"]) for r in rs])
        e_arg = np.array([float(r["best_epoch_smoothed"]) for r in rs])
        # Older CSVs (written before the 1-SE rule) lack the column; fall back to
        # the argmin so a partially-completed sweep still summarizes rather than
        # crashing — but say so, because the two are not interchangeable.
        if all("best_epoch_1se" in r and r["best_epoch_1se"] != "" for r in rs):
            e_1se = np.array([float(r["best_epoch_1se"]) for r in rs])
        else:
            print("[warn] some rows predate the 1-SE rule; falling back to argmin epochs "
                  "for this config. Delete sweep_results.csv and re-run for a clean E*.")
            e_1se = e_arg
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        q1, q3 = np.percentile(e_1se, [25, 75])
        table.append({"dropout": dropout, "lr": lr, "n": len(v),
                      "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "se": se, "e_star": int(np.median(e_1se)),
                      "e_argmin": int(np.median(e_arg)),
                      "e_iqr": (int(q1), int(q3))})
        print(f"{dropout:8.2f} {lr:7.0e} {len(v):4d} {v.mean():7.3f} "
              f"{table[-1]['sd']:6.3f} {se:6.3f} {table[-1]['e_star']:8d} "
              f"{table[-1]['e_argmin']:7d} {str(table[-1]['e_iqr']):>11}")

    best = min(table, key=lambda t: t["mean"])
    # Tie-break toward MORE regularization: within one standard error of the
    # winner, prefer the larger dropout (and then the smaller lr). Fixed in
    # advance so it is not a post-hoc choice, and it errs toward the model that
    # generalizes rather than the one that won a noisy comparison.
    within = [t for t in table if t["mean"] <= best["mean"] + (best["se"] if best["se"] == best["se"] else 0.0)]
    pick = max(within, key=lambda t: (t["dropout"], -t["lr"]))
    if pick is not best:
        print(f"\n[tie-break] {len(within)} config(s) within 1 SE of the minimum "
              f"({best['mean']:.3f} +/- {best['se']:.3f}); taking the most regularized")

    sel = {
        "dropout": pick["dropout"], "lr": pick["lr"], "epochs": pick["e_star"],
        "loss": "corn",
        "epochs_rule": "median over runs of the earliest epoch within 1 SE of the smoothed minimum",
        "epochs_argmin_median": pick["e_argmin"],
        "epochs_1se_iqr": list(pick["e_iqr"]),
        # The LODO run trains on 11 drivers, this sweep on 10 -- ~10% more
        # segments, so ~10% more optimizer STEPS at the same epoch count. Recorded
        # so the transfer is visible rather than assumed.
        "n_train_drivers_at_selection": 10,
        "selected_on": "mean smoothed validation set-MAE over 6 rotating folds x seeds",
        "mean_smoothed_val_set_mae": pick["mean"],
        "between_run_sd": pick["sd"], "se": pick["se"], "n_runs": pick["n"],
        "note": ("E* is the median across runs of the EARLIEST epoch within 1 SE of the "
                 "smoothed minimum, not of the argmin: run_lodo_population applies it "
                 "with NO validation set and NO early stopping, so an E* past the "
                 "overfitting knee would go undetected in all 12 folds. Erring early "
                 "costs mild under-training; erring late costs overfitting nothing "
                 "catches. Compare epochs_argmin_median -- a large gap means a flat "
                 "basin (E* barely matters) and a small one means a sharp optimum "
                 "(check epochs_1se_iqr for stability). These hyperparameters have seen "
                 "every driver (rotating folds) -- disclosed trade, identical for both "
                 "arms."),
    }
    out = outdir / "selected_population.json"
    out.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(f"\n[selected] dropout={pick['dropout']} lr={pick['lr']:g} E*={pick['e_star']} "
          f"(1-SE rule; argmin would give {pick['e_argmin']}, IQR {pick['e_iqr']})")
    print(f"           mean smoothed val set-MAE {pick['mean']:.3f}")
    if pick["e_iqr"][1] - pick["e_iqr"][0] > max(5, 0.5 * pick["e_star"]):
        print(f"[warn] E* varies widely across runs (IQR {pick['e_iqr']}) — the optimum is "
              f"not well determined. Consider more seeds before committing to it.")
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
