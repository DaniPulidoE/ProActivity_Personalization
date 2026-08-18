r"""Within-driver (subject-dependent) diagnostic sweep: CORN vs CE.

    2 losses  x  2 (dropout, lr) configurations  x  5 seeds,  on ONE fixed split:
    per driver, the earliest 80 % of their segments train, the latest 20 % val.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
``sweep_population_hparams`` measured every configuration on HELD-OUT DRIVERS and
found that nothing beats a constant prediction: 720 runs, best set-MAE 1.509
against a constant floor of 1.161, QWK ~0.03, and the CORN stages ending WORSE
than their untrained initialization. Two very different situations produce that
same table:

    (a) the signal is real but driver-specific, and does not transfer; or
    (b) the features carry no usable LoA signal, or the training path is broken.

A cross-driver split cannot separate them. This one can, by handing the model the
one thing the population split withholds -- the driver's own history. If a model
beats its floor here, (a) stands and the population checkpoint is legitimately
"an initialization, not a result". If it still cannot, (b) is live and the
personalization arms would be built on sand.

THIS IS NOT A SELECTION INSTRUMENT. The winner here must not become the shipped
configuration: every driver is in both halves, driver identity is ~68 % decodable
from the state features alone, and knowing the driver is worth ~0.23 MAE against
~0.06 for knowing the function. A model can therefore score well here by learning
who is driving rather than when they want autonomy. That is why the ranking
column is the margin against the PER-DRIVER constant floor, not the global one --
see ``train_XLSTM.per_driver_constant_baseline``.

WHY BOTH LOSSES
---------------
Not to re-decide the loss: CORN is fixed by the design (the Laplace UQ layer
needs its closed-form Hessian, soft-CORN keeps the 6.8 % multi-label windows CE
would force us to drop, and conditional-subset training is what localizes
per-driver updates). It is a bug check. ``soft_corn_loss`` is custom code, not
the reference ``coral_pytorch`` implementation, and in the cross-driver sweep the
CORN stages moved AWAY from their init at every window while CE at its best
configuration moved toward it. On data where the driver is known, a working
soft-CORN must be able to fit. If CE learns here and soft-CORN does not, that is
an implementation defect that would otherwise propagate into both study arms and
into the Laplace layer.

Usage::

    python -m ProVoice.training_scripts.sweep_within_driver \
        --in data/labeled_data.jsonl --cache data/cache/segments_w10_hz10.npz \
        --outdir results/within_driver --jobs 4 --threads-per-job 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Imported, never re-implemented: the curve -> (E*, reported metrics) reduction
# must mean the same thing here as in the cross-driver sweep, or the two tables
# cannot be placed side by side in the write-up.
from ProVoice.training_scripts.sweep_population_hparams import curve_stats

LOSSES: Tuple[str, ...] = ("corn", "ce")
# Each loss's own winner from the cross-driver sweep (CORN: 0.15/1e-3, CE:
# 0.10/1e-3), so the head comparison can be read both at each head's best and at
# a shared configuration. Small on purpose: this is a go/no-go diagnostic, not a
# search, and the question it asks does not turn on fine hyperparameter ranking.
CONFIGS: Tuple[Tuple[float, float], ...] = ((0.15, 1e-3), (0.10, 1e-3))
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

ID_COLUMNS = ["loss", "dropout", "lr", "window_seconds", "seed", "split_mode", "val_frac"]
RESULTS_COLUMNS = ID_COLUMNS + [
    "best_set_mae",           # raw per-epoch minimum
    "smoothed_best_set_mae",  # minimum of the smoothed curve -- the ranking quantity
    "best_epoch_smoothed", "best_epoch_1se", "metrics_epoch", "epochs_run",
    "set_acc_at_best", "qwk_at_best", "val_n",
    "mae_argmax", "acc_argmax", "mae_median", "acc_median",
    "pred_loa0", "pred_loa1", "pred_loa2", "pred_loa3", "pred_loa4",
    # Both floors, written by the trainer into every metrics row. `const_*` is
    # the global constant; `pdconst_*` is the per-driver constant fitted on each
    # driver's own train prefix -- the one that binds here.
    "const_set_mae", "const_loa_mae", "pdconst_set_mae", "pdconst_set_acc",
    "pdconst_oracle_set_mae", "pdconst_n_drivers",
    # Untrained-init set-MAE. "Did training move the model at all" is the first
    # question this sweep exists to answer, so it is a column, not a footnote.
    "init_set_mae",
]
assert len(set(RESULTS_COLUMNS)) == len(RESULTS_COLUMNS), "duplicate column"

# Columns copied straight off the last metrics row rather than reduced from the
# curve: each is constant within a run, and the trainer repeats it per row.
_RUN_CONSTANTS = ("const_set_mae", "const_loa_mae", "pdconst_set_mae", "pdconst_set_acc",
                  "pdconst_oracle_set_mae", "pdconst_n_drivers", "init_set_mae")


def run_tag(loss: str, dropout: float, lr: float, seed: int, window_seconds: float) -> str:
    return f"wd_{loss}_d{dropout}_lr{lr}_w{window_seconds:g}_s{seed}"


def read_done(path: pathlib.Path) -> set:
    """(loss, dropout, lr, window, seed) already present, so a restart resumes."""
    if not path.exists():
        return set()
    out = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                out.add((r["loss"], float(r["dropout"]), float(r["lr"]),
                         float(r["window_seconds"]), int(r["seed"])))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def run_one(in_jsonl: str, cache: str, loss: str, dropout: float, lr: float, seed: int,
            epochs: int, patience: int, min_select_epoch: int, window_seconds: float,
            val_frac: float, workdir: pathlib.Path,
            threads: int = 0) -> Optional[Dict[str, float]]:
    """One training run, as a SUBPROCESS.

    Same reason as the cross-driver sweep: each run needs a clean process for its
    own seeding and thread pool, and a crash in one must not take the sweep down.
    """
    tag = run_tag(loss, dropout, lr, seed, window_seconds)
    ckpt = workdir / "ckpt" / f"{tag}.pt"
    mcsv = workdir / "runs" / f"{tag}.csv"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    mcsv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "ProVoice.models.train_XLSTM",
           "--in", in_jsonl, "--out", str(ckpt), "--loss", loss,
           "--split-mode", "within-driver", "--val-frac", str(val_frac),
           "--metrics-csv", str(mcsv), "--dropout", str(dropout), "--lr", str(lr),
           "--seed", str(seed), "--epochs", str(epochs), "--patience", str(patience),
           "--min-select-epoch", str(min_select_epoch),
           "--window-seconds", str(window_seconds)]
    if cache:
        cmd += ["--cache", cache]

    env = dict(os.environ)
    if threads > 0:
        # Set in the ENVIRONMENT, not via a trainer flag: torch reads these when
        # it is imported, so a flag parsed in main() would be too late to shrink
        # a pool that has already been built.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            env[var] = str(threads)

    print(f"[run] {tag}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"  [FAIL] {tag} (exit {proc.returncode})")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
        return None
    if not mcsv.exists():
        print(f"  [FAIL] {tag}: no metrics CSV written")
        return None

    rows = list(csv.DictReader(mcsv.open("r", encoding="utf-8", newline="")))
    stats = curve_stats(rows, min_select_epoch)
    if stats is None:
        print(f"  [FAIL] {tag}: empty metrics CSV")
        return None

    last = rows[-1]
    for col in _RUN_CONSTANTS:
        try:
            stats[col] = float(last.get(col, ""))
        except (TypeError, ValueError):
            stats[col] = float("nan")
    ckpt.unlink(missing_ok=True)               # only the curve is needed downstream
    return stats


def summarize(results_csv: pathlib.Path) -> None:
    """Rank by margin against the PER-DRIVER floor; CORN vs CE paired on seed."""
    if not results_csv.exists():
        print("[summary] no results file yet")
        return
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("[summary] no completed runs yet")
        return

    by: Dict[Tuple[str, float, float], List[dict]] = {}
    for r in rows:
        by.setdefault((r["loss"], float(r["dropout"]), float(r["lr"])), []).append(r)

    print(f"\n{'loss':>5} {'drop':>5} {'lr':>7} {'n':>3} {'set-MAE':>8} {'se':>6} "
          f"{'init':>7} {'gain':>7} {'global':>7} {'perdrv':>7} {'vs pd':>7} "
          f"{'QWK':>6} {'E*':>4}  verdict")
    for (loss, dropout, lr), rs in sorted(by.items()):
        v = np.array([float(r["smoothed_best_set_mae"]) for r in rs])
        init = np.array([float(r["init_set_mae"]) for r in rs])
        qwk = np.array([float(r["qwk_at_best"]) for r in rs])
        gl = float(np.mean([float(r["const_set_mae"]) for r in rs]))
        pdc = float(np.mean([float(r["pdconst_set_mae"]) for r in rs]))
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        gain = float(init.mean() - v.mean())     # positive = training helped
        vs_pd = float(v.mean() - pdc)            # negative = beats the driver constant
        # Two independent questions, so two independent clauses. A model can
        # learn (gain > 0) and still lose to the per-driver constant, which is
        # exactly the "learned WHO, not WHEN" outcome this sweep is looking for.
        verdict = ("learns" if gain > 0.01 else "no learning")
        verdict += ", beats per-driver floor" if vs_pd < 0 else ", loses to per-driver floor"
        e_star = int(np.median([float(r["best_epoch_1se"]) for r in rs]))
        print(f"{loss:>5} {dropout:5.2f} {lr:7.0e} {len(v):3d} {v.mean():8.3f} {se:6.3f} "
              f"{init.mean():7.3f} {gain:+7.3f} {gl:7.3f} {pdc:7.3f} {vs_pd:+7.3f} "
              f"{qwk.mean():6.3f} {e_star:4d}  {verdict}")

    # CORN vs CE, PAIRED on (dropout, lr, seed). The split is identical for every
    # run -- no folds, no resampling -- so the pairing is exact and the only thing
    # left in the difference is the head.
    wide: Dict[Tuple[float, float, int], Dict[str, float]] = {}
    for r in rows:
        key = (float(r["dropout"]), float(r["lr"]), int(r["seed"]))
        wide.setdefault(key, {})[r["loss"]] = float(r["smoothed_best_set_mae"])
    d = np.array([c["corn"] - c["ce"] for c in wide.values() if "corn" in c and "ce" in c])
    if len(d) > 1:
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t = d.mean() / se if se else float("nan")
        print(f"\n[paired] CORN - CE on set-MAE over {len(d)} (dropout, lr, seed) cells: "
              f"{d.mean():+.3f} +/- {se:.3f} (t={t:+.2f}); negative favours CORN")

    print("\nRead the 'vs pd' column, not 'set-MAE': with every driver in both halves "
          "the global constant is not the binding floor.")
    print("NOT A SELECTION INSTRUMENT -- see the module docstring.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Within-driver diagnostic sweep (CORN vs CE).")
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--cache", default="data/cache/segments_w10_hz10.npz")
    ap.add_argument("--outdir", default="results/within_driver")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-select-epoch", dest="min_select_epoch", type=int, default=3)
    ap.add_argument("--window-seconds", dest="window_seconds", type=float, default=10.0)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.2,
                    help="PINNED at 0.2, deliberately, while the personalization path "
                         "moved to 0.3. This sweep is a finished diagnostic — the 20-run "
                         "CORN-vs-CE result (corn 0.956 vs a 1.201 per-driver floor) was "
                         "measured at an 80/20 split, and silently re-running it at 70/30 "
                         "would produce numbers that look comparable and are not. It is "
                         "also a different quantity from the other scripts' --val-frac: a "
                         "TRAIN/VAL split point, not an evaluation tail.")
    ap.add_argument("--losses", default=",".join(LOSSES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--jobs", type=int, default=1,
                    help="Concurrent training subprocesses. Pair with "
                         "--threads-per-job so jobs x threads does not exceed the "
                         "core count, or the runs contend and each gets slower.")
    ap.add_argument("--threads-per-job", dest="threads_per_job", type=int, default=0)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_csv = outdir / "sweep_results.csv"
    if args.summarize_only:
        summarize(results_csv)
        return

    losses = [s.strip() for s in args.losses.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    have = read_done(results_csv)
    todo = [(lo, d, lr, sd)
            for lo in losses for (d, lr) in CONFIGS for sd in seeds
            if (lo, d, lr, float(args.window_seconds), sd) not in have]
    print(f"[sweep] {len(todo)} run(s) to go ({len(have)} already done) -> {results_csv}")

    write_header = not results_csv.exists()
    fh = results_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(RESULTS_COLUMNS)
        fh.flush()

    def work(job: Tuple[str, float, float, int]):
        lo, d, lr, sd = job
        return job, run_one(args.in_jsonl, args.cache, lo, d, lr, sd, args.epochs,
                            args.patience, args.min_select_epoch, args.window_seconds,
                            args.val_frac, outdir, args.threads_per_job)

    def record(job: Tuple[str, float, float, int], stats: Optional[Dict[str, float]]) -> None:
        if stats is None:
            return
        lo, d, lr, sd = job
        head = [lo, d, lr, args.window_seconds, sd, "within-driver", args.val_frac]
        writer.writerow(head + [stats.get(c, "") for c in RESULTS_COLUMNS[len(head):]])
        fh.flush()          # every completed run is durable; a restart resumes

    try:
        if args.jobs > 1:
            with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
                for job, stats in ex.map(work, todo):
                    record(job, stats)
        else:
            for job in todo:
                record(*work(job))
    finally:
        fh.close()
    summarize(results_csv)


if __name__ == "__main__":
    main()
