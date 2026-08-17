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
import concurrent.futures as cf
import csv
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ProVoice.models.train_XLSTM import constant_baseline, iter_jsonl, set_mae
from ProVoice.training_scripts.folds import VALIDATION_FOLDS, train_pids_for_validation_fold

LEVELS = [f"Level_{i}" for i in range(1, 6)]
# FOLD-DERIVED columns: functions of the validation drivers alone, so they can be
# backfilled into an existing results file from its `val_pids` column without
# re-running anything. (pred_loa* cannot — those need the model at its selected
# epoch, and the sweep deletes checkpoints.)
BASELINE_COLUMNS = (["const_set_mae", "const_loa_mae", "const_set_acc", "const_loa_acc"]
                    + [f"lbl_loa{c}" for c in range(5)]     # VAL drivers' marks
                    + [f"trn_loa{c}" for c in range(5)])    # TRAIN drivers' marks

DROPOUTS = (0.10, 0.15, 0.20)
LRS = (1e-3, 2e-3)
SEEDS = (0, 1, 2, 3, 4)
SMOOTH_WINDOW = 5          # epochs, centred — see "HOW E* IS EXTRACTED"
# ONE definition of the adaptation support grid. The CLI default and summarize()'s
# fallback both read it: they disagreed once (CLI 5..60 vs a hard-coded 10/30/60
# fallback), which silently scored runs against a per-driver floor computed on a
# different grid than the runs themselves used.
DEFAULT_ADAPT_KS: Tuple[int, ...] = (5, 10, 20, 30, 40, 50, 60)

# In-flight runs, for the watchdog. run_one owns the entries; the watchdog only
# reads. A plain dict under a lock — there are at most --jobs entries.
_INFLIGHT: Dict[str, float] = {}
_INFLIGHT_LOCK = threading.Lock()

RESULTS_COLUMNS = [
    "dropout", "lr", "window_seconds", "loss", "fold", "val_pids", "seed",
    "best_set_mae",          # the run's raw minimum (what train_XLSTM selects on)
    "smoothed_best_set_mae", # minimum of the smoothed curve — the ranking quantity
    "best_epoch_smoothed",   # argmin of the smoothed curve — diagnostic only
    "best_epoch_1se",        # EARLIEST epoch within 1 SE of that minimum — E* comes from this
    # WHICH EPOCH every *_at_best column and the prediction histogram describe.
    # It equals best_epoch_smoothed, NOT best_epoch_1se: the reported metrics
    # characterize the config at its best, while E* (and therefore the model
    # run_lodo_population actually builds) comes from the earlier 1-SE epoch.
    # They coincide on a sharp optimum and diverge on a flat basin, so record it
    # rather than leaving a reader to assume.
    "metrics_epoch",
    "epochs_run",            # < --epochs when early stopping fired
    "set_acc_at_best", "qwk_at_best", "val_n",
    # Both decoders at the selected epoch, so a CE-vs-CORN comparison can hold
    # the decoder fixed post hoc instead of re-running the sweep. argmax is the
    # MODE (optimal for accuracy), median the rank rule (optimal for MAE).
    "mae_argmax", "acc_argmax", "mae_median", "acc_median",
    # PREDICTION HISTOGRAM at the selected epoch: how many of the val segments
    # were predicted at each LoA, next to how many marked each LoA. A run whose
    # predictions pile onto one level is reproducing the marginal, and its
    # set-MAE is the constant baseline in disguise — invisible in the aggregate
    # metrics, obvious here.
    "pred_loa0", "pred_loa1", "pred_loa2", "pred_loa3", "pred_loa4",
    # The constant floor and BOTH label marginals (lbl_* = val drivers,
    # trn_* = train drivers). Functions of the fold alone — not of dropout, lr,
    # seed or epoch — which is what makes them backfillable from `val_pids`.
    # Listed ONLY here: naming lbl_loa* separately above as well produced a CSV
    # with five duplicated headers.
    *BASELINE_COLUMNS,
    # POST-ADAPTATION columns, APPENDED so existing CSVs stay readable. All NaN
    # unless the trainer ran with --adapt-eval. `*_adapt` mirror the unadapted
    # trio exactly, so `--rank-on adapt_set_mae` swaps one column set for the
    # other and changes nothing else about how a run is summarized.
    "adapt_at_best",              # adapted score AT the unadapted-selected epoch
    "best_adapt_mae",             # raw minimum of the adapted curve
    "smoothed_best_adapt_mae",    # ranking quantity under --rank-on adapt_set_mae
    "best_epoch_smoothed_adapt",
    "best_epoch_1se_adapt",       # E* under --rank-on adapt_set_mae
    "adapt_acc_at_best_adapt",
    "adapt_n_drivers",
    # Untrained backbone + adapted head: the random-feature-reservoir baseline.
    # "Does training the backbone help ADAPTATION?" is a different question from
    # "does training help unadapted accuracy?", and only this answers it.
    "init_adapt_set_mae", "init_adapt_set_acc",
    # FIXED-TAIL query at the adapt-selected epoch. Reported only; the ranking
    # stays on the suffix metric. Levels are not comparable with the suffix
    # columns (different test set) but the shape across K is the interpretable one.
    "adapt_mae_tail", "adapt_acc_tail",
]


def results_columns(adapt_ks: Optional[List[int]] = None) -> List[str]:
    """``RESULTS_COLUMNS`` plus one pair of columns per K in the adapt grid.

    Derived rather than hard-coded so the header and the row are built from the
    same list: the grid is a CLI argument, and a fixed column list would either
    silently drop a K or write a column no run produced.
    """
    cols = list(RESULTS_COLUMNS)
    for k in (adapt_ks or []):
        cols += [f"adapt_mae_k{k}", f"adapt_acc_k{k}"]
    return cols
assert len(set(RESULTS_COLUMNS)) == len(RESULTS_COLUMNS), (
    "duplicate column(s) in RESULTS_COLUMNS: "
    f"{sorted({c for c in RESULTS_COLUMNS if RESULTS_COLUMNS.count(c) > 1})}")

# Identity prefix: written explicitly by the caller, everything after it comes
# from the run's own stats dict. Derived, not a literal, so adding an identity
# column cannot silently misalign the row (it did once, via a hard-coded [5:]).
ID_COLUMNS = ["dropout", "lr", "window_seconds", "loss", "fold", "val_pids", "seed"]
N_ID = len(ID_COLUMNS)
assert RESULTS_COLUMNS[:N_ID] == ID_COLUMNS, (
    f"RESULTS_COLUMNS must start with {ID_COLUMNS}, got {RESULTS_COLUMNS[:N_ID]}")


def load_segment_labels(path: pathlib.Path) -> Dict[str, List[List[int]]]:
    """``participantid -> [multi-hot marks per segment]``, streamed.

    One row per SEGMENT, not per frame: the file repeats a segment's label on
    every one of its ~190 frames, so de-duplicating on segment_id is what keeps
    this to ~1,446 tiny lists instead of half a million.
    """
    seen = set()
    out: Dict[str, List[List[int]]] = {}
    for r in iter_jsonl(path):
        sid = r.get("segment_id")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        marks = [int(float(r.get(k) or 0) > 0) for k in LEVELS]
        if not any(marks):
            continue
        out.setdefault(str(r.get("participantid", "")), []).append(marks)
    return out


def _marks(labels: Dict[str, List[List[int]]], pids: List[str]) -> np.ndarray:
    return np.asarray([m for p in pids for m in labels.get(str(p), [])], dtype=float)


def fold_baseline(labels: Dict[str, List[List[int]]], val_pids: List[str]) -> Dict[str, float]:
    """Constant floor plus BOTH label marginals for a fold.

    Three histograms are needed to read a run, not one:

      * ``trn_loa0..4`` — what the TRAINING drivers marked. This is the marginal
        the model is rewarded for reproducing.
      * ``lbl_loa0..4`` — what the VALIDATION drivers marked. This is what it is
        scored against.
      * ``pred_loa0..4`` — what it actually predicted (added per run, not here).

    The diagnosis follows from comparing them. Predictions tracking ``trn`` while
    ``lbl`` says something else is not a broken model — it is a model that
    learned its training drivers correctly and found they do not generalize,
    which on this cohort is the expected outcome (driver identity is worth ~4x
    more MAE than the task). Predictions tracking neither, piled on one level,
    is collapse. Aggregate set-MAE cannot tell those two apart.
    """
    val = _marks(labels, val_pids)
    held = {str(p) for p in val_pids}
    trn = _marks(labels, [p for p in labels if p not in held])
    out = dict(constant_baseline(val, 5))
    vc = val.sum(axis=0).astype(int) if val.size else np.zeros(5, dtype=int)
    tc = trn.sum(axis=0).astype(int) if trn.size else np.zeros(5, dtype=int)
    for c in range(5):
        out[f"lbl_loa{c}"] = int(vc[c])
        out[f"trn_loa{c}"] = int(tc[c])
    return out


def adapt_fold_baseline(labels: Dict[str, List[List[int]]], val_pids: List[str],
                        adapt_ks: List[int], min_query: int = 20) -> float:
    """The floor for the POST-ADAPTATION metric: a per-driver constant fitted on
    the same support the model adapts on.

    The global constant (``fold_baseline``) is the right reference only while the
    validation drivers are unseen AND unadapted. The moment the model is handed
    the driver's first K labels, "always predict this driver's favourite level"
    becomes available from those same K labels, needs no features, and is what an
    adapted model has to beat. Scoring an adapted number against the global
    constant compares two different games.

    Protocol is matched to ``train_XLSTM.evaluate_adaptation_val`` exactly: same K
    grid, same prefix support / suffix query, same ``min_query``, same aggregation
    (mean over K within a driver, then across drivers). Pure label arithmetic, so
    it costs nothing and needs no model.
    """
    per_driver = []
    for pid in val_pids:
        marks = labels.get(str(pid), [])
        if not marks:
            continue
        lv = np.asarray(marks, dtype=float)
        maes = []
        for K in adapt_ks:
            if len(lv) < K + min_query:
                continue
            sup, qry = lv[:K], lv[K:]
            c = int(np.argmin([set_mae(sup, np.full(len(sup), k, dtype=int))
                               for k in range(5)]))
            maes.append(set_mae(qry, np.full(len(qry), c, dtype=int)))
        if maes:
            per_driver.append(float(np.mean(maes)))
    return float(np.mean(per_driver)) if per_driver else float("nan")


def backfill_baselines(results_csv: pathlib.Path, labels: Dict[str, List[List[int]]]) -> bool:
    """Add the baseline columns to an existing results file, in place.

    The whole point of keying the baseline on the fold: a sweep that has already
    burned hours of GPU does not need re-running to gain this reference. Each row
    carries `val_pids`, which is all the baseline depends on.

    Returns True if the file was rewritten. Writes to a temp file and replaces,
    so an interrupted backfill cannot truncate the results.
    """
    if not results_csv.exists():
        return False
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames or [])
        rows = list(rd)
    if not rows or all(c in fieldnames for c in BASELINE_COLUMNS):
        return False

    cache: Dict[str, Dict[str, float]] = {}
    n_filled = 0
    for r in rows:
        key = r.get("val_pids", "")
        if key not in cache:
            cache[key] = fold_baseline(labels, [p for p in key.split("|") if p])
        r.update({k: cache[key][k] for k in BASELINE_COLUMNS})
        n_filled += 1

    out_fields = fieldnames + [c for c in BASELINE_COLUMNS if c not in fieldnames]
    tmp = results_csv.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(results_csv)
    print(f"[backfill] added the fold-derived columns to {n_filled} existing row(s) in "
          f"{results_csv.name} — no re-running needed")
    for key, b in sorted(cache.items()):
        marg = [b[f"lbl_loa{c}"] for c in range(5)]
        print(f"    val={key or '(none)'}: best constant set-MAE={b['const_set_mae']:.3f} "
              f"@LoA{b['const_loa_mae']} | set-acc={b['const_set_acc']:.3f} "
              f"@LoA{b['const_loa_acc']} | label marks[LoA0..4]={marg}")
    print("    NOTE pred_loa0..4 and the per-decoder metrics stay blank on these rows: "
          "they need the model at its selected epoch, and completed runs' checkpoints "
          "were deleted. New runs fill them in.")
    return True


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


def report_device() -> None:
    """Print the device the CHILD runs will use, once, before any of them start.

    run_one uses capture_output=True, so the trainer's own ``[device]`` line is
    swallowed unless that run fails. Without this the sweep gives no indication
    of CPU vs GPU at all -- and a silent CPU fallback looks exactly like a slow
    GPU. Imported lazily so --resummarize does not pay torch's ~2 s import.
    """
    try:
        import torch
    except Exception as e:                      # pragma: no cover
        print(f"[device] could not import torch: {e}")
        return
    if torch.cuda.is_available():
        print(f"[device] cuda — {torch.cuda.get_device_name(0)} (torch {torch.__version__})\n"
              f"[device] NOTE Windows Task Manager shows the '3D' engine by default and "
              f"CUDA work will NOT appear there. Switch a graph to 'Compute_0'/'Cuda', or "
              f"use `nvidia-smi`, before concluding the GPU is idle.")
    elif torch.version.cuda is None:
        print(f"[device] cpu — torch {torch.__version__} is a CPU-ONLY build. Every run in "
              f"this sweep will use the CPU.\n"
              f"[device] Fix: uv run --no-sync python scripts/setup_cuda_torch.py, then "
              f"launch with `uv run --no-sync` (a plain `uv run` re-syncs and reinstates "
              f"the CPU wheel).")
    else:
        print(f"[device] cpu — torch {torch.__version__} has CUDA {torch.version.cuda} "
              f"compiled in but sees no device; check the driver / nvidia-smi.")


def read_done(path: pathlib.Path) -> set:
    """Already-completed run keys, for resuming.

    The key is ``(dropout, lr, window_seconds, loss, fold, seed)`` — all six.
    window and loss are in it because a 5 s and a 10 s run, or a corn and a ce
    run, at otherwise identical settings are DIFFERENT experiments; without them
    the second would be skipped as already done.
    """
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                # window_seconds is PART of the key: a 5 s and a 10 s run at the
                # same (dropout, lr, fold, seed) are different experiments, and
                # without it the second would be skipped as "already done".
                # Rows predating the column resolve to None and so never collide
                # with a windowed run.
                w = row.get("window_seconds", "")
                w = float(w) if w not in ("", None) else None
                done.add((float(row["dropout"]), float(row["lr"]), w,
                          row.get("loss", "") or None,
                          int(row["fold"]), int(row["seed"])))
            except (KeyError, ValueError):
                continue        # a torn final line from an interrupted run
    return done


def run_tag(dropout: float, lr: float, val_pids: List[str], seed: int,
            window_seconds: Optional[float] = None, loss: Optional[str] = None) -> str:
    """Filename tag for a run's artifacts.

    ``window_seconds`` is part of it because a 5 s and a 10 s run at the same
    (dropout, lr, fold, seed) are DIFFERENT experiments — same tag would have
    them overwrite each other's metric curves. Pass None to reproduce the
    pre-window tag, which is what ``--resummarize`` needs to find older runs.
    """
    t = f"d{dropout}_lr{lr}_f{'-'.join(val_pids)}_s{seed}"
    if window_seconds is not None:
        t += f"_w{window_seconds:g}"
    if loss:
        t += f"_{loss}"
    return t


def _f(v) -> float:
    """float(v) with '' / None / 'nan' all collapsing to NaN, never an exception."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def reduce_curve(y: np.ndarray, min_select_epoch: int) -> Tuple[int, int, np.ndarray]:
    """Smooth a curve, then return (argmin epoch index, 1-SE epoch index, smoothed).

    Extracted so the UNADAPTED and the POST-ADAPTATION curves are reduced by
    exactly the same rule. Selecting the two on different rules would make the
    ``--rank-on`` switch change more than the quantity being ranked.

    See ``curve_stats`` for why the epoch floor and the 1-SE rule exist.
    """
    sm = smooth(y, SMOOTH_WINDOW)
    lo = min(int(min_select_epoch), len(sm) - 1) if len(sm) else 0
    j = lo + int(np.argmin(sm[lo:]))
    resid = y - sm
    sigma = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0
    se_sm = sigma / np.sqrt(min(SMOOTH_WINDOW, len(y)))
    within = np.flatnonzero(sm[lo:] <= sm[j] + se_sm)
    j1 = lo + int(within[0]) if within.size else j
    return j, j1, sm


def curve_stats(rows: List[dict], min_select_epoch: int) -> Optional[Dict[str, float]]:
    """Derive every per-run number from one run's per-epoch metric curve.

    The ONE place the curve is turned into E* candidates and reported metrics, so
    a live sweep and a ``--resummarize`` pass over old curves cannot disagree
    about what the numbers mean. Everything here is a function of the curve
    alone — which is exactly why the overnight compute is salvageable: today's
    changes altered how E* is EXTRACTED, not what was trained.

    Returns None if the curve is unusable.
    """
    if not rows:
        return None
    mae = np.array([float(r["set_mae"]) for r in rows])
    sm = smooth(mae, SMOOTH_WINDOW)
    # Honour the same epoch floor the trainer applies to its own checkpointing.
    # Without this the sweep would re-derive E* from the full curve and hand back
    # the epoch-0 the trainer just refused to select — and E*=0 makes the LODO
    # runner train for zero epochs.
    lo = min(int(min_select_epoch), len(sm) - 1) if len(sm) else 0
    j = lo + int(np.argmin(sm[lo:]))
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
    within = np.flatnonzero(sm[lo:] <= sm[j] + se_sm)
    j1 = lo + int(within[0]) if within.size else j

    def at(col: str, default=float("nan")):
        """Value at the SELECTED epoch. Tolerates metrics CSVs from older runs
        that predate a column rather than failing the whole sweep."""
        v = rows[j].get(col, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    out = {
        "best_set_mae": float(mae.min()),
        "smoothed_best_set_mae": float(sm[j]),
        "best_epoch_smoothed": int(rows[j]["epoch"]),
        "best_epoch_1se": int(rows[j1]["epoch"]),
        # Everything below is read from rows[j] — the smoothed-argmin epoch.
        # NOT the final epoch, and NOT the 1-SE epoch that E* uses.
        "metrics_epoch": int(rows[j]["epoch"]),
        "epochs_run": len(rows),
        "set_acc_at_best": float(rows[j]["set_acc"]),
        "qwk_at_best": float(rows[j]["qwk"]),
        "val_n": int(rows[j]["val_n"]),
        "mae_argmax": at("mae_argmax"), "acc_argmax": at("acc_argmax"),
        "mae_median": at("mae_median"), "acc_median": at("acc_median"),
    }
    for c in range(5):
        out[f"pred_loa{c}"] = at(f"pred_loa{c}", 0.0)

    # POST-ADAPTATION curve, when the trainer was run with --adapt-eval. Reduced
    # by the SAME rule as the unadapted one (reduce_curve), so `--rank-on` picks
    # between two comparable quantities rather than two different procedures.
    #
    # `adapt_at_best` is the adapted score at the epoch the UNADAPTED curve
    # selects; `smoothed_best_adapt_mae` is the adapted score at the epoch the
    # ADAPTED curve selects. The gap between them is the whole argument for
    # switching criteria - if it is ~0 the change is cosmetic, and if it is large
    # then E* chosen on the unadapted curve is landing in the wrong place.
    out["init_adapt_set_mae"] = _f(rows[-1].get("init_adapt_set_mae"))
    out["init_adapt_set_acc"] = _f(rows[-1].get("init_adapt_set_acc"))
    adapt = np.array([float(r.get("adapt_set_mae", "nan") or "nan") for r in rows])
    out["adapt_at_best"] = float(adapt[j]) if np.isfinite(adapt[j]) else float("nan")
    if np.isfinite(adapt).sum() >= 2:
        finite = np.isfinite(adapt)
        # Non-finite epochs (a driver too short for any K) would poison the
        # smoother, so reduce over the finite prefix and map the index back.
        idx = np.flatnonzero(finite)
        ja, j1a, sma = reduce_curve(adapt[idx], min_select_epoch)
        out["smoothed_best_adapt_mae"] = float(sma[ja])
        out["best_adapt_mae"] = float(np.nanmin(adapt))
        out["best_epoch_smoothed_adapt"] = int(rows[idx[ja]]["epoch"])
        out["best_epoch_1se_adapt"] = int(rows[idx[j1a]]["epoch"])
        out["adapt_acc_at_best_adapt"] = float(
            rows[idx[ja]].get("adapt_set_acc", "nan") or "nan")
        out["adapt_n_drivers"] = float(rows[idx[ja]].get("adapt_n_drivers", 0) or 0)
        out["adapt_mae_tail"] = _f(rows[idx[ja]].get("adapt_set_mae_tail"))
        out["adapt_acc_tail"] = _f(rows[idx[ja]].get("adapt_set_acc_tail"))
        # Per-K values are read AT THE ADAPT-SELECTED EPOCH, not at their own
        # per-K minima: the run ships one epoch, so the K curve has to be the
        # cross-section of that epoch. Taking each K's own best would describe a
        # model that never existed.
        best_row = rows[idx[ja]]
        for col, val in best_row.items():
            if col.startswith("adapt_set_mae_k"):
                out[f"adapt_mae_k{col.rsplit('_k', 1)[1]}"] = _f(val)
            elif col.startswith("adapt_set_acc_k"):
                out[f"adapt_acc_k{col.rsplit('_k', 1)[1]}"] = _f(val)
    else:
        for k in ("smoothed_best_adapt_mae", "best_adapt_mae",
                  "adapt_acc_at_best_adapt", "adapt_mae_tail", "adapt_acc_tail"):
            out[k] = float("nan")
        out["best_epoch_smoothed_adapt"] = -1
        out["best_epoch_1se_adapt"] = -1
        out["adapt_n_drivers"] = 0.0
    return out


def watchdog(stop: threading.Event, interval: float, stall_after: float) -> None:
    """Print what is still running, periodically, until ``stop`` is set.

    Without this a sweep is silent between completions. ``run_one`` captures the
    child's output, so a run that hangs produces NO output at all -- and at
    --jobs 4 a single stuck worker quietly removes a quarter of the throughput
    with nothing in the log to show for it. That is not hypothetical now that
    --adapt-eval makes a legitimate run take ~20 minutes: "slow" and "wedged"
    look identical from outside.

    Reports elapsed time per in-flight run and marks anything past
    ``stall_after`` as STALLED. Daemon thread, so it can never hold up exit.
    """
    while not stop.wait(interval):
        with _INFLIGHT_LOCK:
            snapshot = sorted(_INFLIGHT.items(), key=lambda kv: kv[1])
        if not snapshot:
            continue
        now = time.time()
        parts = []
        for tag, t0 in snapshot:
            el = now - t0
            parts.append(f"{tag} {el / 60:.1f}m" + (" STALLED?" if el > stall_after else ""))
        print(f"[heartbeat] {len(snapshot)} run(s) in flight: " + " | ".join(parts),
              flush=True)


def run_one(in_jsonl: str, dropout: float, lr: float, val_pids: List[str], seed: int,
            epochs: int, patience: int, min_delta: float, min_select_epoch: int,
            window_seconds: float, loss: str, workdir: pathlib.Path,
            extra: List[str], cache: str = "",
            threads: int = 0, timeout: float = 0.0) -> Optional[Dict[str, float]]:
    """One training run. Returns its curve summary, or None if it failed.

    Runs train_XLSTM as a SUBPROCESS rather than importing it: each run needs a
    fresh process-global torch RNG state and a clean CUDA allocator, and a crash
    in run 97 of 180 should cost that run, not the sweep.

    ``threads`` caps the child's intra-op thread pool, which is what lets ``--jobs``
    trade width for count. This model does not scale inside a process: at 12 torch
    threads it is only **1.82x** faster than at 1, and 16 threads is slower than 4.
    The tensors are too small to amortize the synchronization and the rest goes
    into barrier spin -- which Windows reports as >90% CPU while throughput stays
    flat.

    MEASURED aggregate step throughput (12-physical/16-logical box, real concurrent
    processes -- NOT extrapolated from the single-process thread curve, which
    overstates this badly by assuming idle cores deliver full speed):

        1 proc x 16 thr   15.8 steps/s   (baseline)
        4 proc x  4 thr   31.8           2.01x
        8 proc x  1 thr   32.1           2.04x
       12 proc x  1 thr   34.7           2.20x

    Per-process step time degrades from 94 ms alone to 346 ms at 12-way, so the
    spare cores are contended rather than idle -- the ceiling is ~2x, not the ~6x
    the thread curve alone suggests. END-TO-END on a real 4-run sweep, which also
    pays the ~4.7 s per-process fixed cost: 259 s sequential -> 112 s at 4x4,
    **2.31x**. 4x4 beat 8x1 in every measurement.

    DETERMINISM: torch CPU reductions depend on the thread count, so the SAME seed
    at a different --threads-per-job produces a slightly different trajectory
    (verified: 25 identical steps diverge at the 8th significant digit; the same
    thread count reproduces bit-exactly). Each run remains a valid sample, but a
    sweep half-run at one setting and resumed at another mixes two numeric
    regimes. Keep --jobs/--threads-per-job FIXED for a whole stage, the same way
    --decision-hz is held fixed across participants.
    """
    tag = run_tag(dropout, lr, val_pids, seed, window_seconds, loss)
    ckpt = workdir / f"ckpt_{tag}.pt"          # written, then discarded — stage 2 retrains
    mcsv = workdir / f"metrics_{tag}.csv"      # KEPT: --resummarize re-reads these
    cmd = [sys.executable, "-m", "ProVoice.models.train_XLSTM",
           "--in", in_jsonl, "--out", str(ckpt), "--loss", loss,
           "--val-pids", ",".join(val_pids), "--metrics-csv", str(mcsv),
           "--dropout", str(dropout), "--lr", str(lr), "--seed", str(seed),
           "--epochs", str(epochs), "--patience", str(patience),
           "--min-delta", str(min_delta),
           "--min-select-epoch", str(min_select_epoch),
           "--window-seconds", str(window_seconds)] + extra
    if cache:
        cmd += ["--cache", cache]
    env = dict(os.environ)
    if threads > 0:
        # Set in the ENVIRONMENT, not via a trainer flag: torch reads these when it
        # is imported, so a flag parsed in main() would be too late to shrink the
        # pool that has already been built.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            env[var] = str(threads)
    # Registered for the watchdog BEFORE the child starts and removed in `finally`,
    # so a crash cannot leave a phantom entry reported as stalled forever.
    with _INFLIGHT_LOCK:
        _INFLIGHT[tag] = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=(timeout if timeout and timeout > 0 else None))
    except subprocess.TimeoutExpired:
        # subprocess.run has already killed the child by the time this is raised.
        print(f"  [TIMEOUT] {tag} exceeded --run-timeout ({timeout:.0f}s) and was killed. "
              f"Its row is NOT recorded, so re-running the same command retries it.",
              flush=True)
        return None
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT.pop(tag, None)
    if proc.returncode != 0:
        print(f"  [FAIL] {tag} (exit {proc.returncode})")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
        return None
    if not mcsv.exists():
        print(f"  [FAIL] {tag}: no metrics CSV written")
        return None
    stats = curve_stats(list(csv.DictReader(mcsv.open("r", encoding="utf-8", newline=""))),
                        min_select_epoch)
    if stats is None:
        print(f"  [FAIL] {tag}: empty metrics CSV")
        return None
    ckpt.unlink(missing_ok=True)               # only the curve is needed downstream
    return stats


def resummarize(results_csv: pathlib.Path, workdir: pathlib.Path,  # noqa: D401
                labels: Dict[str, List[List[int]]], min_select_epoch: int,
                window_seconds: float, loss: str,
                adapt_ks: Optional[List[int]] = None) -> None:
    """Recompute every derived column from the RETAINED per-epoch curves.

    Why this exists: everything that changed about E* — the --min-select-epoch
    floor, the 1-SE rule, the fold-derived baselines — changes how E* is read
    OFF a curve, not what was trained. The curves are still in ``runs/``, so a
    sweep that already burned hours does not have to be repeated to gain them.

    What it CANNOT recover: ``pred_loa*`` and the per-decoder metrics for runs
    whose trainer predated those columns. Those need the model at its selected
    epoch, and the checkpoints are deleted. They stay blank rather than being
    filled with plausible-looking zeros.

    Rows whose curve file is missing are left exactly as they were.

    Rows predating the ``window_seconds`` / ``loss`` columns are stamped with the
    CLI values, which is an ASSUMPTION about what those runs used — the same one
    already applied to the window. It has to be made for both: those two fields
    are part of the resume key, so a row left blank in either would never match a
    subsequent run and the whole sweep would retrain despite having just been
    salvaged. If the assumption is wrong for a file, re-run it rather than
    resummarizing it.
    """
    if not results_csv.exists():
        raise SystemExit(f"--resummarize needs an existing {results_csv}")
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{results_csv} has no rows")

    n_ok = n_missing = 0
    for r in rows:
        dropout, lr = float(r["dropout"]), float(r["lr"])
        seed = int(r["seed"])
        vp = [p for p in r.get("val_pids", "").split("|") if p]
        w = r.get("window_seconds", "")
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = window_seconds          # pre-window rows: assume the CLI value
        # Try the current tag, then the pre-window one, so curves written before
        # window tracking are still found.
        mcsv = None
        # `row_loss`, NOT `loss`: `loss` is the CLI fallback parameter, and
        # rebinding it here shadowed it for the rest of the function — which
        # silently defeated the stamping below and left every resummarized row
        # unresumable.
        row_loss = r.get("loss", "") or None
        for cand in (run_tag(dropout, lr, vp, seed, w, row_loss),
                     run_tag(dropout, lr, vp, seed, w, loss),
                     run_tag(dropout, lr, vp, seed, w),
                     run_tag(dropout, lr, vp, seed)):
            p = workdir / f"metrics_{cand}.csv"
            if p.exists():
                mcsv = p
                break
        if mcsv is None:
            n_missing += 1
            continue
        stats = curve_stats(list(csv.DictReader(mcsv.open("r", encoding="utf-8", newline=""))),
                            min_select_epoch)
        if stats is None:
            n_missing += 1
            continue
        r.update({k: v for k, v in stats.items()})
        # Both key fields, not just the window: see the docstring. Existing
        # values win, so a row that already knows its provenance keeps it.
        r["window_seconds"] = w
        r["loss"] = row_loss or loss
        r.update(fold_baseline(labels, vp))
        n_ok += 1

    fields = list(results_columns(adapt_ks))
    for extra in (c for c in rows[0] if c not in fields):
        fields.append(extra)
    tmp = results_csv.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w_.writeheader()
        for r in rows:
            w_.writerow({k: r.get(k, "") for k in fields})
    tmp.replace(results_csv)
    print(f"[resummarize] recomputed {n_ok} row(s) from retained curves in {workdir}"
          + (f"; {n_missing} row(s) had no curve file and were left unchanged"
             if n_missing else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--outdir", default="results/pop_sweep",
                    help="Results CSV, per-run metric curves and selected_population.json.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", dest="min_delta", type=float, default=0.0)
    ap.add_argument("--loss", choices=["corn", "ce"], default="corn",
                    help="Head/loss for every run in this sweep. Recorded as a column and "
                         "part of both the resume key and the artifact filenames, so a "
                         "corn and a ce sweep can share an --outdir without colliding. "
                         "'ce' is the ablation: only 'corn' supports the Laplace UQ layer, "
                         "so it cannot be the deployed arm.")
    ap.add_argument("--window-seconds", dest="window_seconds", type=float, default=10.0,
                    help="Model input window, passed to the trainer AND recorded as a "
                         "column and as part of the resume key. Sweeping 5/10/20 into the "
                         "SAME --outdir is therefore safe: runs cannot overwrite each "
                         "other's curves and cannot be mistaken for one another on resume. "
                         "Changes context_length (50/100/200), so checkpoints from "
                         "different windows are not interchangeable downstream.")
    ap.add_argument("--resummarize", action="store_true",
                    help="Do not train. Re-read the retained per-epoch curves in "
                         "<outdir>/runs/ and recompute every derived column with the "
                         "CURRENT rules (--min-select-epoch floor, 1-SE E*, fold "
                         "baselines), then re-write the summary. Salvages a sweep that ran "
                         "under older selection logic: what changed is how E* is read off "
                         "a curve, not what was trained. pred_loa* and the per-decoder "
                         "columns stay blank for runs whose trainer predated them.")
    ap.add_argument("--min-select-epoch", dest="min_select_epoch", type=int, default=3,
                    help="Earliest epoch eligible to be selected, passed to the trainer "
                         "AND applied when this script re-derives E* from the curve. "
                         "Keep >= --warmup-epochs; 0 disables. See the trainer flag.")
    ap.add_argument("--dropouts", default=",".join(str(d) for d in DROPOUTS))
    ap.add_argument("--lrs", default=",".join(str(l) for l in LRS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--folds", default="",
                    help="Comma-separated fold indices to run (default: all 6). For "
                         "splitting the sweep across machines.")
    ap.add_argument("--adapt-eval", dest="adapt_eval", action="store_true",
                    help="Score every epoch AFTER per-driver adaptation as well, and make "
                         "that the ranking quantity (implies --rank-on adapt_set_mae unless "
                         "overridden). WHY: the population model is never served unadapted, "
                         "so the default criterion optimizes a proxy the design has already "
                         "declared is not the objective - and on this cohort it is "
                         "NOT RESOLVABLE anyway (config spread 0.023 <= 2 SE 0.026 for "
                         "corn_w10), so the tie-break is doing the selecting. "
                         "xlstm_maml.evaluate_adaptation and sweep_l2sp_tau already select "
                         "post-adaptation; this makes stage 1 agree with them. Both numbers "
                         "are always written, so a run can be re-ranked either way with "
                         "--resummarize --rank-on <column>.")
    ap.add_argument("--adapt-k", dest="adapt_k",
                    default=",".join(str(k) for k in DEFAULT_ADAPT_KS),
                    help="Support sizes for --adapt-eval, averaged WITHIN driver then across "
                         "drivers to give the ranking quantity. Pre-commit this grid: the "
                         "metric is a curve and the aggregation must not be chosen after "
                         "seeing the result. Each K is ALSO carried into sweep_results.csv "
                         "individually (adapt_mae_k<K>/adapt_acc_k<K>, taken at the "
                         "adapt-selected epoch), so the quality-vs-K SHAPE survives and a "
                         "config that wins on the mean while losing at small K is visible. "
                         "Cost is linear in the grid size: ~3.1 s per (driver, K) cell.")
    ap.add_argument("--adapt-tau", dest="adapt_tau", type=float, default=2.0,
                    help="PROVISIONAL L2-SP prior precision for --adapt-eval. The committed "
                         "tau comes from stage 3, which depends on this stage's output - so "
                         "fix it here, disclose it, and verify at stage 3 that the winning "
                         "config is not tau-sensitive.")
    ap.add_argument("--rank-on", dest="rank_on",
                    choices=["set_mae", "adapt_set_mae"], default=None,
                    help="Which column summarize() ranks configurations and derives E* from. "
                         "Defaults to adapt_set_mae when --adapt-eval is set, else set_mae.")
    ap.add_argument("--trainer-arg", dest="trainer_args", action="append", default=[],
                    help="Extra flag passed through to train_XLSTM, repeatable, e.g. "
                         "--trainer-arg=--grad-clip --trainer-arg=0.5")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Concurrent training runs. The runs are independent, and this model "
                         "does not scale INSIDE a process (12 torch threads is only 1.82x "
                         "faster than 1), so throughput comes from count rather than width. "
                         "MEASURED ceiling is ~2x, not the ~6x the thread curve implies, "
                         "because the spare cores are contended rather than free. "
                         "'--jobs 4 --threads-per-job 4' (the default pairing on a 16-thread "
                         "box) measured 2.31x end-to-end and beat 8x1 in every test. "
                         "KEEP THIS FIXED for a whole stage: thread count changes float "
                         "reduction order, so resuming at a different setting mixes two "
                         "numeric regimes into one results table. "
                         "STRONGLY prefer --cache with this: without it every concurrent "
                         "job re-parses the 971 MB JSONL and peaks ~0.6 GB, so 12 jobs is "
                         "~7 GB of RAM and 12-way disk contention for no reason.")
    ap.add_argument("--threads-per-job", dest="threads_per_job", type=int, default=0,
                    help="Torch intra-op threads per run (0 = auto: cpu_count // jobs, "
                         "floored at 1). Set as an environment variable on the child, since "
                         "torch sizes its pool at import.")
    ap.add_argument("--heartbeat-seconds", dest="heartbeat_seconds", type=float, default=300.0,
                    help="How often to print the in-flight runs and their elapsed time "
                         "(0 = never). run_one captures the child's output, so without this "
                         "a sweep is silent between completions and a hung run looks exactly "
                         "like a slow one.")
    ap.add_argument("--stall-warn-seconds", dest="stall_warn_seconds", type=float, default=2400.0,
                    help="Elapsed time after which the heartbeat marks a run STALLED? "
                         "Advisory only - nothing is killed. Default 40 min, comfortably "
                         "above a legitimate --adapt-eval run (~20 min at 7 K values).")
    ap.add_argument("--run-timeout", dest="run_timeout", type=float, default=0.0,
                    help="Kill a training subprocess after this many seconds (0 = never). "
                         "The killed run is not recorded, so re-running the sweep retries "
                         "just that cell. Off by default: a timeout tuned for one machine "
                         "silently truncates a slower one.")
    ap.add_argument("--cache", default="",
                    help="Pre-encoded segment cache (.npz), passed through to the trainer. "
                         "Build with scripts/build_segment_cache.py. Saves ~24 s per run "
                         "(~2.8 h over the pipeline's 420 runs) and, more importantly, makes "
                         "--jobs practical by cutting per-process peak RSS from ~0.6 GB to "
                         "tens of MB. Must match --window-seconds; the trainer errors on a "
                         "stale or mismatched cache rather than silently re-parsing.")
    args = ap.parse_args()

    # --adapt-eval is surfaced as a first-class flag rather than left to
    # --trainer-arg, because the ranking column has to move with it. Composed
    # here so the two cannot drift apart.
    trainer_args = list(args.trainer_args)
    if args.adapt_eval:
        trainer_args += ["--adapt-eval", "--adapt-k", str(args.adapt_k),
                         "--adapt-tau", str(args.adapt_tau)]
        # --adapt-eval IMPLIES --select-on adapt_set_mae. Without it the trainer's
        # --patience and checkpointing key off the UNADAPTED curve while this
        # script ranks and derives E* from the ADAPTED one: a run whose unadapted
        # curve plateaus while its adapted curve is still improving gets stopped
        # early, truncating the very curve the sweep reads. At ~43 s/epoch of
        # adaptation that also throws away the expensive part of the run.
        # Escape hatch: pass --trainer-arg=--select-on --trainer-arg=<col>
        # explicitly and that wins, since argparse takes the LAST occurrence.
        if not any(a.startswith("--select-on") for a in args.trainer_args):
            trainer_args += ["--select-on", "adapt_set_mae"]
    rank_on = args.rank_on or ("adapt_set_mae" if args.adapt_eval else "set_mae")
    adapt_ks = sorted({int(k) for k in str(args.adapt_k).split(",") if k.strip()})
    if rank_on == "adapt_set_mae" and not args.adapt_eval:
        print("[warn] --rank-on adapt_set_mae without --adapt-eval: this only works on a "
              "results CSV whose runs were trained with --adapt-eval; use --resummarize "
              "to re-rank an existing one.")
    args.jobs = max(1, args.jobs)
    # At --jobs 1 leave torch's own default ALONE. os.cpu_count() reports LOGICAL
    # cores (16 here) while torch defaults to PHYSICAL (12), and 16 threads measured
    # SLOWER than 12 (49.0 vs 41.0 ms/step) -- so deriving 16//1 here would have
    # silently made the default single-job path ~20% slower than before this flag
    # existed. Auto only ever divides the machine up; it never widens a run.
    threads = (args.threads_per_job if args.threads_per_job > 0
               else (0 if args.jobs == 1 else max(1, (os.cpu_count() or 4) // args.jobs)))
    if args.jobs > 1 and not args.cache:
        print(f"[warn] --jobs {args.jobs} without --cache: each run will re-parse the source "
              f"JSONL (~24 s, ~0.6 GB peak), so expect ~{0.6 * args.jobs:.1f} GB of RAM and "
              f"heavy disk contention at startup. Build a cache first:\n"
              f"    python -m scripts.build_segment_cache --in {args.in_jsonl} "
              f"--windows {args.window_seconds:g}")

    dropouts = [float(x) for x in args.dropouts.split(",") if x.strip()]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    fold_idx = ([int(x) for x in args.folds.split(",") if x.strip()]
                if args.folds else list(range(len(VALIDATION_FOLDS))))

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "runs"; workdir.mkdir(exist_ok=True)
    results_csv = outdir / "sweep_results.csv"

    # Per-fold constant-prediction floors. Cheap (one pass, de-duplicated to
    # ~1,446 label vectors) and needed both to annotate new rows and to backfill
    # old ones, so it is loaded before anything else touches the CSV.
    labels = load_segment_labels(pathlib.Path(args.in_jsonl))
    if args.resummarize:
        resummarize(results_csv, workdir, labels, args.min_select_epoch,
                    args.window_seconds, args.loss,
                    adapt_ks if args.adapt_eval else None)
        summarize(results_csv, outdir, rank_on, args.in_jsonl, adapt_ks,
                  select_window=args.window_seconds, select_loss=args.loss,
                  adapt_tau=args.adapt_tau if args.adapt_eval else None)
        return
    backfill_baselines(results_csv, labels)

    done = read_done(results_csv)
    if done:
        print(f"[resume] {len(done)} run(s) already in {results_csv}; they will be skipped")

    report_device()
    total = len(dropouts) * len(lrs) * len(fold_idx) * len(seeds)
    print(f"[plan] {len(dropouts)} dropout x {len(lrs)} lr x {len(fold_idx)} fold x "
          f"{len(seeds)} seed = {total} runs (<= {args.epochs} epochs each)")
    for i in fold_idx:
        print(f"  fold {i}: val={list(VALIDATION_FOLDS[i])} "
              f"train={train_pids_for_validation_fold(VALIDATION_FOLDS[i])}")

    # Header and every row are built from ONE list, derived from --adapt-k, so a
    # change to the grid cannot leave them describing different things.
    out_columns = results_columns(adapt_ks if args.adapt_eval else None)

    new = not results_csv.exists()
    fh = results_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    if new:
        writer.writerow(out_columns)
        fh.flush()

    # Flatten to a work list first, so --jobs can dispatch across configs rather
    # than only within the innermost seed loop.
    tasks = []
    for dropout in dropouts:
        for lr in lrs:
            for i in fold_idx:
                val_pids = list(VALIDATION_FOLDS[i])
                base = fold_baseline(labels, val_pids)
                for seed in seeds:
                    if (dropout, lr, args.window_seconds, args.loss, i, seed) in done:
                        continue
                    tasks.append((dropout, lr, i, val_pids, seed, base))

    print(f"[run] {len(tasks)} run(s) to do of {total}; {args.jobs} concurrent x "
          + (f"{threads} torch thread(s) each" if threads else "torch default threads"),
          flush=True)

    def report(t, r) -> str:
        """The per-run block, built as ONE string. With --jobs > 1 several runs
        finish at once, and separate print() calls would interleave line by line."""
        dropout, lr, i, val_pids, seed, base = t
        d = r["smoothed_best_set_mae"] - base["const_set_mae"]
        def pct(v):
            tot = sum(v) or 1
            return "[" + " ".join(f"{100*x/tot:4.1f}" for x in v) + "]"
        pv = [int(r[f"pred_loa{c}"]) for c in range(5)]
        lv = [base[f"lbl_loa{c}"] for c in range(5)]
        tv = [base[f"trn_loa{c}"] for c in range(5)]
        return (f"      set-MAE raw={r['best_set_mae']:.3f} "
                f"smoothed={r['smoothed_best_set_mae']:.3f} "
                f"@epoch {r['best_epoch_smoothed']} "
                f"(1se {r['best_epoch_1se']}, {r['epochs_run']} epochs run) "
                f"| const {base['const_set_mae']:.3f} -> {d:+.3f}"
                f"{'' if d < 0 else '  WORSE THAN CONSTANT'}\n"
                # Three histograms, normalized to shares so unequal totals
                # (train has ~5x the segments of val) do not hide the shape.
                f"      @epoch {r['metrics_epoch']}  %LoA0..4  "
                f"pred {pct(pv)}  val {pct(lv)}  train {pct(tv)}\n"
                f"      argmax MAE={r['mae_argmax']:.3f} "
                f"median MAE={r['mae_median']:.3f}")

    def work(t):
        dropout, lr, i, val_pids, seed, _base = t
        return t, run_one(args.in_jsonl, dropout, lr, val_pids, seed,
                          args.epochs, args.patience, args.min_delta,
                          args.min_select_epoch, args.window_seconds,
                          args.loss, workdir, trainer_args,
                          cache=args.cache, threads=threads,
                          timeout=args.run_timeout)

    # Only THIS thread touches the writer, so the CSV needs no lock: the pool
    # workers just block on their subprocess and hand back the parsed curve.
    n_done = n_failed = 0
    stop_hb = threading.Event()
    hb = None
    if args.heartbeat_seconds and args.heartbeat_seconds > 0:
        hb = threading.Thread(target=watchdog, daemon=True,
                              args=(stop_hb, args.heartbeat_seconds, args.stall_warn_seconds))
        hb.start()
    ex = cf.ThreadPoolExecutor(max_workers=args.jobs)
    futs = {ex.submit(work, t): t for t in tasks}
    try:
        for fut in cf.as_completed(futs):
            n_done += 1
            try:
                t, r = fut.result()
            except Exception as e:
                # ONE run must not take down the batch. curve_stats parses a CSV a
                # crashed trainer may have left torn mid-line (read_done already
                # anticipates exactly that), so this is a realistic failure, not a
                # theoretical one — and at run 97 of 180 re-raising would discard
                # the summary for the 96 that succeeded.
                n_failed += 1
                d, l, i, vp, s, _ = futs[fut]
                print(f"[ERROR] dropout={d} lr={l:g} fold={i}{vp} seed={s}: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            dropout, lr, i, val_pids, seed, base = t
            if r is None:
                n_failed += 1
                continue
            r.update(base)
            # The identity prefix and RESULTS_COLUMNS must stay in step;
            # N_ID is derived from the column list rather than written as
            # a literal, which is what let a sixth identity column slip
            # past a hard-coded [5:].
            writer.writerow([dropout, lr, args.window_seconds, args.loss, i,
                             "|".join(val_pids), seed] +
                            [r.get(c, "") for c in out_columns[N_ID:]])
            fh.flush()
            print(f"[{n_done}/{len(tasks)}] dropout={dropout} lr={lr:g} "
                  f"fold={i}{val_pids} seed={seed}\n" + report(t, r), flush=True)
    except KeyboardInterrupt:
        # Drop everything not yet started, so Ctrl-C stops in seconds rather than
        # after the whole remaining queue drains. Runs already in flight still
        # finish (they cannot be killed through a Future); their rows are already
        # on disk, and the resume key skips them next time either way.
        print(f"\n[interrupt] cancelling {sum(f.cancel() for f in futs)} queued run(s); "
              f"waiting for the {args.jobs} in flight to finish...", flush=True)
    finally:
        ex.shutdown(wait=True)
        stop_hb.set()       # the watchdog is a daemon, but stop it deterministically
        fh.close()          # never left open: rows are flushed per-run, but the
                            # handle must not leak if the loop exits abnormally
    if n_failed:
        print(f"[warn] {n_failed} run(s) failed and were not recorded; re-run the same "
              f"command to retry just those (completed runs are skipped).")
    summarize(results_csv, outdir, rank_on, args.in_jsonl, adapt_ks,
              select_window=args.window_seconds, select_loss=args.loss,
              adapt_tau=args.adapt_tau if args.adapt_eval else None)


def summarize(results_csv: pathlib.Path, outdir: pathlib.Path,
              rank_on: str = "set_mae", label_path: str = "",
              adapt_ks: Optional[List[int]] = None,
              select_window: Optional[float] = None,
              select_loss: Optional[str] = None,
              adapt_tau: Optional[float] = None) -> None:
    """Rank configurations, pick the winner and E*, write selected_population.json.

    ``rank_on`` selects WHICH validation quantity decides both the winning
    configuration and E*:

    * ``set_mae`` (default, historical) - the UNADAPTED held-out-driver score.
    * ``adapt_set_mae`` - the score after the deployed per-driver adaptation,
      available only for runs trained with ``--adapt-eval``.

    The second is the criterion that matches what the population model is FOR:
    it is never served unadapted, and both ``xlstm_maml.evaluate_adaptation``
    and ``sweep_l2sp_tau`` already select on their own post-adaptation numbers.
    Switching changes E* as well as the ranking, which is the point - the epoch
    that minimizes source error need not be the one that adapts best, and stage 2
    consumes E* with no validation set of its own to catch a bad value.

    Both column sets are always written, so a completed sweep can be re-ranked
    either way without retraining.

    ``label_path``/``adapt_ks`` are needed only for ``adapt_set_mae``, to rebuild
    the matched per-driver floor (``adapt_fold_baseline``).

    ``select_window``/``select_loss`` RESTRICT which configurations may win. A
    results file is allowed to hold several (window_seconds, loss) groups — the
    ``--window-seconds`` and ``--loss`` flags advertise that a corn and a ce sweep,
    or a 5 s and a 20 s sweep, can share one ``--outdir`` without their artifacts
    colliding. They do not collide, but they are DIFFERENT EXPERIMENTS, and ranking
    them against each other picks the easier group rather than the better
    configuration. Without these arguments a mixed file therefore refuses to
    produce ``selected_population.json`` instead of silently crowning a winner
    from the wrong group. The full table is always printed either way.
    """
    # DEFAULT_ADAPT_KS, never a second literal: the floor must be computed on the
    # same grid the runs adapted on.
    adapt_ks = list(adapt_ks) if adapt_ks else list(DEFAULT_ADAPT_KS)
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("[summary] no completed runs yet")
        return
    # The three columns that move together with the ranking quantity. Derived
    # once here so a caller cannot rank on one column and read E* off another.
    if rank_on == "adapt_set_mae":
        rank_col, argmin_col, one_se_col = ("smoothed_best_adapt_mae",
                                            "best_epoch_smoothed_adapt",
                                            "best_epoch_1se_adapt")
    else:
        rank_col, argmin_col, one_se_col = ("smoothed_best_set_mae",
                                            "best_epoch_smoothed", "best_epoch_1se")
    # ALL THREE columns are validated, not just the ranking one. They move
    # together by design, and filtering on rank_col alone left `float(r[argmin_col])`
    # below to raise on a row that had the score but not the epoch — after the whole
    # sweep had finished, which is the worst possible moment to lose the summary.
    def _complete(r: dict) -> bool:
        return all(str(r.get(c, "")).strip() not in ("", "nan", "None")
                   for c in (rank_col, argmin_col, one_se_col))

    usable = [r for r in rows if _complete(r)]
    if not usable:
        print(f"[summary] no runs carry a complete {rank_col}/{argmin_col}/{one_se_col} "
              f"triple. Runs must be trained with --adapt-eval before "
              f"--rank-on adapt_set_mae can be used.")
        return
    if len(usable) < len(rows):
        print(f"[summary] {len(rows) - len(usable)} run(s) are missing one of "
              f"{rank_col}/{argmin_col}/{one_se_col} and are excluded.")
    rows = usable
    print(f"[summary] ranking on {rank_col}")

    # Floors for the adapted ranking, keyed by fold. Recomputed from the label
    # file rather than read from the CSV: the per-driver constant depends on the
    # K grid, which is a property of the summarize call, not of the run.
    adapt_floors: Dict[str, float] = {}
    if rank_on == "adapt_set_mae" and label_path:
        try:
            lab = load_segment_labels(pathlib.Path(label_path))
            for r in rows:
                vp = r.get("val_pids", "")
                if vp and vp not in adapt_floors:
                    adapt_floors[vp] = adapt_fold_baseline(
                        lab, [x for x in vp.split("|") if x], adapt_ks)
        except Exception as exc:            # a missing label file must not kill the summary
            print(f"[summary][warn] could not compute the per-driver adaptation floor "
                  f"({exc}); falling back to the GLOBAL constant, which is the wrong "
                  f"reference for an adapted score - read 'vs const' with that in mind.")

    by_cfg: Dict[Tuple[float, float], List[dict]] = {}
    for r in rows:
        try:
            w = float(r.get("window_seconds", "") or "nan")
        except ValueError:
            w = float("nan")
        by_cfg.setdefault((float(r["dropout"]), float(r["lr"]), w,
                           r.get("loss", "") or "?"), []).append(r)

    print(f"\n{'dropout':>8} {'lr':>7} {'n':>4} {'mean':>7} {'sd':>6} {'se':>6} "
          f"{'E*(1se)':>8} {'argmin':>7} {'IQR':>11} {'const':>8} {'vs const':>8}"
          f"   (smoothed val {'ADAPTED ' if rank_on == 'adapt_set_mae' else ''}set-MAE)")
    table = []
    for (dropout, lr, win, lss), rs in sorted(
            by_cfg.items(), key=lambda kv: (str(kv[0][3]), kv[0][0], kv[0][1],
                                            -1e9 if kv[0][2] != kv[0][2] else kv[0][2])):
        v = np.array([float(r[rank_col]) for r in rs])
        e_arg = np.array([float(r[argmin_col]) for r in rs])
        # Older CSVs (written before the 1-SE rule) lack the column; fall back to
        # the argmin so a partially-completed sweep still summarizes rather than
        # crashing — but say so, because the two are not interchangeable.
        if all(one_se_col in r and r[one_se_col] != "" for r in rs):
            e_1se = np.array([float(r[one_se_col]) for r in rs])
        else:
            print("[warn] some rows predate the 1-SE rule; falling back to argmin epochs "
                  "for this config. Delete sweep_results.csv and re-run for a clean E*.")
            e_1se = e_arg
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        q1, q3 = np.percentile(e_1se, [25, 75])
        table.append({"dropout": dropout, "lr": lr, "window_seconds": win,
                      "loss": lss, "n": len(v),
                      "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "se": se, "e_star": int(np.median(e_1se)),
                      "e_argmin": int(np.median(e_arg)),
                      "e_iqr": (int(q1), int(q3))})
        # Mean of the per-fold constant floors the runs of this config were
        # measured against, so "mean" and "const" are on the same footing.
        if rank_on == "adapt_set_mae" and adapt_floors:
            # Matched floor: per-driver constant on the same support the model
            # adapted on. Never the global constant - see adapt_fold_baseline.
            cb = [adapt_floors[r.get("val_pids", "")] for r in rs
                  if r.get("val_pids", "") in adapt_floors]
        else:
            cb = [float(r["const_set_mae"]) for r in rs
                  if r.get("const_set_mae") not in (None, "")]
        table[-1]["const"] = float(np.mean(cb)) if cb else float("nan")
        table[-1]["delta"] = table[-1]["mean"] - table[-1]["const"]
        print(f"{lss:>5} {dropout:8.2f} {lr:7.0e} {win:6.4g} {len(v):4d} {v.mean():7.3f} "
              f"{table[-1]['sd']:6.3f} {se:6.3f} {table[-1]['e_star']:8d} "
              f"{table[-1]['e_argmin']:7d} {str(table[-1]['e_iqr']):>11}"
              f" {table[-1]['const']:8.3f} {table[-1]['delta']:+8.3f}")

    # Candidates for the WINNER — a subset of the printed table. Everything above
    # is shown for context; only one (window, loss) group may be selected from.
    groups = sorted({(t["window_seconds"], t["loss"]) for t in table},
                    key=lambda g: (str(g[1]), -1e9 if g[0] != g[0] else g[0]))
    if select_window is not None or select_loss is not None:
        cand = [t for t in table
                if (select_window is None
                    or t["window_seconds"] == select_window
                    or (t["window_seconds"] != t["window_seconds"]))   # NaN = pre-column row
                and (select_loss is None or t["loss"] in (select_loss, "?"))]
        if not cand:
            print(f"\n[summary] no configuration matches window={select_window} "
                  f"loss={select_loss}; groups present: {groups}. Nothing selected.")
            return
        if len(groups) > 1:
            print(f"\n[summary] {len(groups)} (window, loss) group(s) present {groups}; "
                  f"selecting within window={select_window} loss={select_loss} only.")
    elif len(groups) > 1:
        # Refusing beats guessing: this file mixes experiments, and whichever one
        # happens to be easier would win on a metric that is not comparable across
        # them. Re-run summarize with the group named, or split the --outdir.
        print(f"\n[summary] REFUSING to select: {results_csv.name} holds "
              f"{len(groups)} (window_seconds, loss) group(s) {groups}, which are "
              f"different experiments and are not comparable on one metric. The table "
              f"above is still valid per group. Pass select_window/select_loss (the CLI "
              f"does this from --window-seconds/--loss) or give each group its own "
              f"--outdir. No selected_population.json written.")
        return
    else:
        cand = table

    best = min(cand, key=lambda t: t["mean"])
    # Tie-break toward MORE regularization: within one standard error of the
    # winner, prefer the larger dropout (and then the smaller lr). Fixed in
    # advance so it is not a post-hoc choice, and it errs toward the model that
    # generalizes rather than the one that won a noisy comparison.
    within = [t for t in cand if t["mean"] <= best["mean"] + (best["se"] if best["se"] == best["se"] else 0.0)]
    pick = max(within, key=lambda t: (t["dropout"], -t["lr"]))
    if pick is not best:
        print(f"\n[tie-break] {len(within)} config(s) within 1 SE of the minimum "
              f"({best['mean']:.3f} +/- {best['se']:.3f}); taking the most regularized")

    # WHAT THIS RUN WAS SELECTED ON. These strings are read downstream —
    # run_lodo_population greps `epochs_rule` for "1 SE" and warns if it is absent —
    # so they are built from `rank_on` rather than hard-coded. They used to describe
    # the unadapted criterion unconditionally, which made an --adapt-eval sweep
    # write a record of a selection it had not performed.
    metric_name = ("post-adaptation validation set-MAE" if rank_on == "adapt_set_mae"
                   else "validation set-MAE")
    sel = {
        "dropout": pick["dropout"], "lr": pick["lr"],
        # NOT hard-coded: a duplicate "loss": "corn" key below this line used to
        # overwrite it, so every --loss ce sweep recorded itself as corn.
        "window_seconds": pick.get("window_seconds"), "loss": pick.get("loss"),
        "epochs": pick["e_star"],
        "rank_on": rank_on,
        "epochs_rule": (f"median over runs of the earliest epoch within 1 SE of the "
                        f"smoothed minimum of the {metric_name}"),
        "epochs_argmin_median": pick["e_argmin"],
        "epochs_1se_iqr": list(pick["e_iqr"]),
        # The LODO run trains on 11 drivers, this sweep on 10 -- ~10% more
        # segments, so ~10% more optimizer STEPS at the same epoch count. Recorded
        # so the transfer is visible rather than assumed.
        "n_train_drivers_at_selection": 10,
        "selected_on": (f"mean smoothed {metric_name} over "
                        f"{len({r.get('val_pids', '') for r in rows})} rotating fold(s) "
                        f"x {pick['n']} run(s) per configuration"),
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
    # WHICH floor `constant_floor_set_mae` holds. The key name is kept for
    # compatibility with anything already reading it, but the value is a different
    # quantity under the two criteria and the record has to say which.
    sel["floor_kind"] = ("per-driver constant fitted on the K-label adaptation support"
                         if rank_on == "adapt_set_mae" and adapt_floors
                         else "global constant prediction on the validation drivers")
    sel["constant_floor_set_mae"] = pick.get("const", float("nan"))
    sel["vs_constant_floor"] = pick.get("delta", float("nan"))
    sel["beats_constant"] = bool(pick.get("delta", 0.0) < 0)
    if rank_on == "adapt_set_mae":
        # The adaptation settings are part of what was selected: a different K grid
        # or tau is a different criterion, and E* is not transferable across them.
        sel["adapt_k"] = list(adapt_ks)
        sel["adapt_tau"] = adapt_tau
    out = outdir / "selected_population.json"
    out.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    # WHICH floor the warning is about. Under --rank-on adapt_set_mae the model
    # has already been handed the driver's first K labels, so the reference is
    # the per-driver constant those same labels buy for free — never the global
    # constant, which is a different game (see adapt_fold_baseline).
    floor_name = ("per-driver constant on the adaptation support"
                  if rank_on == "adapt_set_mae" and adapt_floors else "constant prediction")
    if not sel["beats_constant"]:
        print(f"\n[WARNING] the winning config does NOT beat a {floor_name} "
              f"({pick['mean']:.3f} vs {pick.get('const', float('nan')):.3f}). On this "
              f"cohort that is a property of the task, not necessarily of the run: an "
              f"oracle per-FUNCTION constant scores 1.259 against a global constant's "
              f"1.321, while an oracle per-DRIVER constant reaches 1.090 — LoA preference "
              f"is ~4x more about who is driving than about what is being asked. Expect "
              f"the population model to sit near the constant floor, and read the "
              f"personalization arms, not this number, as the result.")
    print(f"\n[selected] dropout={pick['dropout']} lr={pick['lr']:g} E*={pick['e_star']} "
          f"(1-SE rule; argmin would give {pick['e_argmin']}, IQR {pick['e_iqr']})")
    print(f"           mean smoothed val set-MAE {pick['mean']:.3f}")
    if pick["e_iqr"][1] - pick["e_iqr"][0] > max(5, 0.5 * pick["e_star"]):
        print(f"[warn] E* varies widely across runs (IQR {pick['e_iqr']}) — the optimum is "
              f"not well determined. Consider more seeds before committing to it.")
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
