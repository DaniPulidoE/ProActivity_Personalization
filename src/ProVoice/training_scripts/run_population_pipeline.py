r"""Population-sweep pipeline: four sweeps, in order, resumable at any point.

    1. corn_w10   soft-CORN, 10 s window, FULL grid          (3 dropout x 2 lr)
    2. corn_w5    soft-CORN,  5 s window, winner of stage 1
    3. corn_w20   soft-CORN, 20 s window, winner of stage 1
    4. ce_w10     softmax/CE, 10 s window, FULL grid         (the ablation)

Each stage writes to its OWN directory — its own ``sweep_results.csv``, its own
``runs/`` curve folder, its own ``selected_population.json``. Nothing is shared,
so a stage can be deleted and redone without touching the others, and the four
tables stay independently readable.

WHY THE WINDOWS ARE NOT A FULL GRID
-----------------------------------
Window length is a DATA CONTRACT, not a regularizer: it changes what the model
sees, not how hard it is pushed to fit. It should not interact strongly with
dropout or lr, so tuning those once at 10 s and reusing the winner at 5 s and
20 s costs 2 x 30 runs instead of 2 x 180 — about 9 h total instead of 20.
The assumption is worth one spot-check (shorter windows carry less signal per
sample, which could favour more dropout); if the 5 s or 20 s winner looks
anomalous, re-run that window with the full grid.

WHY CE IS LAST AND FULL-GRID
----------------------------
It is the ablation, not a candidate: the Laplace UQ layer refuses a non-CORN
head, so CE cannot be the deployed arm whatever it scores. It runs last so an
interruption costs the ablation rather than the result. It gets the full grid
because a handicapped ablation is worse than no ablation — if CE loses, it has
to lose at its own best setting. Use ``--ce-seeds`` to trade seeds for wall
clock if time runs short; the head effect measured so far (~0.10-0.14 MAE at a
fixed decoder) is larger than the seed noise.

RECOVERY
--------
Re-run the exact same command. Two independent layers:

  * WITHIN a stage — the sweep skips any (dropout, lr, window, loss, fold, seed)
    already in its CSV, so an interrupted stage continues from the next run.
  * ACROSS stages — a stage is "done" when its CSV holds every expected row.
    Completed stages are re-summarized (seconds) rather than retrained, and
    stages 2 and 3 refuse to start until stage 1 is genuinely complete, since
    they need its winner.

No state file: completion is derived from the artifacts themselves, so deleting
a stage's directory is all it takes to force a redo, and a crashed run cannot
leave the pipeline believing something finished that did not.

Usage::

    python -m ProVoice.training_scripts.run_population_pipeline \
        --in data/labeled_data.jsonl --outdir results/pop_pipeline

    # resume after an interruption: the identical command
    # inspect without running anything:
    python -m ProVoice.training_scripts.run_population_pipeline --outdir ... --status
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional

from ProVoice.models.train_XLSTM import cache_name
from ProVoice.models.xlstm_model import DEFAULT_RESAMPLE_HZ
from ProVoice.training_scripts.folds import VALIDATION_FOLDS
from ProVoice.training_scripts.sweep_population_hparams import DROPOUTS, LRS, SEEDS

# name, loss, window_seconds, grid: "full" | "inherit" (winner of the named stage)
STAGES = [
    {"name": "corn_w10", "loss": "corn", "window": 10.0, "inherit": None},
    {"name": "corn_w5",  "loss": "corn", "window": 5.0,  "inherit": "corn_w10"},
    {"name": "corn_w20", "loss": "corn", "window": 20.0, "inherit": "corn_w10"},
    {"name": "ce_w10",   "loss": "ce",   "window": 10.0, "inherit": None},
]


def stage_dir(base: pathlib.Path, name: str) -> pathlib.Path:
    return base / name


def rows_done(outdir: pathlib.Path) -> int:
    csv_path = outdir / "sweep_results.csv"
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def expected_runs(dropouts: List[float], lrs: List[float],
                  seeds: List[int], n_folds: int) -> int:
    return len(dropouts) * len(lrs) * len(seeds) * n_folds


def read_winner(outdir: pathlib.Path) -> Optional[Dict]:
    p = outdir / "selected_population.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_stage(stage: Dict, base: pathlib.Path, args, dropouts: List[float],
              lrs: List[float], seeds: List[int]) -> bool:
    """Invoke one sweep. Returns True if the stage ended complete."""
    outdir = stage_dir(base, stage["name"])
    outdir.mkdir(parents=True, exist_ok=True)
    n_expected = expected_runs(dropouts, lrs, seeds, len(VALIDATION_FOLDS))
    n_before = rows_done(outdir)

    if n_before >= n_expected:
        print(f"[{stage['name']}] already complete ({n_before}/{n_expected} runs) — "
              f"re-summarizing only")
    else:
        print(f"[{stage['name']}] {n_before}/{n_expected} runs done; "
              f"loss={stage['loss']} window={stage['window']}s "
              f"dropouts={dropouts} lrs={lrs}", flush=True)

    cmd = [sys.executable, "-m", "ProVoice.training_scripts.sweep_population_hparams",
           "--in", args.in_jsonl, "--outdir", str(outdir),
           "--loss", stage["loss"], "--window-seconds", str(stage["window"]),
           "--dropouts", ",".join(str(d) for d in dropouts),
           "--lrs", ",".join(str(l) for l in lrs),
           "--seeds", ",".join(str(s) for s in seeds),
           "--epochs", str(args.epochs), "--patience", str(args.patience),
           "--min-select-epoch", str(args.min_select_epoch),
           "--jobs", str(args.jobs)]
    if args.threads_per_job > 0:
        cmd += ["--threads-per-job", str(args.threads_per_job)]
    # Each stage needs the cache built for ITS OWN window — the encoding differs,
    # and the trainer errors rather than accepting a mismatched one.
    if args.cache_dir:
        cache = pathlib.Path(args.cache_dir) / cache_name(stage["window"], args.resample_hz)
        if not cache.exists():
            print(f"[{stage['name']}] SKIPPED — no cache for this window at {cache}. Build it:\n"
                  f"    python -m scripts.build_segment_cache --in {args.in_jsonl} "
                  f"--outdir {args.cache_dir} --windows {stage['window']:g}")
            return False
        cmd += ["--cache", str(cache)]
    t0 = time.time()
    # Streamed, not captured: these stages run for hours and the caller needs to
    # see progress. A failure is caught by the return code and by the row count.
    proc = subprocess.run(cmd)
    dt = time.time() - t0
    n_after = rows_done(outdir)
    ok = proc.returncode == 0 and n_after >= n_expected
    print(f"[{stage['name']}] {'COMPLETE' if ok else 'INCOMPLETE'} — "
          f"{n_after}/{n_expected} runs, {dt / 60:.1f} min"
          + ("" if proc.returncode == 0 else f", exit {proc.returncode}"))
    return ok


def summarize_pipeline(base: pathlib.Path) -> None:
    """One table across all four stages, plus a machine-readable roll-up."""
    out = []
    for st in STAGES:
        w = read_winner(stage_dir(base, st["name"]))
        if not w:
            continue
        out.append({"stage": st["name"], "loss": st["loss"], "window_seconds": st["window"],
                    **{k: w.get(k) for k in
                       ("dropout", "lr", "epochs", "mean_smoothed_val_set_mae", "se",
                        "constant_floor_set_mae", "vs_constant_floor", "beats_constant",
                        "n_runs")}})
    if not out:
        print("\n[pipeline] nothing to summarize yet")
        return
    print(f"\n{'stage':>9} {'loss':>5} {'win_s':>6} {'drop':>5} {'lr':>7} {'E*':>4} "
          f"{'val MAE':>8} {'se':>6} {'const':>7} {'vs const':>9}")
    for r in out:
        print(f"{r['stage']:>9} {r['loss']:>5} {r['window_seconds']:>6g} "
              f"{r['dropout']:>5} {r['lr']:>7g} {r['epochs']:>4} "
              f"{r['mean_smoothed_val_set_mae']:>8.3f} {r['se']:>6.3f} "
              f"{r['constant_floor_set_mae']:>7.3f} {r['vs_constant_floor']:>+9.3f}")
    (base / "pipeline_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] -> {base / 'pipeline_summary.json'}")
    # The window comparison is the point of stages 1-3; say which won rather than
    # leaving it to be eyeballed off the table.
    corn = [r for r in out if r["loss"] == "corn"]
    if len(corn) > 1:
        best = min(corn, key=lambda r: r["mean_smoothed_val_set_mae"])
        print(f"[windows] best CORN window = {best['window_seconds']:g}s "
              f"(val set-MAE {best['mean_smoothed_val_set_mae']:.3f}); "
              f"others: " + ", ".join(
                  f"{r['window_seconds']:g}s {r['mean_smoothed_val_set_mae']:.3f}"
                  for r in corn if r is not best))
        if not best["beats_constant"]:
            print("[windows] NOTE none of them beats the constant floor — the window "
                  "choice is a comparison between models that all lose to 'always "
                  "predict one level'. Read it as such.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--outdir", default="results/pop_pipeline",
                    help="Parent directory; each stage gets a subdirectory under it.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-select-epoch", dest="min_select_epoch", type=int, default=3)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--ce-seeds", dest="ce_seeds", default="",
                    help="Seeds for the CE ablation only (default: same as --seeds). "
                         "Fewer seeds here is the cheapest place to buy wall clock.")
    ap.add_argument("--only", default="",
                    help="Comma-separated stage names to run (default: all four, in order).")
    ap.add_argument("--status", action="store_true",
                    help="Report progress and exit without running anything.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Concurrent training runs per stage. See the same flag on "
                         "sweep_population_hparams for the measurements: this model gains "
                         "little from torch intra-op threads (1.82x at 12), and concurrency "
                         "tops out around 2.0-2.2x because the spare cores are contended "
                         "rather than idle. '--jobs 4 --threads-per-job 4' is the sweet spot.")
    ap.add_argument("--threads-per-job", dest="threads_per_job", type=int, default=0,
                    help="Torch threads per run (0 = auto: cpu_count // jobs).")
    ap.add_argument("--cache-dir", dest="cache_dir", default="data/cache",
                    help="Directory of pre-encoded segment caches, one per window. Saves the "
                         "~24 s parse each of the 420 runs would otherwise repeat and keeps "
                         "concurrent runs small enough in RAM to be worth launching. Build "
                         "all three windows at once with scripts/build_segment_cache.py. "
                         "Pass '' to disable and read the JSONL directly.")
    ap.add_argument("--resample-hz", dest="resample_hz", type=float,
                    default=DEFAULT_RESAMPLE_HZ,
                    help="Only used to locate the cache file; the trainer's own default is "
                         "authoritative for training.")
    args = ap.parse_args()

    base = pathlib.Path(args.outdir)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ce_seeds = ([int(s) for s in args.ce_seeds.split(",") if s.strip()]
                if args.ce_seeds else seeds)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    if args.status:
        print(f"{'stage':>9} {'runs':>12}  state")
        for st in STAGES:
            d = stage_dir(base, st["name"])
            sd = ce_seeds if st["loss"] == "ce" else seeds
            grid_d, grid_l = list(DROPOUTS), list(LRS)
            note = ""
            if st["inherit"]:
                w = read_winner(stage_dir(base, st["inherit"]))
                if w:
                    grid_d, grid_l = [w["dropout"]], [w["lr"]]
                else:
                    # No winner yet, so the real grid (1 config) is unknown and
                    # the full-grid count is only an upper bound — say so rather
                    # than implying this stage is 6x the work it will be.
                    note = f"  (upper bound; becomes {len(sd) * len(VALIDATION_FOLDS)} " \
                           f"once '{st['inherit']}' picks a winner)"
            exp = expected_runs(grid_d, grid_l, sd, len(VALIDATION_FOLDS))
            got = rows_done(d)
            state = ("complete" if got >= exp else
                     "not started" if got == 0 else "partial")
            print(f"{st['name']:>9} {got:>5}/{exp:<6}  {state}{note}")
        summarize_pipeline(base)
        return

    base.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] {len(STAGES)} stages -> {base}")
    for st in STAGES:
        if only and st["name"] not in only:
            print(f"[{st['name']}] skipped (--only)")
            continue
        dropouts, lrs = list(DROPOUTS), list(LRS)
        if st["inherit"]:
            src = stage_dir(base, st["inherit"])
            # A stage that inherits must not start on a PARTIAL parent: the
            # winner would be chosen from whichever runs happened to finish
            # before the interruption, and stages 2/3 would then be tuned to an
            # artifact of when the process died.
            n_exp = expected_runs(list(DROPOUTS), list(LRS), seeds, len(VALIDATION_FOLDS))
            if rows_done(src) < n_exp:
                print(f"[{st['name']}] SKIPPED — parent '{st['inherit']}' is incomplete "
                      f"({rows_done(src)}/{n_exp}). Finish it first; its winner is this "
                      f"stage's grid.")
                continue
            w = read_winner(src)
            if not w:
                print(f"[{st['name']}] SKIPPED — no selected_population.json in {src}")
                continue
            dropouts, lrs = [w["dropout"]], [w["lr"]]
            print(f"[{st['name']}] inheriting dropout={w['dropout']} lr={w['lr']:g} "
                  f"from {st['inherit']}")
        run_stage(st, base, args, dropouts, lrs,
                  ce_seeds if st["loss"] == "ce" else seeds)

    summarize_pipeline(base)


if __name__ == "__main__":
    main()
