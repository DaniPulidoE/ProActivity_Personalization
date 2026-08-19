r"""ANIL stage A — meta-training hyperparameter sweep.

    outer_lr in {3e-5, 1e-4, 3e-4, 1e-3}  x  augmentation in {off, on}
    x  6 rotating meta-validation folds of 2 drivers  x  2 seeds  =  96 runs

Writes ``selected_anil.json``: the winning ``(outer_lr, crop_frac, jitter_std)``
and **M\***, the meta-epoch count stage B then trains for with no meta-validation
at all. Structurally the mirror of ``sweep_population_hparams`` — same rotating
folds, same 1-SE rule, same frozen-then-apply discipline — so the two arms are
tuned by the same protocol and only their objective differs.

WHAT IS AND IS NOT SWEPT
------------------------
The arms must differ in exactly ONE thing: theta_init. Everything downstream is
already shared — ``head_adapt.adapt_head``, the same tau, the same 2000 steps,
the same temporal tail, the same folds. So this sweep tunes "how to produce a
better starting point", never "how to adapt".

  FIXED BY DESIGN      --tau (from the L2-SP sweep; see below), --val-adapt-steps,
                       --val-adapt-lr, --order imaml, --query-max.
                       These DEFINE the deployed adaptation and the arm.
  FIXED BY VERIFICATION --imaml-max-iter, --imaml-tol, --clip. Solver settings:
                       you check the residual, you do not tune it.
  ABLATIONS, ELSEWHERE --order second/first with --inner-steps/--inner-lr. Those
                       knobs are ignored under iMAML, so mixing them into this
                       grid would compare incomparable things.
  NOT WORTH AN AXIS    --fo-warmup-epochs. Derivative-order annealing exists to
                       tame instability from differentiating THROUGH the inner
                       trajectory; iMAML has no trajectory to differentiate
                       through, so the failure mode is structurally absent. Under
                       --order imaml the warm-up epochs silently run first-order
                       truncated SGD — a different objective — and silently
                       activate --inner-steps/--inner-lr, which the help calls
                       ignored. Left at 0.
  SWEPT                --outer-lr, augmentation, and M*.

TAU IS NOT AN ANIL HYPERPARAMETER
---------------------------------
It is the prior precision of the adaptation BOTH arms run, and it also defines
iMAML's inner problem. Re-tuning it here would make the arms differ in two ways
at once. It is read from the L2-SP sweep's ``selected_tau.json`` and frozen;
``--tau`` overrides only for a dry run before that sweep exists.

WARM START — WHY EACH ROTATION NEEDS ITS OWN POPULATION MODEL
-------------------------------------------------------------
Meta-training warm-starts from a population checkpoint. If that checkpoint were
trained on all 12 drivers, every meta-validation driver would already be in the
init, and the selection signal would be contaminated at the REPRESENTATION level
— a far worse leak than the one-scalar hyperparameter leak this design accepts.
So each rotation gets a population model trained on its own 10 non-val drivers,
built here if absent (cheap: E* epochs, no validation) and cached.

SELECTION METRIC
----------------
Meta-validation set-MAE, from ``evaluate_adaptation`` — which runs the DEPLOYED
adaptation over the true session PREFIX of held-out drivers. Not the
meta-training query loss: that is computed on ``--episode-start any`` episodes
and answers a different question. Both are recorded per epoch; only the
prefix-based one selects. A gap between them is expected and is NOT evidence of
meta-overfitting (docs/meta_optimization_options.md).

M* uses the same 1-SE rule as E*, for the same reason: stage B meta-trains with
no meta-validation, so an M* past the meta-overfitting knee goes undetected in
all 12 folds at once.

Usage::

    python -m ProVoice.training_scripts.sweep_anil_hparams \
        --in data/labeled_data.jsonl \
        --selected-tau results/l2sp_sweep/selected_tau.json \
        --selected-population results/pop_pipeline/corn_w10/selected_population.json \
        --outdir results/anil_sweep
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
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from ProVoice.training_scripts.blocked_stats import blocked_effects
from ProVoice.training_scripts.folds import (
    VALIDATION_FOLDS, train_pids_for_validation_fold,
)
from ProVoice.training_scripts.run_lodo_population import write_subset_streaming
from ProVoice.training_scripts.sweep_population_hparams import smooth, SMOOTH_WINDOW

OUTER_LRS = (3e-5, 1e-4, 3e-4, 1e-3)
# (crop_frac, jitter_std). "off" and a light "on"; augmentation is the designated
# meta-overfitting remedy and ships OFF by default, which is precisely why it
# belongs on an axis rather than being fixed by fiat.
AUGMENTATIONS = ((0.0, 0.0), (0.2, 0.02))
SEEDS = (0, 1)

RESULTS_COLUMNS = [
    "outer_lr", "crop_frac", "jitter_std", "tau", "fold", "val_pids", "seed",
    "best_val_mae",            # raw minimum of the meta-val curve
    "smoothed_best_val_mae",   # minimum of the smoothed curve — the RANKING quantity
    "best_epoch_smoothed",     # argmin of the smoothed curve
    "best_epoch_1se",          # earliest within 1 SE — M* comes from this
    "metrics_epoch",           # which epoch the *_at_best columns describe
    "epochs_run",
    "val_qwk_at_best",
    "val_acc_at_best",
    # The meta-TRAINING signal, reported alongside so a run that failed to train
    # is distinguishable from one that trained fine and did not transfer.
    "query_loss_first", "query_loss_at_best", "query_loss_last",
    "query_loss_drop",         # first - last: did meta-training progress at all?
    # iMAML solve health. A biased implicit gradient invalidates the run, and it
    # is invisible in the val curve.
    "inner_res_max", "inner_unconverged_frac",
]
ID_COLUMNS = ["outer_lr", "crop_frac", "jitter_std", "tau", "fold", "val_pids", "seed"]
N_ID = len(ID_COLUMNS)
assert RESULTS_COLUMNS[:N_ID] == ID_COLUMNS
assert len(set(RESULTS_COLUMNS)) == len(RESULTS_COLUMNS)


def parse_val_ks(spec: str) -> List[int]:
    try:
        ks = sorted({int(t) for t in spec.split(",") if t.strip()})
    except ValueError as e:
        raise SystemExit(f"--val-ks must be comma-separated integers, got {spec!r}") from e
    if not ks or ks[0] < 1:
        raise SystemExit(f"--val-ks must be positive integers, got {ks}")
    return ks


def results_columns(val_ks: List[int]) -> List[str]:
    """Header for the results table, widened by the meta-validation K grid.

    The per-K columns are what make the ranking auditable. A single mean over
    {5,10,20,30,60} can improve because the arm got better at K=60 while getting
    worse at K=5 -- opposite conclusions for the deployment story -- and the mean
    cannot distinguish those. They are also the only way to compare this arm's
    shape against the L2-SP learning curve, since the two use different
    evaluation protocols and so are comparable in SHAPE but not in level.
    """
    return (RESULTS_COLUMNS
            + [f"val_set_mae_k{k}" for k in val_ks]
            + [f"val_set_acc_k{k}" for k in val_ks])


def read_json(p: pathlib.Path) -> Optional[Dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_tag(outer_lr: float, crop: float, jit: float, val_pids: List[str], seed: int) -> str:
    return f"olr{outer_lr:g}_c{crop:g}_j{jit:g}_f{'-'.join(val_pids)}_s{seed}"


def read_done(path: pathlib.Path) -> set:
    """Completed ``(outer_lr, crop, jitter, tau, fold, seed)`` keys, for resuming.

    tau is PART of the key even though it is meant to be frozen. It is recorded
    per row, so a results file can legitimately contain rows from more than one
    tau (a dry run at --tau, then the real run off selected_tau.json). Without it
    in the key the second run would skip every row of the first as "already
    done" and silently report a sweep conducted at the wrong prior precision.
    """
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((float(row["outer_lr"]), float(row["crop_frac"]),
                          float(row["jitter_std"]), float(row["tau"]),
                          int(row["fold"]), int(row["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def ensure_fold_population(in_jsonl: str, fold_idx: int, val_pids: List[str],
                           pop_cfg: Dict, ckpt_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Population model for one rotation: trained on the 10 NON-validation drivers.

    Not shared with stage 2's LODO checkpoints — those hold out ONE driver, these
    hold out the fold's TWO. Using an all-12 model instead would put every
    meta-validation driver inside the warm start, contaminating the selection
    signal at the representation level.
    """
    # Keyed by the population CONFIG as well as the fold. Keying on the fold
    # alone means a cached model built under a different window/dropout/lr/E*
    # is silently reused — and since it is a warm start, every meta-training run
    # downstream would then begin from a model that does not match the config
    # this sweep claims to be using. Cheap to rebuild; impossible to notice.
    sig = (f"w{pop_cfg.get('window_seconds', 10.0):g}"
           f"_d{pop_cfg['dropout']:g}_lr{pop_cfg['lr']:g}_e{pop_cfg['epochs']}"
           f"_{pop_cfg.get('loss', 'corn')}")
    ckpt = ckpt_dir / f"pop_metaval_{fold_idx}_{'-'.join(val_pids)}_{sig}.pt"
    if ckpt.exists():
        return ckpt
    train_pids = train_pids_for_validation_fold(val_pids)
    print(f"  [warm-start] building population model for fold {fold_idx} "
          f"on {len(train_pids)} drivers ({pop_cfg['epochs']} epochs)", flush=True)
    with tempfile.TemporaryDirectory() as td:
        sub = pathlib.Path(td) / "train.jsonl"
        n = write_subset_streaming(pathlib.Path(in_jsonl), train_pids, sub)
        cmd = [sys.executable, "-m", "ProVoice.models.train_XLSTM",
               "--in", str(sub), "--out", str(ckpt),
               "--loss", pop_cfg.get("loss", "corn"),
               "--no-val", "--epochs", str(pop_cfg["epochs"]),
               "--dropout", str(pop_cfg["dropout"]), "--lr", str(pop_cfg["lr"]),
               "--window-seconds", str(pop_cfg.get("window_seconds", 10.0)),
               # Fixed seed on purpose: the warm start is held CONSTANT across
               # ANIL seeds so that seed variance measures meta-training
               # variance alone, not population-training variance on top.
               "--seed", "0"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
            return None
        print(f"  [warm-start] {n} rows -> {ckpt.name}")
    return ckpt


def curve_stats(rows: List[dict], min_select_epoch: int,
                val_ks: List[int]) -> Optional[Dict[str, float]]:
    """Per-run numbers from one meta-training curve. Mirrors the population sweep."""
    rows = [r for r in rows if r.get("val_set_mae") not in ("", None)]
    if not rows:
        return None
    mae = np.array([float(r["val_set_mae"]) for r in rows])
    ql = np.array([float(r["query_loss"]) for r in rows])
    sm = smooth(mae, SMOOTH_WINDOW)
    lo = min(int(min_select_epoch), len(sm) - 1)
    j = lo + int(np.argmin(sm[lo:]))
    resid = mae - sm
    sigma = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0
    se = sigma / np.sqrt(min(SMOOTH_WINDOW, len(mae)))
    within = np.flatnonzero(sm[lo:] <= sm[j] + se)
    j1 = lo + int(within[0]) if within.size else j

    def num(col, idx=None, default=float("nan")):
        try:
            return float(rows[j if idx is None else idx][col])
        except (KeyError, TypeError, ValueError):
            return default

    res = [float(r["inner_res_max"]) for r in rows if r.get("inner_res_max") not in ("", None)]
    unc = [float(r["inner_unconverged"]) for r in rows
           if r.get("inner_unconverged") not in ("", None)]
    nep = [float(r["n_episodes"]) for r in rows if r.get("n_episodes") not in ("", None)]
    return {
        "best_val_mae": float(mae.min()),
        "smoothed_best_val_mae": float(sm[j]),
        "best_epoch_smoothed": int(rows[j]["epoch"]),
        "best_epoch_1se": int(rows[j1]["epoch"]),
        "metrics_epoch": int(rows[j]["epoch"]),
        "epochs_run": len(rows),
        "val_qwk_at_best": num("val_set_qwk"),
        "val_acc_at_best": num("val_set_acc"),
        # Per-K, read at the SELECTED epoch -- the same row every other
        # *_at_best column describes, so the breakdown and the mean cannot
        # describe different points on the curve.
        **{f"val_set_mae_k{k}": num(f"val_set_mae_k{k}") for k in val_ks},
        **{f"val_set_acc_k{k}": num(f"val_set_acc_k{k}") for k in val_ks},
        "query_loss_first": float(ql[0]),
        "query_loss_at_best": float(ql[j]),
        "query_loss_last": float(ql[-1]),
        "query_loss_drop": float(ql[0] - ql[-1]),
        "inner_res_max": max(res) if res else float("nan"),
        "inner_unconverged_frac": (sum(unc) / sum(nep)) if unc and sum(nep) else float("nan"),
    }


def run_one(in_jsonl: str, init_ckpt: pathlib.Path, outer_lr: float, crop: float,
            jit: float, tau: float, val_pids: List[str], seed: int, args,
            workdir: pathlib.Path) -> Optional[Dict[str, float]]:
    tag = run_tag(outer_lr, crop, jit, val_pids, seed)
    out = workdir / f"anil_{tag}.pt"        # written, then discarded — stage B retrains
    mcsv = workdir / f"metrics_{tag}.csv"   # KEPT: this is the curve M* comes from
    cmd = [sys.executable, "-m", "ProVoice.models.xlstm_maml",
           "--in", in_jsonl, "--init", str(init_ckpt), "--out", str(out),
           "--metrics-csv", str(mcsv),
           # Explicit, never relying on defaults — the same discipline the
           # population sweep applies to --loss.
           "--order", "imaml", "--episode-start", args.episode_start,
           "--tau", str(tau),
           "--outer-lr", str(outer_lr), "--outer-opt", args.outer_opt,
           "--crop-frac", str(crop), "--jitter-std", str(jit),
           "--val-pids", ",".join(val_pids), "--seed", str(seed),
           "--meta-epochs", str(args.meta_epochs), "--episodes", str(args.episodes),
           "--meta-batch", str(args.meta_batch), "--patience", str(args.patience),
           "--k-min", str(args.k_min), "--k-max", str(args.k_max),
           # Explicit: this decides what val_set_mae means, hence what M* and
           # the whole ranking are selected for. Never inherited from the
           # child's default.
           "--val-ks", args.val_ks,
           "--fo-warmup-epochs", "0"]
    # ARM SYMMETRY: threaded to the child rather than assumed, so a
    # plain and an augmented arm cannot be produced by the same command.
    if args.embed_fcd:
        cmd += ["--embed-fcd"]
    env = dict(os.environ)
    if getattr(args, "threads_per_job", 0) > 0:
        # In the ENVIRONMENT, not as a flag: torch sizes its pool at import, so a
        # flag parsed in the child's main() would be too late. Same mechanism the
        # population sweep uses.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            env[var] = str(args.threads_per_job)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"  [FAIL] {tag} (exit {proc.returncode})")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
        return None
    if not mcsv.exists():
        print(f"  [FAIL] {tag}: no metrics CSV")
        return None
    stats = curve_stats(list(csv.DictReader(mcsv.open("r", encoding="utf-8", newline=""))),
                        args.min_select_epoch, parse_val_ks(args.val_ks))
    if stats is None:
        print(f"  [FAIL] {tag}: no meta-validation rows in the curve")
        return None
    out.unlink(missing_ok=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--outdir", default="results/anil_sweep")
    ap.add_argument("--selected-tau", dest="selected_tau",
                    default="results/l2sp_sweep/selected_tau.json",
                    help="L2-SP sweep output. tau is READ from here and frozen — it is the "
                         "prior precision of the adaptation BOTH arms run, so re-tuning it "
                         "per arm would make them differ in two ways at once.")
    ap.add_argument("--selected-population", dest="selected_population",
                    default="results/pop_pipeline/corn_w10/selected_population.json",
                    help="Population sweep output: supplies dropout/lr/E*/window for the "
                         "per-rotation warm-start models.")
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the frozen tau. For a dry run BEFORE the L2-SP sweep "
                         "exists; the real run should read it from --selected-tau so both "
                         "arms provably share one value.")
    ap.add_argument("--pop-ckpt-dir", dest="pop_ckpt_dir", default="trained_models/anil_warmstart")
    ap.add_argument("--embed-fcd", dest="embed_fcd", action="store_true",
                    help="Meta-train an FCD-AUGMENTED head ([z_64 | FCD_12]) — passed straight through "
                         "to xlstm_maml. MUST MATCH THE L2-SP ARM. That arm was run "
                         "with --embed-fcd, so omitting it here makes the two arms "
                         "adapt heads of different width, and the comparison would "
                         "then measure more than theta_init — which is the one thing "
                         "it exists to isolate. compare_arms_k_curve refuses a "
                         "head-width mismatch, but only after the meta-training is "
                         "already spent.")
    ap.add_argument("--outer-lrs", default=",".join(f"{x:g}" for x in OUTER_LRS))
    ap.add_argument("--augs", default="0,0;0.2,0.02",
                    help="Semicolon-separated crop_frac,jitter_std pairs.")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--folds", default="", help="Fold indices to run (default: all 6).")
    ap.add_argument("--meta-epochs", dest="meta_epochs", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--meta-batch", dest="meta_batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--k-min", dest="k_min", type=int, default=5)
    ap.add_argument("--k-max", dest="k_max", type=int, default=10)
    ap.add_argument("--outer-opt", dest="outer_opt", default="nadam")
    ap.add_argument("--episode-start", dest="episode_start", default="any",
                    help="TRAINING episode start. 'any' is augmentation over episode "
                         "position and is the diversity defence against meta-overfitting; "
                         "meta-validation always uses the true session prefix regardless. "
                         "Passed explicitly because the default encodes no decision.")
    ap.add_argument("--summarize-only", dest="summarize_only", action="store_true",
                    help="Re-rank an existing results CSV and rewrite selected_anil.json "
                         "without running anything. The ranking is a pure function of the "
                         "table, so a change to how configs are compared must not cost "
                         "another overnight sweep.")
    ap.add_argument("--val-ks", dest="val_ks", default="5,10,20,30,60",
                    help="Meta-validation support sizes, passed to xlstm_maml. The ranking "
                         "quantity is the flat mean over drivers x this grid, and the results "
                         "CSV gains a val_set_mae_k<K>/val_set_acc_k<K> pair per K at the "
                         "selected epoch. Widening this is nearly free (the driver is embedded "
                         "once and sliced) and it is what stops the arm from being tuned "
                         "exclusively for the first three minutes of labelling. Changing it "
                         "changes the header, so a results file started under a different grid "
                         "is refused rather than appended to.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Concurrent meta-training runs. They are independent "
                         "subprocesses, so this is the same trade the population sweep "
                         "makes: throughput comes from COUNT rather than width, because "
                         "the model does not scale inside a process. Warm starts are "
                         "built SERIALLY before the pool opens, so two runs on the same "
                         "fold can never race to create the same checkpoint. Pair with "
                         "--threads-per-job so jobs x threads does not exceed the core "
                         "count. KEEP IT FIXED for a whole sweep: thread count changes "
                         "float reduction order, so resuming at a different setting mixes "
                         "two numeric regimes into one results table.")
    ap.add_argument("--threads-per-job", dest="threads_per_job", type=int, default=0,
                    help="Torch intra-op threads per run (0 = leave torch's default). "
                         "Set as an environment variable on the child, since torch sizes "
                         "its pool at import.")
    ap.add_argument("--min-select-epoch", dest="min_select_epoch", type=int, default=3)
    args = ap.parse_args()

    tau = args.tau
    if tau is None:
        sel = read_json(pathlib.Path(args.selected_tau))
        if not sel or "tau" not in sel:
            raise SystemExit(
                f"No tau. Run sweep_l2sp_tau first (expected {args.selected_tau}), or pass "
                f"--tau for a dry run. tau must be SHARED with the L2-SP arm: it is the "
                f"prior precision of the adaptation both arms run, and it defines iMAML's "
                f"inner problem.")
        tau = float(sel["tau"])
        print(f"[tau] {tau:g} (frozen, from {args.selected_tau})")
    else:
        print(f"[tau] {tau:g} (OVERRIDE — for the real run take it from the L2-SP sweep)")

    pop_cfg = read_json(pathlib.Path(args.selected_population))
    if not pop_cfg or "epochs" not in pop_cfg:
        raise SystemExit(
            f"Need the population config (expected {args.selected_population}) to build "
            f"per-rotation warm-start models. Run the population pipeline first.")
    print(f"[warm-start cfg] dropout={pop_cfg['dropout']} lr={pop_cfg['lr']:g} "
          f"E*={pop_cfg['epochs']} window={pop_cfg.get('window_seconds')}s")

    outer_lrs = [float(x) for x in args.outer_lrs.split(",") if x.strip()]
    augs = [tuple(float(v) for v in a.split(",")) for a in args.augs.split(";") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fold_idx = ([int(x) for x in args.folds.split(",") if x.strip()]
                if args.folds else list(range(len(VALIDATION_FOLDS))))

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "runs"; workdir.mkdir(exist_ok=True)
    ckpt_dir = pathlib.Path(args.pop_ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_csv = outdir / "anil_sweep_results.csv"
    done = read_done(results_csv)
    if done:
        print(f"[resume] {len(done)} run(s) already in {results_csv}; they will be skipped")

    total = len(outer_lrs) * len(augs) * len(fold_idx) * len(seeds)
    print(f"[plan] {len(outer_lrs)} outer_lr x {len(augs)} aug x {len(fold_idx)} fold x "
          f"{len(seeds)} seed = {total} meta-training runs (<= {args.meta_epochs} epochs)")

    if args.summarize_only:
        if not results_csv.exists():
            raise SystemExit(f"--summarize-only: {results_csv} does not exist")
        # tau comes from the TABLE, never from selected_tau.json. Re-ranking must
        # describe the runs that exist; reading the live tau file here would stamp
        # whatever tau was selected LATER onto rows produced under an earlier one,
        # and selected_anil.json is what run_lodo_anil trusts.
        ran = sorted({r["tau"] for r in csv.DictReader(
            results_csv.open("r", encoding="utf-8", newline="")) if r.get("tau")})
        if len(ran) != 1:
            raise SystemExit(
                f"--summarize-only: {results_csv} mixes tau values {ran}. Ranking across "
                f"them would compare configurations tuned against different anchors; "
                f"split the file by tau first.")
        csv_tau = float(ran[0])
        if csv_tau != tau:
            print(f"[tau] {csv_tau:g} (from the results table). NOTE: "
                  f"{args.selected_tau} currently says {tau:g} — the table wins, because "
                  f"these runs were meta-trained at {csv_tau:g}.")
        summarize(results_csv, outdir, csv_tau)
        return

    COLS = results_columns(parse_val_ks(args.val_ks))
    new = not results_csv.exists()
    if not new:
        # Appending rows of a different width to an existing file produces a
        # table no reader can parse and no one notices until the analysis.
        with results_csv.open("r", encoding="utf-8", newline="") as _f:
            hdr = next(csv.reader(_f), [])
        if hdr and hdr != COLS:
            extra, missing = set(hdr) - set(COLS), set(COLS) - set(hdr)
            raise SystemExit(
                f"{results_csv} was written with a different column set "
                f"(+{sorted(extra)} / -{sorted(missing)}). Almost certainly a different "
                f"--val-ks, which also makes the val_set_mae values incomparable, so "
                f"resuming into it would pool two rankings. Use a fresh --outdir, or "
                f"re-run the earlier rows under the new grid.")
    fh = results_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    if new:
        writer.writerow(COLS); fh.flush()

    # PHASE 1 -- warm starts, SERIALLY. One per fold, cached. Building them inside
    # the parallel phase would let two runs on the same fold race to write the same
    # checkpoint file, and the loser would meta-train from a half-written init.
    inits: Dict[int, pathlib.Path] = {}
    for i in fold_idx:
        ck = ensure_fold_population(args.in_jsonl, i, list(VALIDATION_FOLDS[i]),
                                    pop_cfg, ckpt_dir)
        if ck is None:
            print(f"[fold {i}] SKIPPED — warm-start model could not be built")
            continue
        inits[i] = ck

    # PHASE 2 -- the grid, flattened so --jobs dispatches across folds and not
    # only within the innermost seed loop.
    tasks = [(i, olr, crop, jit, seed)
             for i in fold_idx if i in inits
             for olr in outer_lrs
             for (crop, jit) in augs
             for seed in seeds
             if (olr, crop, jit, tau, i, seed) not in done]
    print(f"[run] {len(tasks)} run(s) to do of {total}; {max(1, args.jobs)} concurrent x "
          + (f"{args.threads_per_job} torch thread(s) each" if args.threads_per_job
             else "torch default threads"), flush=True)

    def work(t):
        i, olr, crop, jit, seed = t
        return t, run_one(args.in_jsonl, inits[i], olr, crop, jit, tau,
                          list(VALIDATION_FOLDS[i]), seed, args, workdir)

    # Only THIS thread touches the writer, so the CSV needs no lock: the pool
    # workers block on their subprocess and hand back the parsed curve.
    n_done = n_failed = 0
    ex = cf.ThreadPoolExecutor(max_workers=max(1, args.jobs))
    futs = {ex.submit(work, t): t for t in tasks}
    try:
        for fut in cf.as_completed(futs):
            n_done += 1
            try:
                (i, olr, crop, jit, seed), r = fut.result()
            except Exception as e:
                # One run must not take the batch down; at run 40 of 48,
                # re-raising would discard the 39 that succeeded.
                n_failed += 1
                print(f"[ERROR] {futs[fut]}: {type(e).__name__}: {e}", flush=True)
                continue
            if r is None:
                n_failed += 1
                continue
            val_pids = list(VALIDATION_FOLDS[i])
            writer.writerow([olr, crop, jit, tau, i, "|".join(val_pids), seed]
                            + [r[c] for c in COLS[N_ID:]])
            fh.flush()
            print(f"[{n_done}/{len(tasks)}] outer_lr={olr:g} crop={crop:g} "
                  f"jitter={jit:g} fold={i}{val_pids} seed={seed}\n"
                  f"      val set-MAE {r['smoothed_best_val_mae']:.3f} "
                  f"@epoch {r['best_epoch_smoothed']} (1se {r['best_epoch_1se']}) | "
                  f"query_loss {r['query_loss_first']:.3f} -> "
                  f"{r['query_loss_last']:.3f} (drop {r['query_loss_drop']:+.3f}) | "
                  f"inner_res {r['inner_res_max']:.1e}", flush=True)
    except KeyboardInterrupt:
        print(f"\n[interrupt] cancelling {sum(f.cancel() for f in futs)} queued run(s); "
              f"waiting for those in flight...", flush=True)
    finally:
        ex.shutdown(wait=True)
        fh.close()
    if n_failed:
        print(f"[warn] {n_failed} run(s) failed and were not recorded; re-run the same "
              f"command to retry just those (completed runs are skipped).")
    summarize(results_csv, outdir, tau)


def summarize(results_csv: pathlib.Path, outdir: pathlib.Path, tau: float) -> None:
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        print("[summary] no completed runs yet")
        return
    by: Dict[tuple, List[dict]] = {}
    for r in rows:
        by.setdefault((float(r["outer_lr"]), float(r["crop_frac"]), float(r["jitter_std"])),
                      []).append(r)

    paired = blocked_effects(
        (cfg, (r["fold"], r["seed"]), float(r["smoothed_best_val_mae"]))
        for cfg, rs in by.items() for r in rs
        if r.get("smoothed_best_val_mae") not in ("", None))
    if paired is None:
        print("[rank] design too unbalanced to block on fold — falling back to the RAW mean. "
              "Its se is dominated by between-fold difficulty, so treat the ranking as "
              "indicative only and re-run the missing cells.")
    else:
        nb = next(iter(paired.values()))[2]
        print(f"[rank] ranking on the FOLD-BLOCKED mean over {nb} complete (fold, seed) "
              f"block(s); 'val MAE' below is that estimate, 'raw' is the unblocked mean.")

    print(f"\n{'outer_lr':>9} {'crop':>5} {'jit':>5} {'n':>4} {'val MAE':>8} {'raw':>7} "
          f"{'sd':>6} {'se':>6} {'M*':>4} {'argmin':>7} {'qloss drop':>11} {'res':>8}")
    table = []
    for (olr, crop, jit), rs in sorted(by.items()):
        v = np.array([float(r["smoothed_best_val_mae"]) for r in rs])
        m1 = np.array([float(r["best_epoch_1se"]) for r in rs])
        ma = np.array([float(r["best_epoch_smoothed"]) for r in rs])
        qd = np.array([float(r["query_loss_drop"]) for r in rs])
        res = np.array([float(r["inner_res_max"]) for r in rs if r["inner_res_max"] not in ("", "nan")])
        raw_mean = float(v.mean())
        raw_se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        if paired is not None:
            mean, se, n_blocks = paired[(olr, crop, jit)]
        else:
            mean, se, n_blocks = raw_mean, raw_se, 0
        table.append({"outer_lr": olr, "crop_frac": crop, "jitter_std": jit, "n": len(v),
                      "mean": mean, "raw_mean": raw_mean, "raw_se": raw_se,
                      "n_blocks": n_blocks, "blocked": paired is not None,
                      "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "se": se,
                      "m_star": int(np.median(m1)), "argmin": int(np.median(ma)),
                      "qloss_drop": float(qd.mean()),
                      "res_max": float(res.max()) if res.size else float("nan")})
        t = table[-1]
        print(f"{olr:9.0e} {crop:5.2f} {jit:5.3f} {len(v):4d} {t['mean']:8.3f} "
              f"{raw_mean:7.3f} {t['sd']:6.3f} {se:6.3f} {t['m_star']:4d} {t['argmin']:7d} "
              f"{t['qloss_drop']:+11.3f} {t['res_max']:8.1e}")

    best = min(table, key=lambda t: t["mean"])
    # Tie-break toward MORE regularization, matching the population sweep: within
    # 1 SE, prefer augmentation on and then the smaller outer_lr.
    within = [t for t in table
              if t["mean"] <= best["mean"] + (best["se"] if best["se"] == best["se"] else 0.0)]
    pick = max(within, key=lambda t: (t["crop_frac"] + t["jitter_std"], -t["outer_lr"]))
    if pick is not best:
        print(f"\n[tie-break] {len(within)} config(s) within 1 SE; taking the more regularized")

    lrs_tried = sorted({t["outer_lr"] for t in table})
    edge = pick["outer_lr"] in (min(lrs_tried), max(lrs_tried)) and len(lrs_tried) > 1
    sel = {
        "outer_lr": pick["outer_lr"], "crop_frac": pick["crop_frac"],
        "jitter_std": pick["jitter_std"], "meta_epochs": pick["m_star"],
        "tau": tau, "order": "imaml",
        "meta_epochs_rule": "median over runs of the earliest epoch within 1 SE of the "
                            "smoothed meta-val minimum",
        "meta_epochs_argmin_median": pick["argmin"],
        "mean_val_set_mae": pick["mean"], "se": pick["se"], "n_runs": pick["n"],
        # Which estimator actually chose this, and how far it sits from the raw
        # mean. A large gap between them means the fold spread was doing the
        # ranking before, and any earlier selection off this grid is suspect.
        "ranking_estimator": ("fold-blocked" if pick["blocked"] else "raw mean (UNBLOCKED)"),
        "n_blocks": pick["n_blocks"],
        "raw_mean_val_set_mae": pick["raw_mean"], "raw_se": pick["raw_se"],
        "mean_query_loss_drop": pick["qloss_drop"],
        "worst_inner_residual": pick["res_max"],
        "outer_lr_on_grid_edge": bool(edge),
        "note": ("tau is FROZEN from the L2-SP sweep — both arms run the identical "
                 "adaptation, so only theta_init differs. M* is applied by run_lodo_anil "
                 "with NO meta-validation, hence the 1-SE rule. Selection is on the "
                 "prefix-based meta-val MAE; query_loss is reported only to show whether "
                 "meta-training progressed, and the two are computed on different episode "
                 "distributions by design."),
    }
    (outdir / "selected_anil.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(f"\n[selected] outer_lr={pick['outer_lr']:g} crop={pick['crop_frac']:g} "
          f"jitter={pick['jitter_std']:g} M*={pick['m_star']} "
          f"(val set-MAE {pick['mean']:.3f} +/- {pick['se']:.3f})")

    if edge:
        print("[WARNING] the winning outer_lr sits on a GRID EDGE. A null result from this "
              "arm would not be interpretable — extend the outer_lr range and re-run that "
              "axis before concluding meta-learning does not help.")
    if pick["qloss_drop"] <= 0:
        print("[WARNING] mean query loss did not DROP over meta-training. The meta-objective "
              "is not being optimized, so nothing downstream is testing meta-learning — "
              "check outer_lr and the inner-solve residual before reading any val number.")
    if pick["res_max"] == pick["res_max"] and pick["res_max"] > 1e-3:
        print(f"[WARNING] worst iMAML inner residual {pick['res_max']:.1e}: the implicit "
              f"gradient is only exact at the inner argmin, so a large residual means the "
              f"meta-gradient was biased. Raise --imaml-max-iter or --tau.")
    print(f"[OK] -> {outdir / 'selected_anil.json'}")


if __name__ == "__main__":
    main()
