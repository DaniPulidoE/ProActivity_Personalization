r"""ANIL stage B — leave-one-driver-out meta-training.

For each of the 12 drivers: meta-train on the OTHER 11 at stage A's frozen
configuration, for exactly M\* meta-epochs, with **no meta-validation set and no
epoch selection**. Produces the 12 meta-initializations stage C adapts from.

The exact mirror of ``run_lodo_population``, and deliberately so: the two arms
must be produced by the same protocol, or the comparison measures the protocol.

WHY NO META-VALIDATION HERE
---------------------------
Otherwise every fold would need drivers held back for epoch selection, which
either costs meta-training data (already only 11 drivers) or tempts selecting on
the test driver. Fixing M\* in advance removes the choice: per-fold selection
variance disappears and both arms inherit the identical discipline. The cost —
M\* may be slightly wrong for a given fold — is noise, not bias, and it lands on
both arms equally.

This is also why stage A extracts M\* with the 1-SE rule rather than the argmin:
nothing here can catch an M\* past the meta-overfitting knee.

WARM START
----------
Each fold warm-starts from **its own** stage-2 population checkpoint,
``pop_heldout_<pid>.pt`` — the model trained on the same 11 drivers. Using a
single all-12 population model would put the held-out driver into the ANIL
init while the L2-SP arm's init excludes it, which is both a leak and an
asymmetry between the arms.

That shared warm start is also what makes the **Population++ control** meaningful
(see the note at the end): ANIL gets the population model's training PLUS
meta-training, so "ANIL beat L2-SP" needs separating from "ANIL got more
gradient steps".

OUTPUTS
-------
  ``trained_models/lodo_anil/anil_heldout_<pid>.pt``  x12 — stage C adapts these
  ``<outdir>/lodo_anil.csv``                          per-fold provenance + timing

Note there is no floor table here, unlike stage 2: an ANIL init is not meant to
be good unadapted, and scoring it that way would invite a comparison the design
does not make. The arms meet in stage C, after adaptation.

Usage::

    python -m ProVoice.training_scripts.run_lodo_anil \
        --in data/labeled_data.jsonl \
        --selected results/anil_sweep/selected_anil.json \
        --pop-ckpt-dir trained_models/lodo \
        --ckpt-dir trained_models/lodo_anil
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

from ProVoice.training_scripts.folds import lodo_folds, ALL_PIDS
from ProVoice.training_scripts.run_lodo_population import (
    write_subset_streaming, present_pids,
)


def read_json(p: pathlib.Path) -> Optional[Dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--selected", default="results/anil_sweep/selected_anil.json",
                    help="Stage A output. Individual settings can be overridden below.")
    ap.add_argument("--pop-ckpt-dir", dest="pop_ckpt_dir", default="trained_models/lodo",
                    help="Stage 2 output: pop_heldout_<pid>.pt, the per-fold warm start.")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="trained_models/lodo_anil")
    ap.add_argument("--outdir", default="results/lodo_anil")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outer-lr", dest="outer_lr", type=float, default=None)
    ap.add_argument("--crop-frac", dest="crop_frac", type=float, default=None)
    ap.add_argument("--jitter-std", dest="jitter_std", type=float, default=None)
    ap.add_argument("--meta-epochs", dest="meta_epochs", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--meta-batch", dest="meta_batch", type=int, default=4)
    ap.add_argument("--k-min", dest="k_min", type=int, default=5)
    ap.add_argument("--k-max", dest="k_max", type=int, default=10)
    ap.add_argument("--outer-opt", dest="outer_opt", default="nadam")
    ap.add_argument("--episode-start", dest="episode_start", default="any")
    ap.add_argument("--pids", default="", help="Subset of test drivers to run.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Reuse a fold's meta-init if it is already on disk (resume).")
    args = ap.parse_args()

    sel = read_json(pathlib.Path(args.selected)) or {}
    if sel:
        print(f"[selected] {args.selected}")
    outer_lr = args.outer_lr if args.outer_lr is not None else sel.get("outer_lr")
    crop = args.crop_frac if args.crop_frac is not None else sel.get("crop_frac")
    jit = args.jitter_std if args.jitter_std is not None else sel.get("jitter_std")
    m_star = args.meta_epochs if args.meta_epochs is not None else sel.get("meta_epochs")
    tau = args.tau if args.tau is not None else sel.get("tau")
    missing = [n for n, v in (("outer_lr", outer_lr), ("crop_frac", crop),
                              ("jitter_std", jit), ("meta_epochs (M*)", m_star),
                              ("tau", tau)) if v is None]
    if missing:
        raise SystemExit(
            f"Missing {missing}. Run sweep_anil_hparams first, or pass them explicitly. "
            f"Meta-training LODO folds at un-chosen settings would make these inits "
            f"incomparable to the L2-SP arm and to each other.")
    print(f"[config] outer_lr={outer_lr:g} crop={crop:g} jitter={jit:g} "
          f"M*={m_star} tau={tau:g} order=imaml")
    if sel and "1 SE" not in str(sel.get("meta_epochs_rule", "")):
        print("[warn] M* was not chosen by the 1-SE rule. With no meta-validation here, an "
              "M* at the argmin of a noisy curve can sit past the meta-overfitting knee in "
              "every fold, undetected.")
    if sel.get("outer_lr_on_grid_edge"):
        print("[warn] the selected outer_lr sat on a GRID EDGE in stage A. A null result "
              "from this arm will not be interpretable — consider extending that axis first.")

    src = pathlib.Path(args.in_jsonl)
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    present = present_pids(src)
    want = set(p.strip() for p in args.pids.split(",") if p.strip()) or None
    pop_dir = pathlib.Path(args.pop_ckpt_dir)
    ckpt_dir = pathlib.Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for test_pid, train_pids in lodo_folds():
        if test_pid not in present or (want and test_pid not in want):
            continue
        train_pids = [p for p in train_pids if p in present]
        assert test_pid not in train_pids, (
            f"held-out driver {test_pid} is in its own meta-training set — the LODO "
            f"estimate would be meaningless")
        init = pop_dir / f"pop_heldout_{test_pid}.pt"
        if not init.exists():
            print(f"[fold {test_pid}] SKIPPED — no warm start at {init}. Run stage 2 "
                  f"(run_lodo_population) first; the ANIL init must start from the SAME "
                  f"population model the L2-SP arm uses for this fold.")
            continue
        out = ckpt_dir / f"anil_heldout_{test_pid}.pt"
        if args.skip_existing and out.exists():
            print(f"[fold {test_pid}] reusing existing {out.name}")
            results.append({"pid": test_pid, "n_train_drivers": len(train_pids),
                            "init": init.name, "out": out.name, "minutes": 0.0,
                            "reused": True})
            continue

        print(f"[fold {test_pid}] meta-training on {len(train_pids)} drivers "
              f"({m_star} meta-epochs, no meta-validation)", flush=True)
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            sub = pathlib.Path(td) / "train.jsonl"
            n = write_subset_streaming(src, train_pids, sub)
            cmd = [sys.executable, "-m", "ProVoice.models.xlstm_maml",
                   "--in", str(sub), "--init", str(init), "--out", str(out),
                   # No --val-pids: that is what makes this "no meta-validation,
                   # no epoch selection". xlstm_maml then saves every epoch, so
                   # the file left on disk is the FINAL one, which is what M*
                   # means here.
                   "--order", "imaml", "--episode-start", args.episode_start,
                   "--tau", str(tau), "--outer-lr", str(outer_lr),
                   "--outer-opt", args.outer_opt,
                   "--crop-frac", str(crop), "--jitter-std", str(jit),
                   "--meta-epochs", str(m_star), "--episodes", str(args.episodes),
                   "--meta-batch", str(args.meta_batch),
                   "--k-min", str(args.k_min), "--k-max", str(args.k_max),
                   "--fo-warmup-epochs", "0", "--seed", str(args.seed)]
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                raise SystemExit(f"fold {test_pid} meta-training failed (exit {proc.returncode})")
        dt = (time.time() - t0) / 60.0
        print(f"  {n} rows, {dt:.1f} min -> {out}")
        results.append({"pid": test_pid, "n_train_drivers": len(train_pids),
                        "init": init.name, "out": out.name, "minutes": round(dt, 2),
                        "reused": False})

    if not results:
        raise SystemExit("no folds ran")
    out_csv = outdir / "lodo_anil.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    ran = [r for r in results if not r["reused"]]
    print(f"\n[OK] {len(results)} meta-init(s) in {ckpt_dir}"
          + (f"; {sum(r['minutes'] for r in ran):.0f} min of meta-training" if ran else ""))
    print(f"[OK] -> {out_csv}")
    print(f"[next] stage C — the K curve for BOTH arms at tau={tau:g}:\n"
          f"    python -m ProVoice.training_scripts.compare_arms_k_curve \\\n"
          f"        --l2sp-ckpt-dir {pop_dir} --anil-ckpt-dir {ckpt_dir} --tau {tau:g}")
    print("[note] ANIL warm-starts from the population model, so it received the "
          "population arm's training PLUS meta-training. To separate 'meta-learning "
          "helped' from 'more optimization helped', add the compute-matched "
          "Population++ control: stage 2 re-run with a larger --epochs.")


if __name__ == "__main__":
    main()
