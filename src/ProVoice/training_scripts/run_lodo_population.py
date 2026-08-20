"""Stage 2 — leave-one-driver-out population models.

For each of the 12 drivers: train a population model on the OTHER 11 at the
hyperparameters stage 1 selected, for exactly E* epochs, with **no validation
set and no epoch selection**. Then score the held-out driver.

WHY NO VALIDATION SET
---------------------
Every fold would otherwise need drivers held back for epoch selection, which
either costs training data or (worse) tempts one to select on the test driver.
Fixing E* in advance removes the choice: each fold trains all 11 non-test
drivers for the same number of steps, and the only thing the held-out driver is
ever used for is the number reported at the end. Per-fold selection variance
disappears, and both study arms inherit the identical protocol.

The cost is that E* may be slightly wrong for a given fold. That is noise, not
bias — and it lands on both arms equally, since they share these checkpoints.

WHAT IS REPORTED
----------------
Two numbers per driver, side by side:

  * **full**  — set-MAE / set-accuracy over ALL of the held-out driver's
    segments: how the population model does on an unseen driver overall.
  * **tail**  — the same metrics over that driver's chronologically LAST
    ``--val-frac`` segments. This is the floor the L2-SP learning curves are
    measured against, because stage 3 scores every adapted head on exactly this
    tail. Comparing an adapted head against the *full*-set floor would compare
    two different measurements.

These are the UNADAPTED model's numbers — the K=0 point of the personalization
curve, and the baseline any arm has to beat.

OUTPUTS
-------
  ``trained_models/lodo/pop_heldout_<pid>.pt``  x12 — stage 3 adapts from these
  ``<outdir>/lodo_population.csv``               per-driver floor table

Usage::

    python -m ProVoice.training_scripts.run_lodo_population \\
        --in data/labeled_data.jsonl \\
        --selected results/pop_sweep/selected_population.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ProVoice.models.xlstm_model import load_checkpoint, logits_to_label
from ProVoice.models.train_XLSTM import (
    iter_jsonl, normalize_row, SeqDataset, make_collate,
    set_accuracy, set_mae, set_qwk, set_macro_f1,
)
from ProVoice.training_scripts.folds import lodo_folds, ALL_PIDS


# MEMORY. data/labeled_data.jsonl is 931 MB / 508,282 rows x 73 keys; parsing it
# into a list of dicts costs a MEASURED 4.0 GB. This process spawns a trainer
# subprocess that loads its own copy, so anything held here is doubled — holding
# `rows` across the 12-fold loop is what turns a 4.5 GB run into a 9 GB one.
#
# Nothing below ever materializes the whole file. The subset writer streams
# lines, and the per-driver scoring pass keeps only the ONE driver's normalized
# rows (~1/12th, and ~25 of the 73 keys).


def write_subset_streaming(src: pathlib.Path, pids: List[str], dst: pathlib.Path) -> int:
    """Copy the lines whose participantid is in ``pids``, without parsing the rest.

    Reads and writes line by line, so peak memory is one line (~2 KB) rather than
    the 4 GB a parsed list of the whole file would cost.
    """
    keep = set(pids)
    n = 0
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if str(obj.get("participantid", "")) in keep:
                fo.write(line if line.endswith("\n") else line + "\n")
                n += 1
    return n


def driver_rows(src: pathlib.Path, pid: str) -> List[dict]:
    """Normalized rows for ONE driver. Raw dicts are freed as they are normalized."""
    return [normalize_row(r) for r in iter_jsonl(src)
            if str(r.get("participantid", "")) == pid]


def present_pids(src: pathlib.Path) -> set:
    """Which participantids the file contains, without holding any of it."""
    return {str(r.get("participantid", "")) for r in iter_jsonl(src)}


@torch.no_grad()
def score(model, arch, df: pd.DataFrame, device: str) -> Tuple[dict, int]:
    """Set-aware metrics for ``model`` over every segment in ``df``."""
    ds = SeqDataset(df, context_length=arch["context_length"],
                    window_seconds=arch.get("window_seconds"),
                    resample_hz=arch.get("resample_hz"))
    if len(ds) == 0:
        return {"set_mae": float("nan"), "set_acc": float("nan"),
                "set_qwk": float("nan"), "set_macro_f1": float("nan")}, 0
    dl = DataLoader(ds, batch_size=32, shuffle=False,
                    collate_fn=make_collate(arch["context_length"]))
    head_type = arch.get("head_type", "softmax")
    yp, yl = [], []
    model.eval()
    for xb, lb, vb in dl:
        logits = model(xb.to(device), lengths=lb.to(device))
        yp.append(logits_to_label(logits, head_type).cpu().numpy())
        yl.append(vb.numpy())
    Yp, Yl = np.concatenate(yp), np.concatenate(yl)
    return ({"set_mae": set_mae(Yl, Yp), "set_acc": set_accuracy(Yl, Yp),
             "set_qwk": set_qwk(Yl, Yp, 5), "set_macro_f1": set_macro_f1(Yl, Yp, 5)},
            len(Yp))


def tail_df(df: pd.DataFrame, val_frac: float) -> pd.DataFrame:
    """The chronologically LAST ``val_frac`` of a driver's segments.

    Chronology is first appearance in the JSONL (``groupby(sort=False)``), the
    same convention ``sweep_train_frac.build_segments`` uses — NOT lexicographic
    segment_id order, so the two stages cut the tail at the same place.
    """
    gids = list(dict.fromkeys(df["segment_id"]))
    n_val = max(1, round(val_frac * len(gids)))
    return df[df["segment_id"].isin(set(gids[len(gids) - n_val:]))].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--selected", default="results/pop_sweep/selected_population.json",
                    help="Stage-1 output. --dropout/--lr/--epochs override it individually.")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="trained_models/lodo")
    ap.add_argument("--outdir", default="results/lodo")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.3,
                    help="Tail fraction scored separately. MUST match stage 3's --val-frac, "
                         "or the floor and the learning curves are measured on different "
                         "segments.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--window-seconds", dest="window_seconds", type=float, default=None,
                    help="Model input window. Falls back to the value recorded in "
                         "--selected. THIS IS A DATA CONTRACT, not a hyperparameter: it "
                         "sets context_length (window x resample_hz), so a checkpoint "
                         "built at one value cannot be adapted or served at another. It "
                         "was previously not passed at all, leaving the trainer on its "
                         "own default and silently producing 10 s checkpoints for a sweep "
                         "that may have selected 20 s.")
    ap.add_argument("--pids", default="", help="Comma-separated subset of test drivers to run.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Reuse a fold's checkpoint if it is already on disk (resume).")
    args = ap.parse_args()

    sel = {}
    selp = pathlib.Path(args.selected)
    if selp.exists():
        sel = json.loads(selp.read_text(encoding="utf-8"))
        print(f"[selected] {selp}: {sel.get('dropout')=} {sel.get('lr')=} {sel.get('epochs')=}")
    dropout = args.dropout if args.dropout is not None else sel.get("dropout")
    lr = args.lr if args.lr is not None else sel.get("lr")
    epochs = args.epochs if args.epochs is not None else sel.get("epochs")
    window_seconds = (args.window_seconds if args.window_seconds is not None
                      else sel.get("window_seconds"))
    if window_seconds is None:
        raise SystemExit(
            "No --window-seconds and none recorded in --selected. Refusing to fall back "
            "to the trainer's default: the resulting checkpoints would silently carry a "
            "context_length the rest of the pipeline does not expect.")
    print(f"[arch] window_seconds={window_seconds:g} "
          f"(context_length is derived from it)")
    if dropout is None or lr is None or epochs is None:
        raise SystemExit(
            "Need dropout, lr and epochs (E*). Run sweep_population_hparams first, or pass "
            "them explicitly. Training LODO folds at un-chosen hyperparameters would make "
            "the floor incomparable to everything downstream.")
    # E* is applied here with NO validation set and NO early stopping, so nothing
    # downstream can catch an epoch count past the overfitting knee. Two things
    # make that acceptable rather than reckless, and both are worth seeing:
    #   * E* uses the 1-SE rule (earliest epoch statistically indistinguishable
    #     from the smoothed optimum), so it errs toward under-training.
    #   * It was selected on 10-driver training sets and is applied to 11-driver
    #     ones -- ~10 % more segments, hence ~10 % more optimizer STEPS per epoch.
    # If the sweep's E* came from the argmin instead, say so loudly.
    if sel:
        rule = sel.get("epochs_rule", "")
        n_sel = sel.get("n_train_drivers_at_selection")
        if "1 SE" not in rule:
            print(f"[warn] E*={epochs} was not chosen by the 1-SE rule ({rule or 'unknown'}). "
                  f"With no validation set here, an E* at the argmin of a noisy curve can "
                  f"sit past the overfitting knee in every fold, undetected.")
        if n_sel:
            print(f"[note] E*={epochs} was selected on {n_sel}-driver training sets; these "
                  f"folds train on 11, i.e. ~{100 * (11 / n_sel - 1):.0f}% more optimizer "
                  f"steps per epoch at the same epoch count.")

    src = pathlib.Path(args.in_jsonl)
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    present = present_pids(src)
    if not present:
        raise SystemExit(f"no rows in {src}")
    missing = [p for p in ALL_PIDS if p not in present]
    if missing:
        print(f"[warn] drivers declared in folds.py but absent from the data: {missing}")

    want = set(p.strip() for p in args.pids.split(",") if p.strip()) or None
    ckpt_dir = pathlib.Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for test_pid, train_pids in lodo_folds():
        if test_pid not in present or (want and test_pid not in want):
            continue
        train_pids = [p for p in train_pids if p in present]
        assert test_pid not in train_pids, (
            f"held-out driver {test_pid} appears in its own training set — the LODO "
            f"estimate would be meaningless")
        ckpt = ckpt_dir / f"pop_heldout_{test_pid}.pt"

        if args.skip_existing and ckpt.exists():
            print(f"[fold {test_pid}] reusing existing {ckpt}")
        else:
            print(f"[fold {test_pid}] training on {len(train_pids)} drivers "
                  f"({epochs} epochs, no validation set)", flush=True)
            with tempfile.TemporaryDirectory() as td:
                sub = pathlib.Path(td) / "train.jsonl"
                n = write_subset_streaming(src, train_pids, sub)
                cmd = [sys.executable, "-m", "ProVoice.models.train_XLSTM",
                       "--in", str(sub), "--out", str(ckpt), "--loss", "corn",
                       "--no-val", "--epochs", str(epochs),
                       "--window-seconds", str(window_seconds),
                       "--dropout", str(dropout), "--lr", str(lr), "--seed", str(args.seed)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-10:]))
                    raise SystemExit(f"fold {test_pid} training failed (exit {proc.returncode})")
                print(f"  trained on {n} labelled rows -> {ckpt}")

        model, arch = load_checkpoint(str(ckpt))
        model.to(device)
        # ONE driver's normalized rows (~1/12th of the file, ~25 of its 73 keys),
        # re-streamed per fold rather than sliced out of a 4 GB in-memory copy.
        df = pd.DataFrame(driver_rows(src, test_pid))
        m_full, n_full = score(model, arch, df, device)
        m_tail, n_tail = score(model, arch, tail_df(df, args.val_frac), device)
        del df
        results.append({"pid": test_pid, "n_train_drivers": len(train_pids),
                        "n_seg_full": n_full, "n_seg_tail": n_tail,
                        **{f"full_{k}": v for k, v in m_full.items()},
                        **{f"tail_{k}": v for k, v in m_tail.items()}})
        print(f"  [{test_pid}] full: set-MAE={m_full['set_mae']:.3f} "
              f"set-acc={m_full['set_acc']:.3f} (n={n_full})   "
              f"tail: set-MAE={m_tail['set_mae']:.3f} "
              f"set-acc={m_tail['set_acc']:.3f} (n={n_tail})", flush=True)

    if not results:
        raise SystemExit("no folds ran")

    out_csv = outdir / "lodo_population.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\n{'driver':>7} {'full MAE':>9} {'full acc':>9} {'tail MAE':>9} {'tail acc':>9}")
    for r in results:
        print(f"{r['pid']:>7} {r['full_set_mae']:9.3f} {r['full_set_acc']:9.3f} "
              f"{r['tail_set_mae']:9.3f} {r['tail_set_acc']:9.3f}")
    # Mean over DRIVERS, not over segments: drivers contribute 94-136 segments
    # each, so a segment-weighted mean would quietly weight the long sessions.
    for scope in ("full", "tail"):
        mae = np.array([r[f"{scope}_set_mae"] for r in results], dtype=float)
        acc = np.array([r[f"{scope}_set_acc"] for r in results], dtype=float)
        se = mae.std(ddof=1) / np.sqrt(len(mae)) if len(mae) > 1 else float("nan")
        print(f"{scope:>7} mean over {len(mae)} drivers: set-MAE={mae.mean():.3f} "
              f"(SE {se:.3f})  set-acc={acc.mean():.3f}")
    print(f"\n[OK] -> {out_csv}")
    print(f"[next] stage 3 adapts from {ckpt_dir}/pop_heldout_<pid>.pt")


if __name__ == "__main__":
    main()
