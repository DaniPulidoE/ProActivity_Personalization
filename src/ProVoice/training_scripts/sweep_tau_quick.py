r"""Quick L2-SP anchor (tau) sweep - a go/no-go check before the population sweep.

    tau grid  x  K grid  x  1-2 folds,  on ONE trained backbone per fold.

WHY THIS RUNS FIRST
-------------------
``--adapt-tau`` defaults to 2.0 and that value is a PLACEHOLDER: the committed tau
comes from stage 3 (``sweep_l2sp_tau``), which needs stage-2 checkpoints, which
need stage 1, which is the sweep this check precedes. The circularity is real and
has to be broken somewhere.

It matters because tau controls how far adaptation is allowed to move off the
population head. ``head_adapt`` derives ``lambda = tau / 2K``, so a tau that is
too large pins the adapted head to the anchor and personalization does almost
nothing - which is indistinguishable, from the outside, from "this cohort has no
personalizable signal". Before spending ~5 hours on a 36-run population sweep at
a guessed tau, spend ~10 minutes finding out whether that guess cripples the
thing being measured.

WHAT IT IS NOT
--------------
Not a replacement for ``sweep_l2sp_tau``. That one runs all 12 drivers against
their own LODO checkpoints and produces the committed value with per-driver
learning curves. This one runs 1-2 folds against a quickly-trained backbone and
answers one question: **is the default tau in the right ballpark, or is it
strangling adaptation?** Treat the winner as a provisional input to stage 1, not
as the study's tau.

NO LEAK
-------
Each fold's backbone is trained on that fold's 10 TRAINING drivers and adapted on
its 2 held-out ones - the same guarantee ``sweep_l2sp_tau`` documents when it
insists driver d be adapted from ``pop_heldout_d.pt``. Adapting from a model that
has already seen the driver would make every curve optimistic AND tune tau for a
regime that never ships.

COST
----
The backbone is trained once per fold and CACHED (``--ckpt-dir``); re-runs at a
different tau grid cost nothing but the adaptation. Embeddings are computed once
per fold, so each (tau, K) cell is ~2000 AdamW steps on a (K x 64) tensor,
~3 s. A 7 x 4 grid on one fold is ~3 minutes of adaptation on top of one
training run.

Usage::

    python -m ProVoice.training_scripts.sweep_tau_quick \
        --in data/labeled_data.jsonl --cache data/cache/segments_w10_hz10.npz \
        --outdir results/tau_quick --folds 0 --epochs 12
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ProVoice.models.head_adapt import (
    DEFAULT_ADAPT_LR, DEFAULT_ADAPT_STEPS, DEFAULT_TAU, adapt_head_tensors,
)
from ProVoice.models.train_XLSTM import (
    SeqDataset, cache_meta, load_segment_cache, make_collate, set_accuracy, set_mae,
)
from ProVoice.models.xlstm_model import load_checkpoint, logits_to_probs, probs_to_label
from ProVoice.training_scripts.folds import VALIDATION_FOLDS, train_pids_for_validation_fold

# Spans BELOW the 2.0 placeholder, because the suspicion being tested is that the
# default anchor is too STRONG and is pinning the adapted head to the population
# one. If the winner lands at 0.1 the grid needs extending downward, which the
# summary flags as an edge win.
DEFAULT_TAUS: Tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
# Fewer K than the population sweep: this is a tau check, not a learning curve.
# K=5 is the smallest deployable budget and K=60 the largest (~25 min of
# labelling, matching sweep_l2sp_tau's k-cap).
DEFAULT_KS: Tuple[int, ...] = (5, 10, 30, 60)
MIN_QUERY = 20            # matches train_XLSTM._ADAPT_MIN_QUERY

RESULTS_COLUMNS = [
    "fold", "val_pids", "adapt_params", "tau", "K",
    "adapt_set_mae", "adapt_set_acc",
    "unadapted_set_mae",          # the same checkpoint, head un-touched
    "pdconst_set_mae",            # per-driver constant on the same K support
    "vs_pdconst", "vs_unadapted",
    "n_drivers", "grad_norm_max",
]


def train_fold_checkpoint(in_jsonl: str, cache: str, fold: Tuple[str, ...],
                          ckpt: pathlib.Path, epochs: int, dropout: float, lr: float,
                          seed: int, window_seconds: float, loss: str,
                          min_select_epoch: int) -> bool:
    """Train one backbone on the fold's TRAINING drivers. Cached on disk.

    A subprocess for the same reason the sweeps use one: a clean RNG state and a
    crash that costs this fold rather than the script. Unlike the sweeps, the
    checkpoint is KEPT - it is the whole input to the tau grid, and re-running at
    a different grid must not retrain.
    """
    if ckpt.exists():
        print(f"[ckpt] reusing {ckpt.name}")
        return True
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "ProVoice.models.train_XLSTM",
           "--in", in_jsonl, "--out", str(ckpt), "--loss", loss,
           "--val-pids", ",".join(fold),
           "--dropout", str(dropout), "--lr", str(lr), "--seed", str(seed),
           "--epochs", str(epochs), "--patience", str(epochs),
           "--min-select-epoch", str(min_select_epoch),
           "--window-seconds", str(window_seconds)]
    if cache:
        cmd += ["--cache", cache]
    print(f"[train] fold {'|'.join(fold)}: {epochs} epochs on "
          f"{len(train_pids_for_validation_fold(fold))} drivers ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not ckpt.exists():
        print(f"[train][FAIL] fold {'|'.join(fold)} (exit {proc.returncode})")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-10:]))
        return False
    print(f"[train] done in {time.time() - t0:.0f}s -> {ckpt.name}")
    return True


def embed_fold(model, cache: Dict, fold: Tuple[str, ...], context_length: int,
               device: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[str], Dict[str, int]]:
    """Pooled embeddings for the fold's held-out drivers, computed ONCE.

    Everything downstream is a linear head on these, so the backbone forward
    happens once per fold and every (tau, K) cell after it is a matmul. Returns
    ``(Z, V, pids, segment_ids, chrono_index)``; the last is first-appearance
    order in the source file, which is chronological within a driver and is what
    makes ``segs[:K]`` a genuine session prefix rather than a UUID sort.
    """
    ds = SeqDataset.from_cache(cache, context_length, pids=set(fold))
    if not len(ds):
        raise SystemExit(f"no segments for fold {fold} in the cache")
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=make_collate(context_length))
    zs, vs = [], []
    model.eval()
    with torch.no_grad():
        for xb, lb, vb in dl:
            h = model.backbone(model.in_proj(xb.to(device).to(torch.float32)))
            idx = (lb.to(h.device).long() - 1).clamp(min=0)
            zs.append(h[torch.arange(h.size(0), device=h.device), idx].detach().cpu())
            vs.append(vb)
    chrono = {str(s): i for i, s in enumerate(cache["seg_order"])}
    return (torch.cat(zs), torch.cat(vs).float(), np.asarray(ds.pids),
            list(ds.segment_ids), chrono)


def per_driver_order(pids: np.ndarray, sids: List[str],
                     chrono: Dict[str, int]) -> Dict[str, List[int]]:
    """Each driver's segment indices in CHRONOLOGICAL order."""
    out: Dict[str, List[int]] = {}
    for pid in np.unique(pids):
        where = np.flatnonzero(pids == pid)
        out[str(pid)] = sorted(where, key=lambda i: chrono.get(sids[i], len(chrono) + i))
    return out


def score(Z: torch.Tensor, V: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
          head_type: str) -> Tuple[float, float]:
    with torch.no_grad():
        probs = logits_to_probs(torch.nn.functional.linear(Z, w, b), head_type)
        pred = probs_to_label(probs, head_type).cpu().numpy()
    lv = V.numpy()
    return set_mae(lv, pred), set_accuracy(lv, pred)


def const_floor(V: torch.Tensor, sup: List[int], qry: List[int]) -> float:
    """Per-driver constant fitted on the support, scored on the query.

    The floor that binds once the driver's first K labels are in hand. Computed
    here rather than imported so it uses the exact support/query indices the
    model saw, not a re-derived approximation of them.
    """
    lv = V.numpy()
    s, q = lv[sup], lv[qry]
    c = int(np.argmin([set_mae(s, np.full(len(s), k, dtype=int)) for k in range(5)]))
    return set_mae(q, np.full(len(q), c, dtype=int))


def sweep_fold(fold: Tuple[str, ...], fold_i: int, model, cache: Dict, arch: Dict,
               taus: List[float], ks: List[int], adapt_params_list: List[str],
               steps: int, adapt_lr: float, device: str) -> List[dict]:
    head_type = arch.get("head_type", "corn")
    Z, V, pids, sids, chrono = embed_fold(model, cache, fold, arch["context_length"], device)
    order = per_driver_order(pids, sids, chrono)
    w0 = model.head.weight.detach().cpu()
    b0 = model.head.bias.detach().cpu()
    print(f"[fold {fold_i}] {'|'.join(fold)}: {len(Z)} segments, "
          f"{ {p: len(v) for p, v in order.items()} }")

    rows = []
    for ap in adapt_params_list:
        for tau in taus:
            for K in ks:
                maes, accs, unad, floors, gnorms = [], [], [], [], []
                for pid, idx in order.items():
                    if len(idx) < K + MIN_QUERY:
                        continue
                    sup, qry = idx[:K], idx[K:]
                    w, b, info = adapt_head_tensors(
                        Z[sup], V[sup], w0, b0, tau=tau, head_type=head_type,
                        steps=steps, lr=adapt_lr, adapt_params=ap)
                    m, a = score(Z[qry], V[qry], w, b, head_type)
                    maes.append(m); accs.append(a)
                    unad.append(score(Z[qry], V[qry], w0, b0, head_type)[0])
                    floors.append(const_floor(V, sup, qry))
                    gnorms.append(float(info.get("grad_norm", float("nan"))))
                if not maes:
                    continue
                r = {"fold": fold_i, "val_pids": "|".join(fold), "adapt_params": ap,
                     "tau": tau, "K": K,
                     "adapt_set_mae": float(np.mean(maes)),
                     "adapt_set_acc": float(np.mean(accs)),
                     "unadapted_set_mae": float(np.mean(unad)),
                     "pdconst_set_mae": float(np.mean(floors)),
                     "n_drivers": len(maes),
                     "grad_norm_max": float(np.nanmax(gnorms)) if gnorms else float("nan")}
                r["vs_pdconst"] = r["adapt_set_mae"] - r["pdconst_set_mae"]
                r["vs_unadapted"] = r["adapt_set_mae"] - r["unadapted_set_mae"]
                rows.append(r)
                print(f"    {ap:>4} tau={tau:<6g} K={K:<3} "
                      f"MAE={r['adapt_set_mae']:.3f}  "
                      f"vs pdconst {r['vs_pdconst']:+.3f}  "
                      f"vs unadapted {r['vs_unadapted']:+.3f}", flush=True)
    return rows


def summarize(rows: List[dict], adapt_params_list: List[str]) -> None:
    """tau x K table of the margin against the per-driver floor, then a verdict."""
    if not rows:
        print("[summary] nothing to summarize")
        return
    for ap in adapt_params_list:
        sub = [r for r in rows if r["adapt_params"] == ap]
        if not sub:
            continue
        taus = sorted({r["tau"] for r in sub})
        ks = sorted({r["K"] for r in sub})
        print(f"\n=== adapt_params={ap} : adapted set-MAE MINUS the per-driver floor ===")
        print("      " + "".join(f"{'K=' + str(k):>10}" for k in ks) + f"{'mean':>10}")
        best = None
        for tau in taus:
            cells = []
            for k in ks:
                v = [r["vs_pdconst"] for r in sub if r["tau"] == tau and r["K"] == k]
                cells.append(float(np.mean(v)) if v else float("nan"))
            m = float(np.nanmean(cells))
            flag = ""
            if best is None or m < best[1]:
                best, flag = (tau, m), ""
            print(f"tau={tau:<6g}" + "".join(f"{c:>10.3f}" for c in cells) + f"{m:>10.3f}{flag}")
        # Reported against BOTH references, because they answer different
        # questions: beating the floor means the model contributes more than a
        # per-driver constant; beating the unadapted head means adaptation did
        # anything at all.
        b_rows = [r for r in sub if r["tau"] == best[0]]
        print(f"\n  best tau = {best[0]:g}  (mean margin vs per-driver floor "
              f"{best[1]:+.3f}; vs unadapted head "
              f"{float(np.mean([r['vs_unadapted'] for r in b_rows])):+.3f})")
        edge = best[0] in (min(taus), max(taus))
        if edge:
            print(f"  [warn] the winner is at the EDGE of the grid - extend it past "
                  f"{best[0]:g} before trusting this value.")
        gmax = float(np.nanmax([r["grad_norm_max"] for r in sub]))
        if gmax > 1e-3:
            print(f"  [warn] max |grad| after adaptation is {gmax:.2e}; the Laplace layer "
                  f"expands about the MAP, so a large value means the inner problem did "
                  f"not converge and --steps should go up.")

    if len(adapt_params_list) > 1 and {"all", "bias"} <= set(adapt_params_list):
        # THE structural question, free here because both variants share this loop.
        a = float(np.mean([r["adapt_set_mae"] for r in rows if r["adapt_params"] == "all"]))
        b = float(np.mean([r["adapt_set_mae"] for r in rows if r["adapt_params"] == "bias"]))
        print(f"\n=== full head vs bias-only ===")
        print(f"  full head {a:.3f}   bias-only {b:.3f}   difference {a - b:+.3f}")
        if abs(a - b) < 0.02:
            print("  Bias-only MATCHES the full head: on this representation, adaptation is")
            print("  learning a per-driver LEVEL OFFSET and nothing else. Both study arms")
            print("  would then be tied by construction - see docs/embedding_informativeness.md")
            print("  section 4. Worth knowing before the LODO runs, not after.")
        else:
            print("  The full head beats bias-only, so adaptation is using the embedding")
            print("  rather than only shifting thresholds.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--cache", default="data/cache/segments_w10_hz10.npz")
    ap.add_argument("--outdir", default="results/tau_quick")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="",
                    help="Where fold backbones are cached (default <outdir>/ckpt). Kept "
                         "between runs: a different tau grid must not retrain.")
    ap.add_argument("--folds", default="0",
                    help="Fold indices from folds.VALIDATION_FOLDS. One is usually enough "
                         "for a ballpark check; folds differ a lot in difficulty, so use "
                         "two if the answer looks marginal.")
    ap.add_argument("--taus", default=",".join(f"{t:g}" for t in DEFAULT_TAUS))
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--adapt-params", dest="adapt_params", default="all",
                    choices=["all", "bias", "both"],
                    help="'all' adapts the full head (what ships). 'bias' adapts only the "
                         "K-1 CORN biases, which can shift the ordinal thresholds but "
                         "cannot reorder segments. 'both' runs the pair and reports the "
                         "difference - if they match, personalization on this "
                         "representation is a per-driver level offset and the arm "
                         "comparison is tied by construction.")
    ap.add_argument("--steps", type=int, default=DEFAULT_ADAPT_STEPS)
    ap.add_argument("--adapt-lr", dest="adapt_lr", type=float, default=DEFAULT_ADAPT_LR)
    # Backbone training (only when a fold's checkpoint is absent).
    ap.add_argument("--epochs", type=int, default=12,
                    help="Epochs for the throwaway fold backbone. Small on purpose: this "
                         "check needs a REPRESENTATIVE population model, not the final one. "
                         "E* from the real sweep is typically 5-13.")
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["corn", "ce"], default="corn")
    ap.add_argument("--window-seconds", dest="window_seconds", type=float, default=10.0)
    ap.add_argument("--min-select-epoch", dest="min_select_epoch", type=int, default=3)
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    fold_idx = [int(x) for x in args.folds.split(",") if x.strip()]
    adapt_params_list = ["all", "bias"] if args.adapt_params == "both" else [args.adapt_params]

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = pathlib.Path(args.ckpt_dir) if args.ckpt_dir else outdir / "ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} | taus={taus} | K={ks} | adapt_params={adapt_params_list}")
    print(f"[plan] {len(fold_idx)} fold(s) x {len(adapt_params_list)} x {len(taus)} tau x "
          f"{len(ks)} K = {len(fold_idx) * len(adapt_params_list) * len(taus) * len(ks)} "
          f"cell(s), ~3 s each after each fold's backbone is trained")

    all_rows: List[dict] = []
    for i in fold_idx:
        fold = VALIDATION_FOLDS[i]
        ckpt = ckpt_dir / f"tauquick_f{'-'.join(fold)}_d{args.dropout}_lr{args.lr}_" \
                          f"e{args.epochs}_s{args.seed}_{args.loss}.pt"
        if not train_fold_checkpoint(args.in_jsonl, args.cache, fold, ckpt, args.epochs,
                                     args.dropout, args.lr, args.seed,
                                     args.window_seconds, args.loss, args.min_select_epoch):
            continue
        model, arch = load_checkpoint(str(ckpt), map_location=device)
        model.to(device)
        cache = load_segment_cache(
            args.cache, cache_meta(args.in_jsonl, None,
                                   arch.get("window_seconds"), arch.get("resample_hz")))
        all_rows += sweep_fold(fold, i, model, cache, arch, taus, ks,
                               adapt_params_list, args.steps, args.adapt_lr, device)

    if not all_rows:
        raise SystemExit("no cells completed")
    out_csv = outdir / "tau_quick.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in RESULTS_COLUMNS})
    summarize(all_rows, adapt_params_list)
    print(f"\n[OK] -> {out_csv}")
    print("PROVISIONAL: feed the winner to the population sweep as --adapt-tau, and "
          "re-derive the committed value at stage 3 (sweep_l2sp_tau) over all 12 drivers.")


if __name__ == "__main__":
    main()
