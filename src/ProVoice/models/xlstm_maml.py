# First-order ANIL meta-training for the xLSTM LoA model.
#
# ANIL (Raghu et al. 2020) = MAML with the inner loop restricted to the output
# head. The inner loop here is a few PROXIMAL SGD steps on the CORN/softmax
# head *including the L2-SP anchor term* — i.e. exactly the adaptation that is
# deployed per-driver by fine_tune_XLSTM.py — so meta-training optimizes the
# initialization for the adaptation procedure actually used at test time.
# The outer loop is FIRST-ORDER (FOMAML-style, no second derivatives):
#   - support embeddings are computed under no_grad (the gradient of the query
#     loss w.r.t. the backbone THROUGH the inner-loop trajectory is the
#     second-order term we deliberately drop),
#   - the query loss is backpropagated into backbone+in_proj through the query
#     forward pass, and the gradient w.r.t. the head INITIALIZATION is
#     approximated by the gradient at the ADAPTED head parameters.
# Head-only inner loop != meta-learning only the head: the outer loop still
# meta-trains the whole backbone for adaptability; that is what distinguishes
# the result from the joint-trained warm start it begins from.
#
# Episode design: support = temporally CONTIGUOUS block of
# K ∈ [k-min, k-max] segments, query = the segments immediately AFTER it —
# never a random support/query split, which would leak within-session
# autocorrelation. --episode-start prefix restricts support to the session
# prefix (exactly the deployment situation); the default 'any' treats every
# mid-session point as a pseudo session start, which preserves the
# support-before-query direction while giving combinatorial episode diversity
# from only ~15 drivers.
#
# Meta-overfitting defenses: warm start from the population checkpoint,
# meta-validation on held-out drivers with early stopping, many subsampled
# episodes per driver, and optional light augmentation (leading-frame crops,
# Gaussian jitter on the numerical feature dims).
#
# The saved checkpoint keeps the population checkpoint's arch contract
# (head_type, context_length, window_seconds) unchanged, so it is a drop-in
# replacement for state_xlstm.pt in fine_tune_XLSTM.py, sweep_train_frac.py,
# and the decision engine: the study comparison is L2-SP fine-tuning from the
# joint-trained init vs. the SAME fine-tuning from this meta-trained init.
#
# Usage:
#   python -m ProVoice.models.xlstm_maml \
#       --in data/labeled_data.jsonl \
#       --init trained_models/state_xlstm.pt \
#       --out trained_models/state_xlstm_anil.pt \
#       --val-pids p013,p014
import argparse, pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ProVoice.decision_engine import truncate_frames_by_seconds
from ProVoice.fcd_config import FCD_NAMES
from ProVoice.models.xlstm_model import (
    save_checkpoint,
    load_checkpoint,
    logits_to_probs,
    encode_frame,
    STATE_NUM,
    STATE_CARLA,
    STATE_CAT,
)
from ProVoice.models.xlstm_model import _as01
from coral_pytorch.losses import corn_loss

from ProVoice.models.train_XLSTM import (
    set_seed,
    read_jsonl,
    normalize_row,
    make_collate,
    mae,
    qwk,
)

LEVELS = [f"Level_{i}" for i in range(1, 6)]

# Feature dims eligible for jitter augmentation: the driver-state and CARLA
# numericals. FCD dims are static per task (jitter would fake nonexistent
# tasks) and the one-hot / length-encoded categoricals are not continuous.
_JITTER_SLICE = slice(len(FCD_NAMES), len(FCD_NAMES) + len(STATE_NUM) + len(STATE_CARLA))

Segment = Tuple[np.ndarray, int]  # (frames (T, D_IN) float32, LoA class 0-4)


def build_driver_segments(
    df: pd.DataFrame,
    window_seconds: float | None,
) -> Dict[str, List[Segment]]:
    """Encode one (X, y) pair per segment, grouped per driver, in chronological order.

    Chronology = first appearance in the JSONL (groupby(sort=False)), NOT the
    lexicographic order of segment_id strings — episode supports must be
    temporal prefixes/blocks. Segments with missing/all-zero Level_* labels
    are skipped and reported, never argmax'd into a bogus class-0 label.
    """
    if not all(k in df.columns for k in LEVELS):
        raise ValueError(f"Input data has no {LEVELS} columns; labels are required.")
    drivers: Dict[str, List[Segment]] = {}
    skipped = []
    for pid, pdf in df.groupby("participantid", sort=False):
        segs: List[Segment] = []
        for gid, g in pdf.groupby("segment_id", sort=False):
            g = g.reset_index(drop=True)
            lv = pd.to_numeric(g[LEVELS].iloc[0], errors="coerce").astype(float).values
            if np.isnan(lv).any() or lv.sum() <= 0:
                skipped.append(gid)
                continue
            y = int(np.argmax(lv))
            rows = [g.iloc[i].to_dict() for i in range(len(g))]
            rows = truncate_frames_by_seconds(rows, window_seconds)
            X = np.stack(
                [encode_frame(r.get("functionname") or "", r) for r in rows],
                axis=0,
            ).astype(np.float32)
            segs.append((X, y))
        if segs:
            drivers[str(pid)] = segs
    if skipped:
        print(f"[warn] skipped {len(skipped)} segment(s) with missing/empty Level_* labels: "
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    return drivers


def augment_segment(X: np.ndarray, rng: np.random.Generator,
                    crop_frac: float, jitter_std: float) -> np.ndarray:
    """Light task augmentation: random leading-frame crop + numerical jitter.

    Cropping drops frames from the FRONT (the readout is the last real frame,
    so this shortens the history window — same effect as a smaller
    window_seconds). Jitter perturbs only the continuous feature dims.
    """
    if crop_frac > 0.0 and X.shape[0] > 4:
        max_drop = int(crop_frac * X.shape[0])
        drop = int(rng.integers(0, max_drop + 1))
        if drop:
            X = X[drop:]
    if jitter_std > 0.0:
        X = X.copy()
        noise = rng.normal(0.0, jitter_std, size=X[:, _JITTER_SLICE].shape)
        X[:, _JITTER_SLICE] += noise.astype(np.float32)
    return X


def embed(model, xb: torch.Tensor, lb: torch.Tensor, device: str,
          grad: bool = False) -> torch.Tensor:
    """in_proj + backbone + last-real-frame readout (same readout as forward()).

    grad=False (support / eval): no graph — in first-order ANIL the backbone
    gradient through the inner-loop trajectory is exactly the term we drop.
    grad=True (query): the graph through the backbone is the ONLY path by
    which the outer loop trains the backbone.
    """
    with torch.enable_grad() if grad else torch.no_grad():
        h = model.backbone(model.in_proj(xb.to(device).to(torch.float32)))
        idx = (lb.to(h.device).long() - 1).clamp(min=0)
        return h[torch.arange(h.size(0), device=h.device), idx]


def adapt_head(Z: torch.Tensor, y: torch.Tensor,
               w0: torch.Tensor, b0: torch.Tensor,
               loss_fn, inner_steps: int, inner_lr: float, l2sp: float,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Proximal SGD on the head only: loss + λ·||θ − θ_meta||² — the deployed
    L2-SP adaptation with the meta head as anchor. Detaching after every step
    makes this first-order; the returned tensors are leaves whose .grad after
    a query-loss backward is the FOMAML head-initialization gradient.
    """
    anchor_w, anchor_b = w0.detach(), b0.detach()
    w = anchor_w.clone().requires_grad_(True)
    b = anchor_b.clone().requires_grad_(True)
    for _ in range(inner_steps):
        logits = F.linear(Z, w, b)
        loss = loss_fn(logits, y)
        loss = loss + l2sp * (((w - anchor_w) ** 2).sum() + ((b - anchor_b) ** 2).sum())
        gw, gb = torch.autograd.grad(loss, (w, b))
        w = (w - inner_lr * gw).detach().requires_grad_(True)
        b = (b - inner_lr * gb).detach().requires_grad_(True)
    return w, b


def adapt_head_deployed(Z: torch.Tensor, y: torch.Tensor,
                        w0: torch.Tensor, b0: torch.Tensor,
                        loss_fn, steps: int, lr: float, l2sp: float,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The DEPLOYED adaptation, for meta-validation: full-batch AdamW on the
    head with the L2-SP anchor — identical to fine_tune_XLSTM.py, whose
    mini-batch loop degenerates to one full-batch step per epoch whenever the
    support fits in a batch (K <= 16 always does). Full-batch => deterministic.
    """
    w = w0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)
    anchor_w, anchor_b = w0.detach(), b0.detach()
    opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=0.0)
    for _ in range(steps):
        loss = loss_fn(F.linear(Z, w, b), y)
        loss = loss + l2sp * (((w - anchor_w) ** 2).sum() + ((b - anchor_b) ** 2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def sample_episode(segs: List[Segment], rng: np.random.Generator,
                   k_min: int, k_max: int, query_max: int, start_mode: str,
                   ) -> Tuple[List[Segment], List[Segment]]:
    """Support = contiguous block of K segments, query = the ones right after it."""
    n = len(segs)
    k = int(rng.integers(k_min, k_max + 1))
    k = min(k, n - 1)  # always leave at least one query segment
    t = 0 if start_mode == "prefix" else int(rng.integers(0, n - k))
    return segs[t:t + k], segs[t + k:t + k + query_max]


def evaluate_adaptation(model, segs: List[Segment], K: int, collate, device: str,
                        loss_fn, head_type: str,
                        steps: int, lr: float, l2sp: float,
                        ) -> Tuple[float, float]:
    """Meta-validation on ONE held-out driver with the DEPLOYED adaptation:
    fine_tune_XLSTM-style AdamW on the head over the driver's first K segments
    (true session prefix), then MAE/QWK on everything after — the same
    temporal protocol as the sweep script."""
    xs, ys, ls = collate(segs[:K])
    Zs = embed(model, xs, ls, device)
    w, b = adapt_head_deployed(Zs, ys.to(device), model.head.weight, model.head.bias,
                               loss_fn, steps, lr, l2sp)
    xq, yq, lq = collate(segs[K:])
    Zq = embed(model, xq, lq, device)
    with torch.no_grad():
        pred = logits_to_probs(F.linear(Zq, w, b), head_type).argmax(dim=-1)
    Yt, Yp = yq.numpy(), pred.cpu().numpy()
    return mae(Yt, Yp), qwk(Yt, Yp, 5)


def main():
    ap = argparse.ArgumentParser(
        description="First-order ANIL meta-training of the xLSTM LoA model "
                    "(head-only proximal inner loop, full-model outer loop).")
    ap.add_argument("--in",     dest="in_jsonl", required=True,
                    help="Multi-driver labeled JSONL (needs participantid per row).")
    ap.add_argument("--init",   dest="init_pt", default="trained_models/state_xlstm.pt",
                    help="Population checkpoint to warm-start from (train_XLSTM.py output). "
                         "Warm starting is a meta-overfitting defense, not an optimization nicety.")
    ap.add_argument("--out",    dest="out_pt", default="trained_models/state_xlstm_anil.pt")
    ap.add_argument("--seed",   type=int, default=42)
    # --- episode design ---
    ap.add_argument("--k-min",  type=int, default=5)
    ap.add_argument("--k-max",  type=int, default=10,
                    help="Support size K is drawn uniformly from [k-min, k-max] per episode "
                         "(deployment regime: <10 labels ≈ 3 min of driving).")
    ap.add_argument("--query-max", type=int, default=20,
                    help="Cap on query segments per episode (the ones right after the support).")
    ap.add_argument("--episode-start", choices=["any", "prefix"], default="any",
                    help="'prefix': support is always the session's first K segments (exact "
                         "deployment match, few distinct episodes). 'any': support starts at a "
                         "random segment — a pseudo session start; more episode diversity.")
    # --- inner loop (must mirror the deployed L2-SP head adaptation) ---
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--inner-lr",    type=float, default=0.1,
                    help="Plain-SGD step size for the head inner loop (the head is a ~260-param "
                         "GLM; SGD needs a larger lr than fine_tune_XLSTM's AdamW).")
    ap.add_argument("--l2sp",  type=float, default=0.01,
                    help="λ of the proximal/L2-SP term inside the inner loop, anchoring the "
                         "adapted head to the meta head. Keep equal to the λ used by "
                         "fine_tune_XLSTM.py at deployment so meta-training matches it.")
    # --- outer loop ---
    ap.add_argument("--meta-epochs",  type=int, default=60)
    ap.add_argument("--episodes",     type=int, default=200,
                    help="Episodes sampled per meta-epoch (spread over the train drivers).")
    ap.add_argument("--meta-batch",   type=int, default=4,
                    help="Episodes averaged into one outer update.")
    ap.add_argument("--outer-lr",     type=float, default=1e-4,
                    help="AdamW lr for backbone+in_proj+head init. Keep small: this is a "
                         "warm-started refinement toward adaptability, not training from scratch.")
    ap.add_argument("--clip",         type=float, default=1.0, help="Grad-norm clip (0 disables).")
    # --- meta-validation / early stopping ---
    ap.add_argument("--val-pids", default="",
                    help="Comma-separated participant ids held out for meta-validation "
                         "(drive this externally for leave-one-driver-out). Empty: hold out "
                         "~20%% of drivers at random.")
    ap.add_argument("--val-adapt-steps", type=int, default=30,
                    help="Full-batch AdamW steps for meta-validation adaptation. Keep equal to "
                         "fine_tune_XLSTM's --epochs (30): with K support segments <= its batch "
                         "size, one fine-tune epoch = one full-batch step, so this reproduces "
                         "the deployed adaptation exactly.")
    ap.add_argument("--val-adapt-lr", type=float, default=2e-3,
                    help="AdamW lr for meta-validation adaptation. Keep equal to "
                         "fine_tune_XLSTM's --lr (2e-3) for the same reason.")
    ap.add_argument("--patience", type=int, default=10,
                    help="Early-stop after this many meta-epochs without val-MAE improvement.")
    # --- task augmentation ---
    ap.add_argument("--crop-frac",  type=float, default=0.0,
                    help="Max fraction of leading frames randomly dropped per segment (window crop).")
    ap.add_argument("--jitter-std", type=float, default=0.0,
                    help="Gaussian noise std on the numerical feature dims (features are ~[0,1]).")
    args = ap.parse_args()
    if not (1 <= args.k_min <= args.k_max):
        raise ValueError(f"Need 1 <= k-min <= k-max, got {args.k_min}, {args.k_max}")

    # seed and cuda
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # checkpoint properties: context_length, head_type, window_seconds
    model, arch = load_checkpoint(args.init_pt)
    model.to(device).train()
    context_length = arch["context_length"]
    head_type = arch.get("head_type", "softmax")
    window_seconds = arch.get("window_seconds")
    print(f"[model] warm start from {args.init_pt}: head_type={head_type} "
          f"context_length={context_length} window_seconds={window_seconds}")
    # CORN or softmax loss
    if head_type == "corn":
        loss_fn = lambda logits, target: corn_loss(logits, target, num_classes=model.n_classes)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # --- data: one segment list per driver, chronological ---
    rows = [normalize_row(r) for r in read_jsonl(pathlib.Path(args.in_jsonl))]
    if not rows:
        raise ValueError("JSONL is empty or contains no valid rows.")
    df = pd.DataFrame(rows)
    if "segment_id" not in df.columns or df["segment_id"].eq("").all():
        raise ValueError("Missing segment_id; cannot build sequences.")
    if df["participantid"].eq("").all():
        raise ValueError("participantid is missing from all rows; meta-learning needs "
                         "per-driver task boundaries.")

    # Fill missing values and cast to the right type for each feature category.
    for k in STATE_CAT:
        if k not in df.columns: df[k] = ""
        df[k] = df[k].fillna("").astype(str)
    for k in STATE_NUM:
        if k not in df.columns: df[k] = 0.0
        df[k] = df[k].apply(_as01)
    for k in STATE_CARLA:
        default = -1 if k == "speed_ratio_limit" else 0.0
        if k not in df.columns: df[k] = default
        df[k] = df[k].fillna(default)

    # build dataset
    drivers = build_driver_segments(df, window_seconds)
    min_segs = args.k_min + 1  # at least one query segment after the smallest support
    small = [p for p, s in drivers.items() if len(s) < min_segs]
    if small: # drop drivers with not enough segments
        print(f"[warn] dropping {len(small)} driver(s) with < {min_segs} segments: {small}")
        drivers = {p: s for p, s in drivers.items() if len(s) >= min_segs}
    if len(drivers) < 2: # not enough drivers
        raise ValueError(f"Need >= 2 usable drivers for meta-learning, got {len(drivers)}.")
    print(f"[data] {len(drivers)} drivers, {sum(len(s) for s in drivers.values())} segments: "
          f"{ {p: len(s) for p, s in drivers.items()} }")

    # train-val split over drivers
    if args.val_pids: # user-specified hold-out
        val_pids = [p.strip() for p in args.val_pids.split(",") if p.strip()]
        missing = [p for p in val_pids if p not in drivers]
        if missing:
            raise ValueError(f"--val-pids not found in data: {missing}; have {sorted(drivers)}")
    else: # random hold-out of ~20% of drivers (at least one, but only if >= 3 total)
        pids = sorted(drivers)
        rng.shuffle(pids)
        val_pids = pids[:max(1, round(0.2 * len(pids)))] if len(pids) >= 3 else []
    train_pids = [p for p in sorted(drivers) if p not in val_pids]
    # error handling for too few drivers after the split
    if not train_pids:
        raise ValueError("No meta-training drivers left after the validation hold-out.")
    if not val_pids:
        print("[warn] no meta-validation drivers (need >= 3 total) — no early stopping; "
              "the LAST epoch's model is saved.")
    print(f"[split] meta-train drivers={train_pids}  meta-val drivers={val_pids}")

    # collate
    collate = make_collate(context_length)
    # optimizer
    meta_opt = torch.optim.AdamW(model.parameters(), lr=args.outer_lr, weight_decay=0.0)
    # output file
    outp = pathlib.Path(args.out_pt); outp.parent.mkdir(parents=True, exist_ok=True)

    # Meta-val K values: adaptation quality at both ends of the deployment
    # support-size range (clipped per driver so a query tail always remains).
    val_ks = sorted({args.k_min, args.k_max})

    best_val = float("inf")
    bad_epochs = 0
    for ep in range(args.meta_epochs):
        # ---- outer loop over sampled episodes ----
        ep_losses = []
        # iterate through episodes (each episode has args.meta_batch drivers, each with one support/query sample)
        for start in range(0, args.episodes, args.meta_batch):
            # clip number of drivers if the last batch is smaller than meta_batch
            nb = min(args.meta_batch, args.episodes - start)
            meta_opt.zero_grad() # reset optimizer grads
            batch_loss = 0.0 # loss accumulator for the outer loop
            for _ in range(nb): # iterate through the meta-batch (the drivers)
                pid = train_pids[int(rng.integers(0, len(train_pids)))] # sample driver (with replacement)
                support, query = sample_episode( # sample episode from driver
                    drivers[pid], rng, args.k_min, args.k_max,
                    args.query_max, args.episode_start)
                if args.crop_frac > 0.0 or args.jitter_std > 0.0: # jitter augmentation (if specified in arguments)
                    support = [(augment_segment(X, rng, args.crop_frac, args.jitter_std), y)
                               for X, y in support]
                    query = [(augment_segment(X, rng, args.crop_frac, args.jitter_std), y)
                             for X, y in query]

                # Inner: adapt the head on support (no backbone graph — first-order).
                xs, ys, ls = collate(support)
                Zs = embed(model, xs, ls, device)
                w, b = adapt_head(Zs, ys.to(device), model.head.weight, model.head.bias,
                                  loss_fn, args.inner_steps, args.inner_lr, args.l2sp) # new weights and bias

                # Outer: query loss under the ADAPTED head, backbone in the graph.
                xq, yq, lq = collate(query)
                Zq = embed(model, xq, lq, device, grad=True)
                loss_q = loss_fn(F.linear(Zq, w, b), yq.to(device)) / nb # loss with the new weights and bias on query
                # Populates in_proj/backbone grads through Zq and leaves the
                # FOMAML head-init gradient on the w/b leaf tensors.
                loss_q.backward() # gradient through the backbone and the adapted head (not the original head)
                batch_loss += float(loss_q.detach()) * nb # for logging
                with torch.no_grad(): # modify head based on the gradient on the adapted head (FOMAML-style)
                    for p, leaf in ((model.head.weight, w), (model.head.bias, b)):
                        if leaf.grad is not None:
                            if p.grad is None:
                                p.grad = leaf.grad.clone()
                            else:
                                p.grad += leaf.grad
            if args.clip > 0: # gradient clipping (optional)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            meta_opt.step() # weight update
            ep_losses.append(batch_loss / nb)
        train_loss = float(np.mean(ep_losses))

        # ---- meta-validation: deployment-style adaptation on held-out drivers ----
        if val_pids:
            model.eval()
            maes, qwks = [], []
            for pid in val_pids:
                for K in val_ks:
                    Kc = min(K, len(drivers[pid]) - 1)
                    m, q = evaluate_adaptation(
                        model, drivers[pid], Kc, collate, device, loss_fn, head_type,
                        args.val_adapt_steps, args.val_adapt_lr, args.l2sp)
                    maes.append(m); qwks.append(q)
            model.train()
            val_mae, val_qwk = float(np.mean(maes)), float(np.mean(qwks))
            print(f"[epoch {ep:02d}] query_loss={train_loss:.4f} "
                  f"val_MAE={val_mae:.3f} val_QWK={val_qwk:.3f} "
                  f"(drivers={len(val_pids)}, K={val_ks})")
            if val_mae < best_val:
                best_val = val_mae
                bad_epochs = 0
                save_checkpoint(model, str(outp), arch=arch)
                print(f"[OK] saved -> {outp}")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"[stop] no val-MAE improvement in {args.patience} epochs.")
                    break
        else:
            print(f"[epoch {ep:02d}] query_loss={train_loss:.4f}")
            save_checkpoint(model, str(outp), arch=arch)

    if val_pids:
        print(f"[BEST] val_MAE={best_val:.3f} -> {outp}")
    print("[next] compare inits with the SAME per-driver protocol, e.g.:\n"
          f"  python -m scripts.sweep_train_frac --in-data data/labeled_pXXX.jsonl "
          f"--in-model {outp} --out results/sweep_pXXX_anil.png")


if __name__ == "__main__":
    main()
