# ANIL meta-training for the xLSTM LoA model (second-order by default,
# first-order as ablation via --order).
#
# ANIL (Raghu et al. 2020) = MAML with the inner loop restricted to the output
# head. The inner loop here is a few PROXIMAL SGD steps on the soft-CORN/softmax
# head *including the L2-SP anchor term* — i.e. exactly the adaptation that is
# deployed per-driver by fine_tune_XLSTM.py — so meta-training optimizes the
# initialization for the adaptation procedure actually used at test time.
# The outer loop runs in one of three modes (--order):
#   - 'second' (default — standard ANIL as published): the inner-loop
#     trajectory is differentiated exactly (create_graph, no detach). Support
#     embeddings stay in the graph, so the backbone receives BOTH the query
#     pathway and the support pathway (how the support embeddings shaped the
#     adapted head — the term that trains the representation for
#     adaptability). The double backward is confined to the tiny head + loss;
#     the xLSTM backbone itself only ever does ordinary first-order backprop,
#     so no second derivatives pass through the recurrent unroll.
#   - 'first' (FOMAML-style ablation, FO-ANIL — Yüksel et al. 2024): support
#     embeddings are computed under no_grad (dropping the support pathway) and
#     the gradient w.r.t. the head INITIALIZATION is approximated by the
#     gradient at the ADAPTED head parameters. Cheaper, immune to
#     inner-Jacobian amplification, but biased toward joint training.
#   - 'imaml' (implicit MAML, Rajeswaran et al. 2019): the inner problem is
#     SOLVED to its argmin (full-batch LBFGS) and the meta-gradient comes from
#     implicit differentiation of the stationarity condition — i.e.
#     meta-learning the anchor of the CONVERGED L2-SP adaptation, independent
#     of inner-lr/steps. Exact at this head size (dense Hessian solve, no
#     CG/GGN approximation), and captures both the anchor and the support
#     pathways. Only meaningful when λ is binding enough that the deployed
#     fine-tune also (approximately) converges its proximal problem; the
#     implicit gradient is biased whenever the inner solve stops early, so
#     the solver residual is reported per epoch.
# Head-only inner loop != meta-learning only the head: the outer loop still
# meta-trains the whole backbone for adaptability; that is what distinguishes
# the result from the joint-trained warm start it begins from.
#
# Outer-loop optimization (decision record: docs/meta_optimization_options.md;
# neither knob touches the deployed adaptation or the two-arm comparison):
#   - --outer-opt (default nadam): NAdam's Nesterov look-ahead (Dozat 2016;
#     LaANIL, Tammisetti et al. 2024) damps oscillation from noisy
#     meta-gradients — the realistic failure mode at ~15 tasks. Chosen over
#     cosine LR annealing, whose late-schedule payoff is neutralized here by
#     best-checkpoint selection + early stopping.
#   - --fo-warmup-epochs: derivative-order annealing (MAML++, Antoniou et al.
#     2019) — the first N meta-epochs run in first-order mode before switching
#     to the requested --order. FO's conservative bias makes it a stabilizing
#     pretraining stage; MAML++ reports no gradient explosions under DA where
#     second-order-only runs were unstable. This is the designated remedy if
#     the pre-registered instability criterion fires.
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
# (head_type, context_length, window_seconds, resample_hz) unchanged, so it is a drop-in
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
    encode_and_resample,
    STATE_NUM,
    STATE_CARLA,
    STATE_CAT,
    levels_to_distribution,
    soft_corn_loss,
)
from ProVoice.models.xlstm_model import _as01

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

# (frames (T, D_IN) float32, argmax LoA class 0-4, multi-hot marked levels).
# The int is what the ordinal METRICS need; the multi-hot is the training
# target, so a driver who marks several acceptable LoAs is not collapsed.
Segment = Tuple[np.ndarray, int, np.ndarray]


def build_driver_segments(
    df: pd.DataFrame,
    window_seconds: float | None,
    resample_hz: float | None = None,
) -> Dict[str, List[Segment]]:
    """Encode one (X, y, levels) triple per segment, grouped per driver, chronologically.

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
            # Both losses take the marked SET, so a multi-label segment trains
            # on what the driver gave instead of being rejected. The argmax int
            # rides along only for the ordinal metrics.
            levels = (lv > 0).astype(np.float32)
            y = int(np.argmax(lv))
            rows = [g.iloc[i].to_dict() for i in range(len(g))]
            rows = truncate_frames_by_seconds(rows, window_seconds)
            # Same fixed grid as train_XLSTM.SeqDataset — the meta-learner must
            # see the segments exactly as the population model did.
            X = encode_and_resample(rows, resample_hz, window_seconds)
            segs.append((X, y, levels))
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
    window_seconds). With resampling on, the grid is uniform, so dropping k
    leading rows is exactly k/resample_hz seconds of history — the augmentation
    became time-calibrated rather than rate-dependent. Jitter perturbs only the
    continuous feature dims.
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

    grad=False (first-order support / eval): no graph — in first-order ANIL
    the backbone gradient through the inner-loop trajectory is exactly the
    term we drop.
    grad=True (query; also support under --order second): the graph through
    the backbone is how the outer loop trains the backbone.
    """
    with torch.enable_grad() if grad else torch.no_grad():
        h = model.backbone(model.in_proj(xb.to(device).to(torch.float32)))
        idx = (lb.to(h.device).long() - 1).clamp(min=0)
        return h[torch.arange(h.size(0), device=h.device), idx]


def adapt_head(Z: torch.Tensor, target: torch.Tensor,
               w0: torch.Tensor, b0: torch.Tensor,
               loss_fn, inner_steps: int, inner_lr: float, l2sp: float,
               second_order: bool = False,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Proximal SGD on the head only: loss + λ·||θ − θ_meta||² — the deployed
    L2-SP adaptation with the meta head as anchor.

    second_order=False (FOMAML): anchor detached, every step detached; the
    returned tensors are leaves whose .grad after a query-loss backward is the
    first-order approximation of the head-initialization gradient.
    second_order=True (standard ANIL): the trajectory stays in the graph
    (create_graph, no detach), so a query-loss backward computes the exact
    Jacobian-weighted gradient w.r.t. the meta head — which is both the INIT
    and the ANCHOR here, matching deployment where the L2-SP anchor is the
    meta head itself, so gradient also flows through the anchor pathway — and
    w.r.t. Z (the support pathway into the backbone).
    """
    if second_order:
        anchor_w, anchor_b = w0, b0
        w, b = w0, b0
    else:
        anchor_w, anchor_b = w0.detach(), b0.detach()
        w = anchor_w.clone().requires_grad_(True)
        b = anchor_b.clone().requires_grad_(True)
    for _ in range(inner_steps):
        logits = F.linear(Z, w, b)
        loss = loss_fn(logits, target)
        loss = loss + l2sp * (((w - anchor_w) ** 2).sum() + ((b - anchor_b) ** 2).sum())
        gw, gb = torch.autograd.grad(loss, (w, b), create_graph=second_order)
        if second_order:
            w, b = w - inner_lr * gw, b - inner_lr * gb
        else:
            w = (w - inner_lr * gw).detach().requires_grad_(True)
            b = (b - inner_lr * gb).detach().requires_grad_(True)
    return w, b


def solve_head_proximal(Z: torch.Tensor, target: torch.Tensor,
                        w0: torch.Tensor, b0: torch.Tensor,
                        loss_fn, l2sp: float, max_iter: int, tol: float,
                        ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the proximal head problem argmin_θ loss(θ) + λ·||θ − θ_meta||²
    to (near-)optimality with full-batch LBFGS. The objective is strictly
    convex (GLM losses + the 2λI of the prox term), so the argmin is unique.

    Returns (w*, b*, residual) where residual = ||∇(inner objective)|| at the
    returned point. The implicit iMAML gradient is exact only at the argmin —
    callers should surface residual > tol instead of silently trusting it.
    """
    anchor_w, anchor_b = w0.detach(), b0.detach()
    w = anchor_w.clone().requires_grad_(True)
    b = anchor_b.clone().requires_grad_(True)

    def objective():
        return (loss_fn(F.linear(Z, w, b), target)
                + l2sp * (((w - anchor_w) ** 2).sum() + ((b - anchor_b) ** 2).sum()))

    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter,
                            tolerance_grad=0.1 * tol, tolerance_change=0.0,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    gw, gb = torch.autograd.grad(objective(), (w, b))
    residual = float((gw.pow(2).sum() + gb.pow(2).sum()).sqrt())
    return w.detach(), b.detach(), residual


def imaml_meta_step(Zs: torch.Tensor, vs: torch.Tensor,
                    Zq: torch.Tensor, vq: torch.Tensor,
                    w0: torch.Tensor, b0: torch.Tensor,
                    loss_fn, l2sp: float, scale: float,
                    max_iter: int, tol: float) -> Tuple[float, float]:
    """One iMAML episode: implicit meta-gradients from the stationarity of
    the proximal inner problem (Rajeswaran et al. 2019), computed exactly at
    this head size (dense solve, no CG).

    Let G(θ; ψ) = loss_s(θ; Zs) + λ·||θ − θ_meta||², θ* = argmin_θ G, and ψ
    any meta-input (θ_meta, or Zs and through it the backbone). From
    ∇_θ G(θ*; ψ) = 0:   dθ*/dψ = −(∇²_θ G)⁻¹ · ∂²G/∂ψ∂θ.
    With v = (∇²_θ G)⁻¹ · ∇_θ L_q(θ*), every meta-gradient is one
    vector-Jacobian product:  dL_q/dψ = −(∂/∂ψ)[∇_θ G(θ*; ψ) · v],
    which yields 2λ·v for the head (θ_meta enters only the anchor term) and
    the support pathway for the backbone (ψ = Zs) in a single backward.

    Side effects: accumulates gradients into w0/b0 (.grad) and into the
    graphs of Zs and Zq (i.e. the backbone). `scale` multiplies the query
    loss (pass 1/meta_batch). Returns (scaled query loss, inner residual).
    """
    Zs_det = Zs.detach()
    ws, bs, residual = solve_head_proximal(Zs_det, vs, w0, b0, loss_fn,
                                           l2sp, max_iter, tol)

    # Query pathway: backbone grads through Zq; g_q lands on the wl/bl leaves.
    wl = ws.clone().requires_grad_(True)
    bl = bs.clone().requires_grad_(True)
    loss_q = loss_fn(F.linear(Zq, wl, bl), vq) * scale
    loss_q.backward()
    gq = torch.cat([wl.grad.reshape(-1), bl.grad.reshape(-1)])

    # v = (∇²G at θ*)⁻¹ · g_q. The Hessian includes the prox term's +2λI, so
    # it is PD and the dense solve is exact.
    n_out, d = ws.shape
    anchor_w, anchor_b = w0.detach(), b0.detach()
    theta_star = torch.cat([ws.reshape(-1), bs.reshape(-1)])

    def flat_objective(t: torch.Tensor) -> torch.Tensor:
        wf, bf = t[:n_out * d].view(n_out, d), t[n_out * d:]
        return (loss_fn(F.linear(Zs_det, wf, bf), vs)
                + l2sp * (((wf - anchor_w) ** 2).sum() + ((bf - anchor_b) ** 2).sum()))

    H = torch.autograd.functional.hessian(flat_objective, theta_star)
    v = torch.linalg.solve(H, gq)
    vw, vb = v[:n_out * d].view(n_out, d), v[n_out * d:]

    # One VJP delivers the remaining meta-gradients: −(∂/∂ψ)[∇_θG · v] puts
    # the anchor pathway into the LIVE w0/b0 (= 2λ·v·scale) and the support
    # pathway into the backbone through Zs's graph.
    wr = ws.clone().requires_grad_(True)
    br = bs.clone().requires_grad_(True)
    inner = (loss_fn(F.linear(Zs, wr, br), vs)
             + l2sp * (((wr - w0) ** 2).sum() + ((br - b0) ** 2).sum()))
    rw, rb = torch.autograd.grad(inner, (wr, br), create_graph=True)
    corr = -((rw * vw).sum() + (rb * vb).sum())
    corr.backward()
    return float(loss_q.detach()), residual


def adapt_head_deployed(Z: torch.Tensor, target: torch.Tensor,
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
        loss = loss_fn(F.linear(Z, w, b), target)
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
    xs, _ys, ls, vs = collate(segs[:K])
    Zs = embed(model, xs, ls, device)
    w, b = adapt_head_deployed(Zs, vs.to(device), model.head.weight, model.head.bias,
                               loss_fn, steps, lr, l2sp)
    xq, yq, lq, _vq = collate(segs[K:])
    Zq = embed(model, xq, lq, device)
    with torch.no_grad():
        pred = logits_to_probs(F.linear(Zq, w, b), head_type).argmax(dim=-1)
    Yt, Yp = yq.numpy(), pred.cpu().numpy()
    return mae(Yt, Yp), qwk(Yt, Yp, 5)


def main():
    ap = argparse.ArgumentParser(
        description="ANIL meta-training of the xLSTM LoA model "
                    "(head-only proximal inner loop, full-model outer loop; "
                    "--order selects exact second-order or FOMAML-style first-order).")
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
    ap.add_argument("--order", choices=["second", "first", "imaml"], default="second",
                    help="'second' (standard ANIL): differentiate exactly through the inner "
                         "loop; the backbone also receives the support-pathway gradient. "
                         "'first' (FO-ANIL ablation): gradients taken at the adapted head, "
                         "support embedded under no_grad — cheaper, biased toward joint training. "
                         "'imaml': solve the inner problem to its argmin (LBFGS) and use exact "
                         "implicit gradients — meta-learns the anchor of the CONVERGED L2-SP "
                         "adaptation; inner-steps/inner-lr are ignored, and λ should be binding.")
    ap.add_argument("--inner-steps", type=int, default=5,
                    help="Head inner-loop SGD steps (orders first/second; ignored for imaml).")
    ap.add_argument("--inner-lr",    type=float, default=0.1,
                    help="Plain-SGD step size for the head inner loop (the head is a ~260-param "
                         "GLM; SGD needs a larger lr than fine_tune_XLSTM's AdamW). Orders "
                         "first/second only; ignored for imaml.")
    ap.add_argument("--l2sp",  type=float, default=0.01,
                    help="λ of the proximal/L2-SP term inside the inner loop, anchoring the "
                         "adapted head to the meta head. Keep equal to the λ used by "
                         "fine_tune_XLSTM.py at deployment so meta-training matches it.")
    ap.add_argument("--fo-warmup-epochs", type=int, default=0,
                    help="Derivative-order annealing (MAML++): run the first N meta-epochs "
                         "in first-order mode, then switch to the requested --order. 0 = off. "
                         "Stabilizer for --order second/imaml (no-op for --order first); "
                         "keep --patience comfortably larger than the epochs remaining after "
                         "the switch, since val-MAE may transiently worsen at the transition.")
    ap.add_argument("--imaml-max-iter", type=int, default=200,
                    help="LBFGS iteration cap for the iMAML inner solve.")
    ap.add_argument("--imaml-tol", type=float, default=1e-4,
                    help="Gradient-norm tolerance for the iMAML inner solve. The implicit "
                         "meta-gradient is exact only at the argmin, so episodes ending above "
                         "this are counted and reported per epoch. Default matches float32's "
                         "realistic LBFGS floor (~1e-4); the induced meta-gradient bias is "
                         "~residual/(2λ), negligible when λ is binding.")
    # --- outer loop ---
    ap.add_argument("--meta-epochs",  type=int, default=60)
    ap.add_argument("--episodes",     type=int, default=200,
                    help="Episodes sampled per meta-epoch (spread over the train drivers).")
    ap.add_argument("--meta-batch",   type=int, default=4,
                    help="Episodes averaged into one outer update.")
    ap.add_argument("--outer-lr",     type=float, default=1e-4,
                    help="Meta-optimizer lr for backbone+in_proj+head init. Keep small: this is "
                         "a warm-started refinement toward adaptability, not training from scratch.")
    ap.add_argument("--outer-opt", choices=["nadam", "adamw"], default="nadam",
                    help="Meta-optimizer. 'nadam' (default): Nesterov look-ahead momentum — "
                         "damps meta-gradient oscillation from few tasks (LaANIL 2024, "
                         "Dozat 2016). 'adamw' reproduces the pre-2026-07-13 behavior. "
                         "Both run with weight_decay=0 (the L2-SP anchor is the regularizer).")
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
    if args.fo_warmup_epochs < 0:
        raise ValueError(f"--fo-warmup-epochs must be >= 0, got {args.fo_warmup_epochs}")
    if args.fo_warmup_epochs > 0 and args.order == "first":
        print("[warn] --fo-warmup-epochs has no effect with --order first (already first-order).")
    if args.fo_warmup_epochs >= args.meta_epochs and args.order != "first":
        print(f"[warn] --fo-warmup-epochs ({args.fo_warmup_epochs}) >= --meta-epochs "
              f"({args.meta_epochs}): the requested order '{args.order}' will never run.")

    # seed and cuda
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The effective order is per-epoch (derivative-order annealing); see the
    # top of the meta-epoch loop.

    # checkpoint properties: context_length, head_type, window_seconds, resample_hz
    model, arch = load_checkpoint(args.init_pt)
    model.to(device).train()
    context_length = arch["context_length"]
    head_type = arch.get("head_type", "softmax")
    window_seconds = arch.get("window_seconds")
    resample_hz = arch.get("resample_hz")
    print(f"[model] warm start from {args.init_pt}: head_type={head_type} "
          f"context_length={context_length} window_seconds={window_seconds} "
          f"resample_hz={resample_hz} order={args.order}"
          + (f" (first-order warm-up for {args.fo_warmup_epochs} epoch(s))"
             if args.fo_warmup_epochs > 0 and args.order != "first" else ""))
    # soft-CORN or softmax loss. Both take the multi-hot marked levels, so the
    # inner loop, the iMAML solve and meta-validation all share one target type.
    if head_type == "corn":
        loss_fn = lambda logits, lvl: soft_corn_loss(logits, lvl)
    else:
        _ce = nn.CrossEntropyLoss()
        loss_fn = lambda logits, lvl: _ce(logits, levels_to_distribution(lvl))

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
    drivers = build_driver_segments(df, window_seconds, resample_hz)
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
    # meta-optimizer (weight_decay=0 in both: the L2-SP anchor is the regularizer)
    opt_cls = torch.optim.NAdam if args.outer_opt == "nadam" else torch.optim.AdamW
    meta_opt = opt_cls(model.parameters(), lr=args.outer_lr, weight_decay=0.0)
    print(f"[opt] outer optimizer: {args.outer_opt} (lr={args.outer_lr})")
    # output file
    outp = pathlib.Path(args.out_pt); outp.parent.mkdir(parents=True, exist_ok=True)

    # Meta-val K values: adaptation quality at both ends of the deployment
    # support-size range (clipped per driver so a query tail always remains).
    val_ks = sorted({args.k_min, args.k_max})

    best_val = float("inf")
    bad_epochs = 0
    for ep in range(args.meta_epochs):
        # Derivative-order annealing (MAML++)
        warm = ep < args.fo_warmup_epochs
        second = args.order == "second" and not warm
        imaml = args.order == "imaml" and not warm
        if args.order != "first" and args.fo_warmup_epochs > 0 and ep == args.fo_warmup_epochs:
            print(f"[order] first-order warm-up done -> switching to '{args.order}'")

        # ---- outer loop over sampled episodes ----
        ep_losses = []
        ep_res = []  # iMAML inner-solve residuals (empty for other orders)
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
                    support = [(augment_segment(X, rng, args.crop_frac, args.jitter_std), y, v)
                               for X, y, v in support]
                    query = [(augment_segment(X, rng, args.crop_frac, args.jitter_std), y, v)
                             for X, y, v in query]

                # Embed support/query. Support keeps its graph whenever the
                # support pathway into the backbone is used (second/imaml);
                # first order drops it. Query always keeps its graph.
                xs, _ys, ls, vs = collate(support)
                Zs = embed(model, xs, ls, device, grad=second or imaml)
                xq, _yq, lq, vq = collate(query)
                Zq = embed(model, xq, lq, device, grad=True)

                if imaml:
                    # iMAML: solve the proximal head problem to its argmin and
                    # accumulate the exact implicit meta-gradients (anchor,
                    # support and query pathways) — see imaml_meta_step.
                    lq_val, res = imaml_meta_step(
                        Zs, vs.to(device), Zq, vq.to(device),
                        model.head.weight, model.head.bias,
                        loss_fn, args.l2sp, 1.0 / nb,
                        args.imaml_max_iter, args.imaml_tol)
                    batch_loss += lq_val * nb # for logging
                    ep_res.append(res)
                else:
                    # Inner: adapt the head on support with truncated proximal SGD.
                    w, b = adapt_head(Zs, vs.to(device), model.head.weight, model.head.bias,
                                      loss_fn, args.inner_steps, args.inner_lr, args.l2sp,
                                      second_order=second) # new weights and bias

                    # Outer: query loss under the ADAPTED head, backbone in the graph.
                    loss_q = loss_fn(F.linear(Zq, w, b), vq.to(device)) / nb # loss with the new weights and bias on query
                    # Second order: this single backward computes everything exactly —
                    # head-init grads through the inner trajectory, backbone grads
                    # through BOTH the support and query pathways.
                    # First order: populates in_proj/backbone grads through Zq only and
                    # leaves the FOMAML head-init gradient on the w/b leaf tensors.
                    loss_q.backward() # gradient through the backbone and the adapted head (not the original head)
                    batch_loss += float(loss_q.detach()) * nb # for logging
                    if not second:
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
        # iMAML only: how trustworthy were the implicit gradients this epoch?
        res_note = (f" inner_res_max={max(ep_res):.1e}"
                    f" unconverged={sum(r > args.imaml_tol for r in ep_res)}/{len(ep_res)}"
                    if ep_res else "")

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
                  f"(drivers={len(val_pids)}, K={val_ks}){res_note}")
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
            print(f"[epoch {ep:02d}] query_loss={train_loss:.4f}{res_note}")
            save_checkpoint(model, str(outp), arch=arch)

    if val_pids:
        print(f"[BEST] val_MAE={best_val:.3f} -> {outp}")
    print("[next] compare inits with the SAME per-driver protocol, e.g.:\n"
          f"  python -m scripts.sweep_train_frac --in-data data/labeled_pXXX.jsonl "
          f"--in-model {outp} --out results/sweep_pXXX_anil.png")


if __name__ == "__main__":
    main()
