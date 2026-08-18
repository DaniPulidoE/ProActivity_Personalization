# ANIL meta-training for the xLSTM LoA model (iMAML by default since 2026-08-14;
# second- and first-order are ablations via --order).
#
# ANIL (Raghu et al. 2020) = MAML with the inner loop restricted to the output
# head, including the L2-SP anchor term — i.e. the same objective the deployed
# per-driver adaptation minimizes.
#
# WHY iMAML IS THE DEFAULT rather than an upgrade to adopt later. The deployed
# adaptation (head_adapt.adapt_head) runs 2000 full-batch AdamW steps to the MAP.
# A path-differentiated outer loop must backprop THROUGH the inner trajectory, so
# its inner loop has to be a handful of graph-resident SGD steps — it structurally
# cannot be the deployed procedure, only an approximation of it. Implicit
# differentiation needs the inner problem's SOLUTION and never its path, so iMAML
# meta-learns the anchor of exactly the objective that gets served. Verified
# numerically on participant 001 at tau=2: max|theta_newton - theta_adam| is
# 1.6e-05 at K=5 and <= 3.4e-06 for K in {10, 30, 99}.
#
# The reverse fix — serving a truncated few-step head so deployment matches
# path-differentiated meta-training — is ruled out by the UQ layer: the Laplace
# expansion is taken about the MAP, so a truncated head would make the posterior
# valid for the L2-SP arm and invalid for ANIL, which the design forbids.
#
# The outer loop runs in one of three modes (--order):
#   - 'second' (ABLATION — standard ANIL as published): the inner-loop
#     trajectory is differentiated exactly (create_graph, no detach). Support
#     embeddings stay in the graph, so the backbone receives BOTH the query
#     pathway and the support pathway (how the support embeddings shaped the
#     adapted head — the term that trains the representation for
#     adaptability). The double backward is confined to the tiny head + loss;
#     the xLSTM backbone itself only ever does ordinary first-order backprop,
#     so no second derivatives pass through the recurrent unroll.
#   - 'first' (ABLATION, FOMAML-style FO-ANIL — Yuksel et al. 2024): support
#     embeddings are computed under no_grad (dropping the support pathway) and
#     the gradient w.r.t. the head INITIALIZATION is approximated by the
#     gradient at the ADAPTED head parameters. Cheaper, immune to
#     inner-Jacobian amplification, but biased toward joint training.
#   - 'imaml' (DEFAULT — implicit MAML, Rajeswaran et al. 2019): the inner problem is
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
import argparse, csv, pathlib
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
    logits_to_label,
    encode_and_resample,
    STATE_NUM,
    STATE_CARLA,
    STATE_CAT,
    SENTINEL_VALUES,
    levels_to_distribution,
    soft_corn_loss,
)
from ProVoice.models.xlstm_model import _as01
from ProVoice.models.laplace_head import corn_curvature_blocks
from ProVoice.models.head_adapt import augment_z, install_fcd_head
from ProVoice.models.head_adapt import (
    adapt_head_tensors,
    l2sp_from_tau,
    DEFAULT_ADAPT_LR,
    DEFAULT_ADAPT_STEPS,
    DEFAULT_TAU,
)

from ProVoice.models.train_XLSTM import (
    set_seed,
    read_jsonl,
    iter_jsonl,
    normalize_row,
    make_collate,
    set_accuracy,
    set_mae,
    set_qwk,
)

LEVELS = [f"Level_{i}" for i in range(1, 6)]

# Feature dims eligible for jitter augmentation: the driver-state and CARLA
# numericals. FCD dims are static per task (jitter would fake nonexistent
# tasks) and the one-hot / length-encoded categoricals are not continuous.
_JITTER_SLICE = slice(len(FCD_NAMES), len(FCD_NAMES) + len(STATE_NUM) + len(STATE_CARLA))

# (frames (T, D_IN) float32, multi-hot marked levels). The multi-hot is the
# whole label: it is the training target AND the metric target, so a driver who
# marks several acceptable LoAs is never collapsed to one of them.
Segment = Tuple[np.ndarray, np.ndarray]


def build_driver_segments(
    df: pd.DataFrame,
    window_seconds: float | None,
    resample_hz: float | None = None,
) -> Dict[str, List[Segment]]:
    """Encode one (X, levels) pair per segment, grouped per driver, chronologically.

    Chronology = first appearance in the JSONL (groupby(sort=False)), NOT the
    lexicographic order of segment_id strings — episode supports must be
    temporal prefixes/blocks. Segments with missing/all-zero Level_* labels
    are skipped and reported, never collapsed into a bogus class-0 label.
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
            # on what the driver gave instead of being rejected — and the
            # meta-validation metrics take the same set, so early stopping is
            # not steered by a collapsed stand-in for it.
            levels = (lv > 0).astype(np.float32)
            rows = g.to_dict("records")
            rows = truncate_frames_by_seconds(rows, window_seconds)
            # Same fixed grid as train_XLSTM.SeqDataset — the meta-learner must
            # see the segments exactly as the population model did.
            X = encode_and_resample(rows, resample_hz, window_seconds)
            segs.append((X, levels))
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
        x = xb.to(device).to(torch.float32)
        h = model.backbone(model.in_proj(x))
        idx = (lb.to(h.device).long() - 1).clamp(min=0)
        z = h[torch.arange(h.size(0), device=h.device), idx]
        # Inferred from the head rather than passed in: `embed` has ~8 call
        # sites across the inner loop, the implicit solve and meta-validation,
        # and a boolean threaded through all of them is a place for the two arms
        # to silently diverge. The head is widened once at startup
        # (`install_fcd_head`) and every embedding follows it.
        return augment_z(z, x, model.head_uses_fcd())


def inner_adapt(Z: torch.Tensor, target: torch.Tensor,
                w0: torch.Tensor, b0: torch.Tensor,
                loss_fn, inner_steps: int, inner_lr: float, tau: float,
                second_order: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """META-TRAINING's inner loop: differentiable proximal SGD on the head only,
    loss + λ·||θ − θ_meta||², with the meta head as anchor.

    NOT ``head_adapt.adapt_head`` — and it cannot be. The outer loop
    differentiates THROUGH this trajectory (``create_graph=second_order``), so
    every step has to stay in the autograd graph; ``adapt_head`` runs
    ``torch.optim.AdamW`` for 2000 steps, which both severs the graph and would
    be intractable to backprop through. This is the one place where the
    meta-training and deployed adaptation procedures genuinely differ, and it is
    why ``--order imaml`` is the variant the design prefers: implicit
    differentiation needs only the inner problem's SOLUTION, never its path, so
    it can solve the identical objective that gets deployed. See the module
    docstring of ``head_adapt``.

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
    # tau -> lambda HERE, from this episode's own support size. Episodes
    # deliberately vary K, and the objective is a batch MEAN, so a fixed lambda
    # would realise tau = 2*K*lambda and the anchor strength would change from
    # episode to episode purely as a side effect of the K sampling — variance
    # injected into the meta-gradient that has nothing to do with the task.
    l2sp = l2sp_from_tau(tau, Z.shape[0])
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


def _solve_head_newton(Z: torch.Tensor, target: torch.Tensor,
                       w0: torch.Tensor, b0: torch.Tensor,
                       loss_fn, l2sp: float, max_iter: int, tol: float,
                       ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Damped Newton on the CORN proximal problem — the fast path.

    The inner objective is ~260 parameters over at most 10 support points and
    is STRICTLY convex (each CORN unit is a logistic regression, plus the prox
    term's 2λI), so Newton converges quadratically and the exact Hessian is
    already available block-diagonally from ``corn_curvature_blocks``. LBFGS
    with a 200-iteration cap and strong-Wolfe line search was solving this to
    ~1e-5 in 17.9 ms — the dominant cost of an iMAML episode once the dense
    autograd Hessian was removed.

    Gradient and objective come from ``loss_fn`` via autograd rather than a
    second hand-derived formula: the Hessian is the expensive part and is
    already shared and verified, while a duplicated gradient could silently
    drift from ``soft_corn_loss`` if that ever changed.

    Backtracking (Armijo) keeps it honest: the Newton direction is a descent
    direction because H is PD, so the line search terminates, and every step
    decreases the objective. Without damping, a full Newton step can overshoot
    on a logistic loss when the anchor starts far from the optimum.

    A unit no label reaches gets a zero curvature block, so its Hessian is the
    pure 2λI prior and the first Newton step lands it exactly on the anchor —
    which is the correct minimiser for a unit with no data.
    """
    anchor_w, anchor_b = w0.detach(), b0.detach()
    w, b = anchor_w.clone(), anchor_b.clone()
    n_out, d = w.shape
    n = Z.shape[0]
    eye = torch.eye(d + 1, dtype=torch.float64, device=Z.device)

    def _obj(w_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
        return (loss_fn(F.linear(Z, w_, b_), target)
                + l2sp * (((w_ - anchor_w) ** 2).sum() + ((b_ - anchor_b) ** 2).sum()))

    def _val_and_grad(w_, b_):
        wg = w_.detach().requires_grad_(True)
        bg = b_.detach().requires_grad_(True)
        val = _obj(wg, bg)
        gw, gb = torch.autograd.grad(val, (wg, bg))
        return float(val.detach()), gw, gb

    for _ in range(max_iter):
        val, gw, gb = _val_and_grad(w, b)
        if float((gw.pow(2).sum() + gb.pow(2).sum()).sqrt()) <= tol:
            break
        H = corn_curvature_blocks(Z, target, w, b) / n + 2.0 * l2sp * eye
        g_blk = torch.cat([gw, gb.unsqueeze(1)], dim=1).to(torch.float64)
        L = torch.linalg.cholesky(H)                       # PD: 2*l2sp > 0
        step = torch.cholesky_solve(g_blk.unsqueeze(-1), L).squeeze(-1)
        dw = -step[:, :d].to(w.dtype)
        db = -step[:, d].to(b.dtype)
        slope = float((gw * dw).sum() + (gb * db).sum())   # < 0, H is PD
        t, ok = 1.0, False
        with torch.no_grad():
            for _ in range(40):
                if float(_obj(w + t * dw, b + t * db)) <= val + 1e-4 * t * slope:
                    ok = True
                    break
                t *= 0.5
        if not ok:            # already at the optimum to float precision
            break
        w = w + t * dw
        b = b + t * db

    _, gw, gb = _val_and_grad(w, b)
    residual = float((gw.pow(2).sum() + gb.pow(2).sum()).sqrt())
    return w.detach(), b.detach(), residual


def solve_head_proximal(Z: torch.Tensor, target: torch.Tensor,
                        w0: torch.Tensor, b0: torch.Tensor,
                        loss_fn, l2sp: float, max_iter: int, tol: float,
                        head_type: str = "corn",
                        ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the proximal head problem argmin_θ loss(θ) + λ·||θ − θ_meta||²
    to (near-)optimality. The objective is strictly convex (GLM losses + the
    2λI of the prox term), so the argmin is unique.

    CORN heads use damped Newton with the exact block-diagonal Hessian
    (:func:`_solve_head_newton`). Softmax heads fall back to LBFGS: their units
    couple through the normaliser, so the block-diagonal Hessian does not apply.

    Returns (w*, b*, residual) where residual = ||∇(inner objective)|| at the
    returned point. The implicit iMAML gradient is exact only at the argmin —
    callers should surface residual > tol instead of silently trusting it.
    """
    if head_type == "corn":
        return _solve_head_newton(Z, target, w0, b0, loss_fn, l2sp, max_iter, tol)

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
                    loss_fn, tau: float, scale: float,
                    max_iter: int, tol: float,
                    head_type: str = "corn") -> Tuple[float, float]:
    """One iMAML episode: implicit meta-gradients from the stationarity of
    the proximal inner problem (Rajeswaran et al. 2019), computed exactly at
    this head size (dense solve, no CG).

    THIS is the variant whose inner problem is the deployed adaptation. The
    objective solved below — batch-mean loss + λ||θ − θ_meta||² with
    λ = τ/(2K) — is character-for-character the one ``head_adapt.adapt_head``
    minimizes at serving time. The OPTIMIZER differs (damped Newton here,
    AdamW there) and that is fine, indeed preferable: the implicit function
    theorem is a statement about the stationary point, not the path to it, and
    Newton converges quadratically on a strictly convex 260-parameter problem.
    What has to match is the objective and the fact that both actually reach
    θ*; ``solve_head_proximal`` returns a residual and ``adapt_head`` returns
    ``grad_norm`` so both are checkable rather than assumed.

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
    # Per-episode lambda from the shared tau (see inner_adapt for why this is
    # not a fixed flag). Everything below — the solve, the Hessian's prox term,
    # and the VJP's anchor term — must use this SAME number.
    l2sp = l2sp_from_tau(tau, Zs.shape[0])
    Zs_det = Zs.detach()
    ws, bs, residual = solve_head_proximal(Zs_det, vs, w0, b0, loss_fn,
                                           l2sp, max_iter, tol, head_type=head_type)

    # Query pathway: backbone grads through Zq; g_q lands on the wl/bl leaves.
    wl = ws.clone().requires_grad_(True)
    bl = bs.clone().requires_grad_(True)
    loss_q = loss_fn(F.linear(Zq, wl, bl), vq) * scale
    loss_q.backward()
    gq = torch.cat([wl.grad.reshape(-1), bl.grad.reshape(-1)])

    # v = (∇²G at θ*)⁻¹ · g_q. The Hessian includes the prox term's +2λI, so
    # it is PD and the solve is exact.
    n_out, d = ws.shape
    anchor_w, anchor_b = w0.detach(), b0.detach()

    if head_type == "corn":
        # Closed-form block-diagonal Hessian instead of
        # torch.autograd.functional.hessian, which costs ONE double-backward per
        # parameter — 260 of them per episode at the deployed head size, ~180x
        # the cost of this path for a bit-identical result (verified to 1e-16 in
        # float64 against autograd, with the autograd Hessian's off-block
        # entries exactly zero).
        #
        # NOTE the normaliser. corn_curvature_blocks is un-normalised, while
        # this inner objective is the BATCH-MEAN loss + λ||·||², so the data
        # term carries 1/B and the prox term contributes exactly 2λI. Reusing
        # LaplacePosterior's τ = 2Nλ here instead would rescale v by B — and B
        # is the episode's support size, drawn per episode, so it would inject
        # episode-dependent noise into the meta-gradient without raising.
        blocks = corn_curvature_blocks(Zs_det, vs, ws, bs)          # (K-1, E+1, E+1)
        eye = torch.eye(d + 1, dtype=blocks.dtype, device=blocks.device)
        H = blocks / Zs_det.shape[0] + 2.0 * l2sp * eye
        # Solve per unit. The flat layout here is [all weights; all biases], so
        # unit j's parameters are NOT contiguous in it — assembling a flat
        # matrix and slicing it is where an index bug would hide. Going
        # straight to (K-1, E+1) blocks sidesteps that entirely, and it is the
        # shape the VJP below wants anyway.
        gq_blk = torch.cat([gq[:n_out * d].view(n_out, d),
                            gq[n_out * d:].unsqueeze(1)], dim=1).to(blocks.dtype)
        L = torch.linalg.cholesky(H)                                # PD: 2λ > 0
        v_blk = torch.cholesky_solve(gq_blk.unsqueeze(-1), L).squeeze(-1)
        vw = v_blk[:, :d].to(ws.dtype)
        vb = v_blk[:, d].to(bs.dtype)
    else:
        # Softmax heads couple their units through the normaliser, so the
        # Hessian is not block-diagonal and the closed form does not apply.
        # Fall back to the dense autograd Hessian.
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


# `adapt_head_deployed` used to live here: a THIRD private copy of the deployed
# adaptation, whose docstring claimed to be "identical to fine_tune_XLSTM.py"
# because "the mini-batch loop degenerates to one full-batch step per epoch
# whenever the support fits in a batch (K <= 16 always does)". That held only for
# the episode sizes here, never for the fine-tuner's real K (~99 on a full
# driver), and stopped holding at all once the fine-tuner moved to a
# K-independent full-batch budget with lambda derived from tau. Meta-validation
# now calls head_adapt.adapt_head_tensors — the same function that serves — so
# the claim is true by construction rather than by comment.


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
                        steps: int, lr: float, tau: float,
                        ) -> Tuple[float, float]:
    """Meta-validation on ONE held-out driver with the DEPLOYED adaptation:
    the driver's first K segments (true session prefix) through the SAME
    ``head_adapt`` call that serves and that draws the learning curve, then
    set-MAE/set-QWK on everything after — the same temporal protocol and the
    same set-aware metrics as the sweep script.

    This is the measurement that decides meta-training early stopping, so it is
    the one place where using anything other than the deployed adaptation would
    select a meta-init for a procedure that never runs.

    ``loss_fn`` is unused now that the shared adapter derives it from
    ``head_type``; it is kept in the signature so callers stay unchanged and the
    two cannot silently disagree about which loss adaptation used.
    """
    xs, ls, vs = collate(segs[:K])
    Zs = embed(model, xs, ls, device)
    w, b, _ = adapt_head_tensors(Zs, vs.to(device), model.head.weight, model.head.bias,
                                 tau=tau, head_type=head_type, steps=steps, lr=lr)
    xq, lq, vq = collate(segs[K:])
    Zq = embed(model, xq, lq, device)
    with torch.no_grad():
        pred = logits_to_label(F.linear(Zq, w, b), head_type)
    Yl, Yp = vq.numpy(), pred.cpu().numpy()
    return set_mae(Yl, Yp), set_qwk(Yl, Yp, 5)


def evaluate_adaptation_multi_k(model, segs: List[Segment], ks: List[int],
                                collate, device: str, loss_fn, head_type: str,
                                steps: int, lr: float, tau: float,
                                ) -> Dict[int, Tuple[float, float, float, int]]:
    """:func:`evaluate_adaptation` at SEVERAL support sizes, embedding once.

    Returns ``{requested_K: (set_mae, set_acc, set_qwk, effective_K)}``.

    The driver's whole segment list goes through the backbone ONE time and the
    result is sliced per K. Calling :func:`evaluate_adaptation` in a loop instead
    repeats the only expensive part -- a backbone forward over the driver's ~120
    segments -- once per K, which is what made a wide meta-validation grid look
    unaffordable.

    What that buys, measured on 120 synthetic segments: a 2-point grid drops
    6.8s -> 4.9s per driver per epoch, and a 5-point grid costs 10.8s. So the
    term that scaled with the grid is gone, but the ADAPTATION does not vanish
    -- 2000 AdamW steps is ~2s per (driver, K) and is now the dominant cost.
    Widening {5,10} -> {5,10,20,30,60} is ~1.6x the old validation cost, not the
    2.5x a naive per-K reading would predict; against ~54s/epoch of meta-training
    that is roughly +12% per epoch.

    HOW EXACT THIS IS, measured rather than asserted. ``make_collate`` right-pads
    every segment to the fixed ``context_length``, so no segment's embedding
    depends on the CONTENT of its batch -- but it does depend slightly on the
    batch's SIZE, because that selects a different matmul kernel and so a
    different float32 reduction order. Measured: the support block comes back
    bit-identical, the query block differs by ~1e-6 per embedding coordinate.
    That does not survive to the metric -- the embeddings are consumed by 2000
    adaptation steps and then an argmax over 5 ordinal classes. Checked against
    :func:`evaluate_adaptation` over 3 seeds x K in {5,10,20,30,60}: all 15 cells
    returned identical set-MAE and set-QWK to full printed precision. So this is
    a faster route to the same numbers, NOT a bit-reproducible one; a run that
    needs bit-level reproduction against the old path will not get it.

    Effective K is returned because it is CLIPPED to leave at least one query
    segment; a caller averaging over the grid needs to know when two requested
    K values collapsed onto the same evaluation and are being double-counted.
    """
    n = len(segs)
    x, l, v = collate(segs)
    Z = embed(model, x, l, device)          # grad=False by default: no graph
    V = v.to(device)
    out: Dict[int, Tuple[float, float, float, int]] = {}
    for K in ks:
        Kc = max(1, min(int(K), n - 1))
        w, b, _ = adapt_head_tensors(Z[:Kc], V[:Kc], model.head.weight, model.head.bias,
                                     tau=tau, head_type=head_type, steps=steps, lr=lr)
        with torch.no_grad():
            pred = logits_to_label(F.linear(Z[Kc:], w, b), head_type)
        Yl, Yp = v[Kc:].numpy(), pred.cpu().numpy()
        out[int(K)] = (set_mae(Yl, Yp), set_accuracy(Yl, Yp), set_qwk(Yl, Yp, 5), Kc)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="ANIL meta-training of the xLSTM LoA model (head-only proximal "
                    "inner loop, full-model outer loop). --order defaults to 'imaml', "
                    "the only variant whose inner problem IS the deployed adaptation; "
                    "'second'/'first' are ablations.")
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
    ap.add_argument("--embed-fcd", dest="embed_fcd", action="store_true",
                    help="Give the adapted head direct access to the task: it sees "
                         "[z_64 | FCD_12] instead of z_64 alone. The backbone is untouched, "
                         "so no retraining is implied; the appended block is initialized AND "
                         "L2-SP-anchored at zero, so K=0 reproduces the population head "
                         "exactly (checked at startup). MUST match the other arm: both arms "
                         "have to adapt the identical object or the comparison is confounded.")
    ap.add_argument("--order", choices=["imaml", "second", "first"], default="imaml",
                    help="DEFAULT 'imaml': solve the inner problem to its argmin (damped "
                         "Newton on a CORN head, LBFGS on a softmax one) and use exact "
                         "implicit gradients. This is the primary variant because it is the "
                         "only one whose inner problem IS the deployed adaptation: implicit "
                         "differentiation needs the inner problem's SOLUTION, never its path, "
                         "so it meta-learns the anchor of the same converged L2-SP objective "
                         "head_adapt.adapt_head minimizes at serving. --inner-steps/--inner-lr "
                         "are ignored; --tau must be > 0 or the inner argmin is neither unique "
                         "nor solvable.\n"
                         "'second' (standard ANIL) and 'first' (FO-ANIL) are ABLATIONS. They "
                         "differentiate through the inner TRAJECTORY, so it must be a handful "
                         "of graph-resident SGD steps rather than 2000 AdamW steps to "
                         "convergence — meaning they structurally cannot run the deployed "
                         "adaptation as their inner loop, and the reverse fix (serving a "
                         "truncated head) is blocked by the Laplace layer, which expands about "
                         "the MAP. 'first' is additionally biased toward joint training.")
    ap.add_argument("--inner-steps", type=int, default=5,
                    help="Head inner-loop SGD steps (orders first/second; ignored for imaml).")
    ap.add_argument("--inner-lr",    type=float, default=0.1,
                    help="Plain-SGD step size for the head inner loop (the head is a ~260-param "
                         "GLM; SGD needs a larger lr than fine_tune_XLSTM's AdamW). Orders "
                         "first/second only; ignored for imaml.")
    ap.add_argument("--tau",  type=float, default=DEFAULT_TAU,
                    help="PRIOR PRECISION of the proximal/L2-SP term inside the inner loop, "
                         "anchoring the adapted head to the meta head. The per-episode λ is "
                         "derived as τ/(2K) from that episode's OWN support size. This is not "
                         "cosmetic here: episodes deliberately vary K, the inner objective is "
                         "a batch mean, so a fixed λ realises τ = 2Kλ and the anchor strength "
                         "would swing from episode to episode as a side effect of the K "
                         "sampling — variance injected straight into the meta-gradient. Keep "
                         "equal to the τ used by fine_tune_XLSTM.py at deployment, or "
                         "meta-training optimizes an initialization for an adaptation that "
                         "never runs.")
    ap.add_argument("--fo-warmup-epochs", type=int, default=0,
                    help="Derivative-order annealing (MAML++): run the first N meta-epochs "
                         "in first-order mode, then switch to the requested --order. 0 = off. "
                         "Stabilizer for --order second/imaml (no-op for --order first); "
                         "keep --patience comfortably larger than the epochs remaining after "
                         "the switch, since val-MAE may transiently worsen at the transition.")
    ap.add_argument("--imaml-max-iter", type=int, default=200,
                    help="Iteration cap for the iMAML inner solve (damped Newton on a "
                         "CORN head, LBFGS on a softmax one). Newton converges "
                         "quadratically on this strictly convex ~260-param problem and "
                         "typically needs <20, so the default is a safety cap, not a "
                         "budget; the per-epoch residual report is the thing to watch.")
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
    ap.add_argument("--val-ks", dest="val_ks", default="5,10,20,30,60",
                    help="Support sizes at which meta-validation adapts the held-out "
                         "drivers. THIS DEFINES WHAT val_set_mae MEANS, and therefore what "
                         "M* and the configuration ranking are selected FOR. The old grid "
                         "was {--k-min, --k-max} = {5,10}, which tuned the arm purely for the "
                         "first three minutes of labelling while the L2-SP learning curve "
                         "shows personalization does its real work above K=30. Decoupled from "
                         "--k-min/--k-max on purpose: those set the TRAINING episode regime, "
                         "this sets what the model is SELECTED for, and there is no reason the "
                         "two must coincide. The reported val_set_mae/val_set_acc are FLAT "
                         "MEANS over drivers x this grid; per-K columns go to --metrics-csv so "
                         "any other weighting is recoverable without re-running. Costs almost "
                         "nothing to widen (see evaluate_adaptation_multi_k). Values are "
                         "clipped per driver to leave a query tail. RECORD IT -- runs with "
                         "different grids produce val_set_mae numbers that are not comparable "
                         "and must not be pooled into one ranking.")
    ap.add_argument("--val-adapt-steps", type=int, default=DEFAULT_ADAPT_STEPS,
                    help="Full-batch steps for meta-validation adaptation, which runs through "
                         "head_adapt.adapt_head_tensors — the same function that serves. "
                         "Defaults to head_adapt.DEFAULT_ADAPT_STEPS so it matches without "
                         "anyone having to keep two numbers in sync. The old default (30, "
                         "justified by 'one fine-tune epoch = one full-batch step when K <= "
                         "batch size') was both stale and far short of the MAP: 300 steps at "
                         "5e-4 still left the objective ~20 % above its optimum.")
    ap.add_argument("--val-adapt-lr", type=float, default=DEFAULT_ADAPT_LR,
                    help="lr for meta-validation adaptation; defaults to "
                         "head_adapt.DEFAULT_ADAPT_LR for the same reason.")
    ap.add_argument("--metrics-csv", dest="metrics_csv", default="",
                    help="Write one row per meta-epoch: the meta-TRAINING query loss and the "
                         "meta-VALIDATION adaptation metrics, side by side. The ANIL sweep "
                         "reads these curves to extract M* rather than parsing stdout. Both "
                         "signals are recorded because they answer different questions and "
                         "are computed on different episode distributions — query_loss on "
                         "`--episode-start any` training episodes, val_set_mae on the true "
                         "session prefix through the DEPLOYED adaptation.")
    ap.add_argument("--patience", type=int, default=10,
                    help="Early-stop after this many meta-epochs without val set-MAE improvement.")
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
    if args.order == "imaml" and args.tau <= 0.0:
        # Without a binding prox term the inner problem is not strictly convex,
        # so its argmin is not unique and the implicit gradient is undefined.
        # Numerically the Hessian is singular too (condition number ~1e17), so
        # this used to surface as a silently wrong solve rather than an error.
        raise ValueError(
            "--order imaml requires --tau > 0: the implicit meta-gradient is "
            "defined by the proximal term, and without it the inner argmin is "
            f"neither unique nor numerically solvable (got --tau {args.tau})."
        )
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
    # Widened once, here. The head is a META-PARAMETER, so from this point it is
    # simply a wider one; `embed` reads the width off the model, which is why no
    # inner-loop, implicit-solve or meta-validation call site needs to change.
    if args.embed_fcd:
        install_fcd_head(model, True)
        print(f"[embed-fcd] meta-learned head widened to {model.head.in_features} inputs")
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
    rows = [normalize_row(r) for r in iter_jsonl(pathlib.Path(args.in_jsonl))]
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
        # See train_XLSTM: SENTINEL_VALUES is the one place a missing-marker is
        # declared, so train / meta-train / serve cannot disagree.
        default = SENTINEL_VALUES.get(k, 0.0)
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

    # Meta-val K values (clipped per driver so a query tail always remains).
    try:
        val_ks = sorted({int(t) for t in args.val_ks.split(",") if t.strip()})
    except ValueError as e:
        raise ValueError(
            f"--val-ks must be comma-separated integers, got {args.val_ks!r}") from e
    if not val_ks or val_ks[0] < 1:
        raise ValueError(f"--val-ks must be positive integers, got {val_ks}")
    print(f"[val] meta-validation K grid: {val_ks} "
          f"(flat mean over {len(val_pids)} driver(s) x {len(val_ks)} K = "
          f"{len(val_pids) * len(val_ks)} cells)")

    # Per-epoch metrics for the sweep. TWO signals, deliberately both:
    #   query_loss  — meta-TRAINING progress, on `--episode-start any` episodes.
    #   val_set_mae — meta-VALIDATION, on the deployed adaptation over the true
    #                 session PREFIX of held-out drivers.
    # They are computed on DIFFERENT episode distributions by design (see
    # docs/meta_optimization_options.md), so a gap between them is not evidence
    # of meta-overfitting and must not be read as one. Selection and M* key off
    # val_set_mae alone; query_loss is here to show whether meta-training is
    # progressing at all, which val_set_mae cannot distinguish from a bad init.
    metrics_fh = None
    metrics_writer = None
    if args.metrics_csv:
        mp = pathlib.Path(args.metrics_csv); mp.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = mp.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_fh)
        metrics_writer.writerow([
            "epoch", "order_this_epoch", "query_loss",
            "val_set_mae", "val_set_qwk", "val_n_drivers", "val_ks",
            "inner_res_max", "inner_unconverged", "n_episodes"]
            # APPENDED, so the leading columns keep their positions and every
            # existing reader (all csv.DictReader) is unaffected. The mean alone
            # cannot answer the question the wider grid was added for -- whether
            # the meta-init helps more as labels accumulate -- because a flat
            # mean hides the slope across K.
            + ["val_set_acc"]
            + [f"val_set_mae_k{k}" for k in val_ks]
            + [f"val_set_acc_k{k}" for k in val_ks])

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
                    support = [(augment_segment(X, rng, args.crop_frac, args.jitter_std), v)
                               for X, v in support]
                    query = [(augment_segment(X, rng, args.crop_frac, args.jitter_std), v)
                             for X, v in query]

                # Embed support/query. Support keeps its graph whenever the
                # support pathway into the backbone is used (second/imaml);
                # first order drops it. Query always keeps its graph.
                xs, ls, vs = collate(support)
                Zs = embed(model, xs, ls, device, grad=second or imaml)
                xq, lq, vq = collate(query)
                Zq = embed(model, xq, lq, device, grad=True)

                if imaml:
                    # iMAML: solve the proximal head problem to its argmin and
                    # accumulate the exact implicit meta-gradients (anchor,
                    # support and query pathways) — see imaml_meta_step.
                    lq_val, res = imaml_meta_step(
                        Zs, vs.to(device), Zq, vq.to(device),
                        model.head.weight, model.head.bias,
                        loss_fn, args.tau, 1.0 / nb,
                        args.imaml_max_iter, args.imaml_tol,
                        head_type=head_type)
                    batch_loss += lq_val * nb # for logging
                    ep_res.append(res)
                else:
                    # Inner: adapt the head on support with truncated proximal SGD.
                    # tau, not lambda: inner_adapt derives this episode's lambda
                    # from its own support size (see that function).
                    w, b = inner_adapt(Zs, vs.to(device), model.head.weight, model.head.bias,
                                       loss_fn, args.inner_steps, args.inner_lr, args.tau,
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
            maes, accs, qwks = [], [], []
            per_k_mae = {k: [] for k in val_ks}
            per_k_acc = {k: [] for k in val_ks}
            clipped = set()
            for pid in val_pids:
                res = evaluate_adaptation_multi_k(
                    model, drivers[pid], val_ks, collate, device, loss_fn, head_type,
                    args.val_adapt_steps, args.val_adapt_lr, args.tau)
                for K, (m, a, q, Kc) in res.items():
                    maes.append(m); accs.append(a); qwks.append(q)
                    per_k_mae[K].append(m); per_k_acc[K].append(a)
                    if Kc != K:
                        clipped.add((pid, K, Kc))
            model.train()
            if clipped and ep == 0:
                # Once, at epoch 0. Two requested K values clipping to the same
                # effective K are counted twice in the flat mean, silently
                # reweighting the selection criterion toward that driver.
                print(f"[warn] --val-ks clipped (pid, asked, used): {sorted(clipped)} — "
                      f"that driver has too few segments to leave a query tail.")
            val_mae, val_qwk = float(np.mean(maes)), float(np.mean(qwks))
            val_acc = float(np.mean(accs))
            kmae = {k: (float(np.mean(v)) if v else float("nan"))
                    for k, v in per_k_mae.items()}
            kacc = {k: (float(np.mean(v)) if v else float("nan"))
                    for k, v in per_k_acc.items()}
            k_note = " ".join(f"K{k}={kmae[k]:.3f}" for k in val_ks)
            print(f"[epoch {ep:02d}] query_loss={train_loss:.4f} "
                  f"val_set-MAE={val_mae:.3f} val_set-ACC={val_acc:.3f} "
                  f"val_set-QWK={val_qwk:.3f} (drivers={len(val_pids)}) "
                  f"[{k_note}]{res_note}")
            if metrics_writer is not None:
                metrics_writer.writerow([
                    ep, ("first" if warm else args.order), train_loss,
                    val_mae, val_qwk, len(val_pids), "|".join(str(k) for k in val_ks),
                    (max(ep_res) if ep_res else ""),
                    (sum(r > args.imaml_tol for r in ep_res) if ep_res else ""),
                    len(ep_losses)]
                    + [val_acc]
                    + [kmae[k] for k in val_ks]
                    + [kacc[k] for k in val_ks])
                metrics_fh.flush()   # the sweep reads these while the run is in flight
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
            if metrics_writer is not None:
                metrics_writer.writerow([
                    ep, ("first" if warm else args.order), train_loss,
                    "", "", 0, "", (max(ep_res) if ep_res else ""),
                    (sum(r > args.imaml_tol for r in ep_res) if ep_res else ""),
                    len(ep_losses)] + ["" for _ in range(1 + 2 * len(val_ks))])
                metrics_fh.flush()
            save_checkpoint(model, str(outp), arch=arch)

    if metrics_fh is not None:
        metrics_fh.close()

    if val_pids:
        print(f"[BEST] val_set-MAE={best_val:.3f} -> {outp}")
    print("[next] compare inits with the SAME per-driver protocol, e.g.:\n"
          f"  python -m scripts.sweep_train_frac --in-data data/labeled_pXXX.jsonl "
          f"--in-model {outp} --out results/sweep_pXXX_anil.png")


if __name__ == "__main__":
    main()
