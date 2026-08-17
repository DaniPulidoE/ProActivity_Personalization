"""Per-driver head adaptation — the ONE optimizer the sweep and the fine-tuner share.

``fine_tune_XLSTM.py`` produces the head that gets SERVED; ``sweep_train_frac.py``
produces the quality-vs-K learning curve the study reads its K values off. Those
two must therefore be the same estimator, and until 2026-08-14 they were not:

===================  ==========================  ============================
                     sweep_train_frac (curve)    fine_tune_XLSTM (served)
===================  ==========================  ============================
batching             full-batch                  mini-batch, --batch 16
optimizer steps      fixed (300)                 epochs * ceil(K/16) -> ~K
epoch selection      none, returns final head    best val set-MAE checkpointed
lr / anchor          5e-4 / lam=0.01             2e-3 / lam=0.01
===================  ==========================  ============================

Two of those differences are functions of K, which is fatal for an instrument
whose whole job is to resolve a curve ALONG K. This module removes all four by
being the only implementation.

Why the sweep's structure won and not the fine-tuner's:

* **Full batch.** The CORN head is ``Linear(64->4)`` = 260 parameters fitted on
  at most ~100 cached embeddings, with a strictly convex objective. There is no
  conditioning or memory argument for mini-batching that; ``--batch 16`` was
  structure inherited from the population trainer. Full-batch also makes the
  step budget independent of K for free, and removes seed variance entirely —
  a given (K, tau, steps, lr) yields one head, every time.
* **No epoch selection.** Selecting the best epoch on the validation tail costs
  three things: at deployment it needs a tail, so some of the driver's K labels
  are spent on selection rather than adaptation; it selects and reports on the
  same segments; and its optimism is LARGER at small K (fewer val segments =>
  noisier argmin), so it lifts the low-K end of the curve more than the high-K
  end and bends the axis the study measures. Fix the step count a priori
  instead — on the development drivers, with the other hyperparameters.
* **It puts the Laplace layer at the MAP.** ``laplace_head`` expands the
  posterior to second order *about the MAP*, which is where its exactness
  argument lives (strictly convex, provably unimodal, exact per-unit Hessian —
  no GGN). An epoch-selected, mini-batch-stopped head is not a stationary
  point, so the linear term is not zero and the Gaussian is centred slightly
  off. Running full-batch to convergence puts theta where the theory assumes.

TAU, NOT LAMBDA — the interface takes prior precision.
------------------------------------------------------
``soft_corn_loss`` is a batch MEAN, so minimizing ``(1/K) sum_i NLL_i +
lam*||theta - theta_pop||^2`` is stationary where ``sum_i grad NLL_i + 2*K*lam
(theta - theta_pop) = 0``: the effective prior precision is **tau = 2*K*lam**,
exactly the scaling ``laplace_head`` documents. Holding ``lam`` fixed while
sweeping K therefore makes the anchor STRONGER as data accumulates — backwards,
and a mechanical distortion of the learning curve. It also inverts the design's
graceful-degradation claim at the low end: as K -> 0 with lam fixed, tau -> 0,
so the driver with the FEWEST labels gets the WEAKEST prior.

A prior is a belief about the driver held before seeing their data, so it must
not depend on how much data is about to arrive. This module therefore takes
``tau`` and derives ``lam = tau / (2K)`` internally, where a caller cannot
forget to. Fix tau once on the development drivers and it is comparable across
every K, every driver and both study arms.

NOT COVERED HERE: ``xlstm_maml``'s inner loop. It runs a few DIFFERENTIABLE SGD
steps (second derivatives flow through them), so it cannot call this function,
and it uses the batch-mean convention with prior ``2*lam`` — see the note in
``laplace_head``. Keeping the deployed adaptation identical across arms means
the ANIL arm must adapt through THIS function at evaluation time even though it
meta-trains through its own; only the initialization is allowed to differ.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ProVoice.models.xlstm_model import levels_to_distribution, soft_corn_loss

# Full-batch steps and LR, MEASURED (2026-08-14) on participant 001's cached
# embeddings at tau=2 rather than guessed. Grid over K in {10, 99}:
#
#     steps   lr      loss @K=10  |grad|     loss @K=99  |grad|
#       300   5e-4      0.444383  3.3e-01      1.049237  1.6e-01   <- old defaults
#       300   5e-3      0.365329  3.9e-03      0.888454  1.1e-02
#      2000   5e-4      0.365582  1.2e-02      0.891121  1.8e-02
#      2000   5e-3      0.365294  1.4e-07      0.886259  1.2e-07   <- chosen
#      2000   1e-2      0.365294  2.1e-07      0.886259  1.4e-07
#      5000   1e-2      0.365294  6.5e-05      0.886267  4.9e-03   (oscillating)
#
# The old 300 @ 5e-4 left the objective 21 % above its optimum at K=10 and 18 %
# at K=99 — under-converged, and under-converged BY A K-DEPENDENT AMOUNT, which
# is the same class of artifact as the drifting anchor this module exists to
# remove. Anyone reading a sweep curve produced before this change is looking
# partly at how far each point's optimizer got.
#
# 1e-2 converges equally well at 2000 steps but starts oscillating by 5000, so
# 5e-3 is the safer of the two. `info['grad_norm']` re-checks this every run;
# these defaults are not meant to be trusted on faith across other tau values.
DEFAULT_ADAPT_STEPS = 2000
DEFAULT_ADAPT_LR = 5e-3
# Prior precision. tau = 2*K*lam, so this matches the previous lam=0.01 default
# at K=100 -- i.e. it reproduces roughly the old anchor strength at a FULL
# per-driver support set, and now holds that strength constant as K shrinks
# instead of letting it collapse. Tune it on the development drivers; it is a
# study-level constant, not a per-run knob.
DEFAULT_TAU = 2.0

# ---------------------------------------------------------------------------
# FCD-AUGMENTED HEAD INPUT  (--embed-fcd)
#
# WHAT: the adapted head sees [z_64 | FCD_12] instead of z_64 alone.
#
# WHY: the label on this cohort is close to a deterministic function of
# (driver x task). A per-(driver, function) constant reaches set-MAE 0.260 on the
# within-driver split where the trained model reaches 0.956, and it wins from
# K=5 onward. The model is GIVEN the task -- FCD occupies dims 0..11 of every
# frame -- but only as 12 constant channels dragged through 100 recurrent steps
# and entangled with driver state by the time the head sees a pooled vector.
# Head adaptation therefore cannot cheaply express "this driver wants more
# autonomy for THIS task", which is the structure that dominates the data.
#
# Concatenating FCD at the head gives adaptation direct access. The 5 functions
# in the study have FCD vectors of rank 5 (with bias), so a LINEAR map over them
# can hit arbitrary per-function values: the augmented head can represent the
# lookup baseline exactly, rather than approximating it.
#
# THE NEW BLOCK IS ANCHORED AT ZERO. The population head has no weights for
# these columns, so theta_pop = [W_pop | 0]. That is not a fudge: it makes the
# L2-SP penalty an ordinary ridge on the new block, and at K=0 the FCD weights
# are zero, so the adapted head is IDENTICAL to the population head. Graceful
# degradation is preserved by construction -- `assert_zero_block_identity`
# checks it rather than trusting it.
#
# Nothing upstream moves: the backbone, the population checkpoint, E* and the
# stage-1/2 results are all untouched. Only the personalization layer gains
# capacity. The Laplace Hessian stays exact and PSD at any embedding width, and
# tau = 2*K*lambda is unchanged.
# ---------------------------------------------------------------------------
FCD_DIM = 12


def fcd_from_frames(X: torch.Tensor) -> torch.Tensor:
    """``(B, T, D)`` frames -> ``(B, FCD_DIM)``.

    Read from timestep 0 because FCD is STATIC per function: verified exactly
    constant across every segment in the cache (max per-column std 0.0 over 100
    frames). Taking t=0 rather than a mean is deliberate -- a mean would silently
    paper over a future feature that is not actually constant.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (B, T, D) frames, got {tuple(X.shape)}")
    if X.shape[2] < FCD_DIM:
        raise ValueError(f"frames have {X.shape[2]} features, need >= {FCD_DIM} for FCD")
    return X[:, 0, :FCD_DIM].to(torch.float32)


def augment_z(Z: torch.Tensor, X: torch.Tensor, embed_fcd: bool) -> torch.Tensor:
    """``[z | FCD]`` when ``embed_fcd``, else ``z`` unchanged.

    One definition, called at every site that builds embeddings for the head, so
    the two arms cannot end up adapting different objects.
    """
    if not embed_fcd:
        return Z
    return torch.cat([Z, fcd_from_frames(X).to(Z.device)], dim=1)


def expand_head_for_fcd(w0: torch.Tensor, b0: torch.Tensor,
                        embed_fcd: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Population head ``(n_out, d)`` -> ``(n_out, d + FCD_DIM)`` with ZEROS appended.

    Both the initialization AND the L2-SP anchor: the appended block starts at 0
    and is pulled back toward 0, so K=0 reproduces the population head exactly.
    Idempotent-safe -- pass ``embed_fcd=False`` and the head comes back untouched.
    """
    if not embed_fcd:
        return w0, b0
    pad = torch.zeros(w0.shape[0], FCD_DIM, dtype=w0.dtype, device=w0.device)
    return torch.cat([w0, pad], dim=1), b0


def install_fcd_head(model, embed_fcd: bool):
    """Widen ``model.head`` to accept ``[z | FCD]``, zero-initialized.

    The appended block is both the initialization and the L2-SP anchor, so a
    freshly widened head predicts EXACTLY what the population head predicted --
    K=0 is unchanged by construction. Idempotent: a head that is already wide,
    or ``embed_fcd=False``, is returned untouched.

    Installing it on the model rather than carrying tensors around is what keeps
    the option from leaking into every call site: ``forward`` infers the
    augmentation from ``head.in_features``, ``save_checkpoint`` persists the wide
    head, and serving picks it up with no flag at all.
    """
    import torch.nn as _nn
    from ProVoice.models.head_adapt import expand_head_for_fcd
    if not embed_fcd or model.head_uses_fcd():
        return model
    w, b = expand_head_for_fcd(model.head.weight.detach(), model.head.bias.detach(), True)
    head = _nn.Linear(w.shape[1], w.shape[0])
    head.weight.data, head.bias.data = w.clone(), b.clone()
    model.head = head.to(w.device)
    return model


def assert_zero_block_identity(model, X: torch.Tensor, lengths: torch.Tensor,
                               atol: float = 1e-6) -> None:
    """THE GATE for --embed-fcd: the augmented head with a zero FCD block must
    reproduce the population head bit-for-bit.

    If this passes, the change is safe by construction: every K=0 prediction, and
    therefore the entire graceful-degradation guarantee, is provably unchanged.
    Cheap enough to run before any sweep that uses the flag.
    """
    with torch.no_grad():
        h = model.backbone(model.in_proj(X.to(torch.float32)))
        idx = (lengths.long() - 1).clamp(min=0)
        z = h[torch.arange(h.size(0), device=h.device), idx]
        w0, b0 = model.head.weight.detach(), model.head.bias.detach()
        base = F.linear(z, w0, b0)
        w1, b1 = expand_head_for_fcd(w0, b0, True)
        aug = F.linear(augment_z(z, X, True), w1, b1)
        gap = (aug - base).abs().max().item()
    if gap > atol:
        raise AssertionError(
            f"--embed-fcd zero-block identity FAILED: max|augmented - population| "
            f"= {gap:.3e} > {atol:g}. The FCD block is not inert at "
            f"initialization, so K=0 would no longer reproduce the population "
            f"model and graceful degradation is broken.")
    print(f"[embed-fcd] zero-block identity OK (max diff {gap:.2e}) — K=0 "
          f"reproduces the population head exactly")


def l2sp_from_tau(tau: float, n: int) -> float:
    """``lam`` for a batch-MEAN objective that realises prior precision ``tau``.

    Inverse of ``tau = 2*n*lam`` (see the module docstring). Pass the result to
    ``LaplacePosterior.fit(..., l2sp=...)``, which re-derives the same tau from
    it, so the trained anchor and the posterior's prior cannot disagree.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"l2sp_from_tau needs at least one support example, got n={n}")
    if float(tau) <= 0.0:
        raise ValueError(
            f"tau must be > 0, got {tau}. tau=0 removes the L2-SP anchor entirely: "
            "adaptation is then unregularized and the Laplace layer is undefined "
            "(its prior precision would be 0).")
    return float(tau) / (2.0 * float(n))


def loss_for_head(head_type: str):
    """The training loss for a head type — one definition, all call sites.

    Both forms consume the multi-hot mark vector directly, so a window where the
    driver marked several acceptable LoAs never has to be collapsed to one.
    """
    if head_type == "corn":
        return soft_corn_loss
    if head_type == "softmax":
        _ce = nn.CrossEntropyLoss()
        return lambda logits, lvl: _ce(logits, levels_to_distribution(lvl))
    raise ValueError(f"Unknown head_type: {head_type!r}")


def adapt_head_tensors(
    Z: torch.Tensor,
    V: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    *,
    tau: float = DEFAULT_TAU,
    head_type: str = "corn",
    steps: int = DEFAULT_ADAPT_STEPS,
    lr: float = DEFAULT_ADAPT_LR,
    adapt_params: str = "all",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Tensor-level adaptation: ``(w, b)`` in, adapted ``(w, b, info)`` out.

    The implementation; :func:`adapt_head` is the ``nn.Linear`` convenience
    wrapper. This form exists for ``xlstm_maml``, which carries the head as raw
    parameter tensors so its meta-gradients can flow through them.

    ``w0``/``b0`` are the anchor AND the initialization — detached copies are
    taken, so the caller's tensors are never mutated and no gradient escapes
    into them. **The returned tensors are detached**: this is the DEPLOYED
    adaptation, run at serving and at meta-VALIDATION, not a differentiable
    inner loop. ANIL's meta-training inner loop is ``xlstm_maml.inner_adapt``,
    which must keep its trajectory in the graph and therefore cannot be this.

    ``adapt_params`` restricts WHICH head parameters move. ``'all'`` (the
    default, and the only value any serving path uses) adapts the 256 weights
    and 4 biases. ``'bias'`` freezes the weight at the anchor and adapts the 4
    biases alone -- the diagnostic of ``docs/embedding_informativeness.md`` §4:
    a bias-only head can slide the CORN thresholds along the ordinal scale
    ("this driver prefers more/less autonomy") but cannot reorder segments, so
    it is the per-driver constant expressed in the head. If it matches the full
    head, personalization on this representation reduces to learning a level
    offset and both study arms are tied by construction.

    The variant lives HERE rather than in the analysis script on purpose: §4
    requires ``tau``, ``steps`` and ``lr`` to be identical across the two, or
    the comparison measures optimizer budget instead of expressiveness. Sharing
    one optimizer is what makes that true by construction. With
    ``adapt_params='all'`` the code path is unchanged from before this flag
    existed -- the weight is still an AdamW parameter with the same init.
    """
    if adapt_params not in ("all", "bias"):
        raise ValueError(
            f"adapt_params must be 'all' or 'bias', got {adapt_params!r}")
    if Z.ndim != 2 or V.ndim != 2 or Z.shape[0] != V.shape[0]:
        raise ValueError(
            f"adapt_head expects (K, d) embeddings and (K, n_classes) levels with "
            f"matching K, got {tuple(Z.shape)} and {tuple(V.shape)}")
    # Devices must already agree. Not moved silently: which device is "right"
    # depends on the caller (the sweep embeds on the GPU then adapts on the CPU,
    # because adaptation is kernel-launch-bound and gains nothing from CUDA),
    # so quietly relocating tensors here would hide a caller's real mistake.
    # Raised early with the offending devices named, because the alternative is
    # an opaque matmul error several frames deeper.
    if w0.device != Z.device or b0.device != Z.device:
        raise ValueError(
            f"device mismatch: embeddings on {Z.device}, head weights on "
            f"{w0.device}/{b0.device}. Move the head to the embeddings' device "
            f"(or vice versa) before calling — e.g. `head.to(Z.device)`.")
    n = int(Z.shape[0])
    lam = l2sp_from_tau(tau, n)
    loss_fn = loss_for_head(head_type)

    anchor_w, anchor_b = w0.detach().clone(), b0.detach().clone()
    # Under 'bias' the weight stays EXACTLY at the anchor: it is excluded from
    # the parameter list, so no optimizer state is created for it and its L2-SP
    # term is the constant 0 rather than a decaying one. The objective is then
    # the same function restricted to the bias subspace, at the same tau.
    train_w = adapt_params == "all"
    w = anchor_w.clone().requires_grad_(train_w)
    b = anchor_b.clone().requires_grad_(True)
    params = [w, b] if train_w else [b]
    # weight_decay=0 is load-bearing: the L2-SP term below is the regularizer,
    # and an additional decay toward the ORIGIN would pull the head away from
    # the population anchor rather than toward it.
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    def objective() -> torch.Tensor:
        pen = ((w - anchor_w) ** 2).sum() + ((b - anchor_b) ** 2).sum()
        return loss_fn(F.linear(Z, w, b), V) + lam * pen

    for _ in range(int(steps)):
        loss = objective()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Convergence diagnostic, not a training step: one extra backward with no
    # opt.step(), so the returned parameters are exactly what the loop produced.
    final = objective()
    opt.zero_grad()
    final.backward()
    grad_norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in params)))
    opt.zero_grad()

    info = {
        "n": n,
        "tau": float(tau),
        "l2sp": lam,
        "steps": int(steps),
        "lr": float(lr),
        "head_type": head_type,
        "adapt_params": adapt_params,
        "n_adapted": int(sum(p.numel() for p in params)),
        "final_loss": float(final.detach()),
        "grad_norm": grad_norm,
    }
    return w.detach(), b.detach(), info


def adapt_head(
    pop_head: nn.Linear,
    Z: torch.Tensor,
    V: torch.Tensor,
    *,
    tau: float = DEFAULT_TAU,
    head_type: str = "corn",
    steps: int = DEFAULT_ADAPT_STEPS,
    lr: float = DEFAULT_ADAPT_LR,
    adapt_params: str = "all",
) -> Tuple[nn.Linear, Dict[str, Any]]:
    """Adapt a copy of ``pop_head`` to one driver's support set. Deterministic.

    Args:
        pop_head:  the population head. Copied, never mutated — it is also the
                   L2-SP anchor, so it has to survive the call unchanged.
        Z:         (K, embedding_dim) cached embeddings from the frozen backbone.
        V:         (K, n_classes) multi-hot marked-level targets.
        tau:       prior precision. ``lam = tau / (2K)`` is derived here.
        head_type: 'corn' (soft-CORN) or 'softmax' (CE) — take it from the
                   checkpoint arch, never from a CLI flag.
        steps:     full-batch gradient steps. Fixed, K-independent, no selection.
        lr:        AdamW learning rate; weight_decay is 0 because L2-SP IS the
                   decay, anchored at theta_pop rather than at the origin.
        adapt_params: 'all' (every serving path) or 'bias' (the 4 biases only,
                   for the expressiveness diagnostic of
                   docs/embedding_informativeness.md §4).

    Returns ``(head, info)``. ``info`` carries the realised ``l2sp`` (needed by
    the Laplace fit), the final objective value, and ``grad_norm`` — the norm of
    the full objective's gradient at the returned head, i.e. the distance from
    the stationary point the Laplace expansion assumes. Check it: on a convex
    260-parameter problem it should be small, and a large value means ``steps``
    or ``lr`` is wrong, not that the driver is unusual.
    """
    if pop_head.bias is None:
        raise ValueError("adapt_head expects a Linear head WITH a bias term.")
    w, b, info = adapt_head_tensors(
        Z, V, pop_head.weight, pop_head.bias,
        tau=tau, head_type=head_type, steps=steps, lr=lr,
        adapt_params=adapt_params)
    head = copy.deepcopy(pop_head)
    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)
    return head, info
