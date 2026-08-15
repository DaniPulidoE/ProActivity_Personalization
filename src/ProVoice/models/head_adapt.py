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
    """
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
    w = anchor_w.clone().requires_grad_(True)
    b = anchor_b.clone().requires_grad_(True)
    # weight_decay=0 is load-bearing: the L2-SP term below is the regularizer,
    # and an additional decay toward the ORIGIN would pull the head away from
    # the population anchor rather than toward it.
    opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=0.0)

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
    grad_norm = float(torch.sqrt((w.grad ** 2).sum() + (b.grad ** 2).sum()))
    opt.zero_grad()

    info = {
        "n": n,
        "tau": float(tau),
        "l2sp": lam,
        "steps": int(steps),
        "lr": float(lr),
        "head_type": head_type,
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
        tau=tau, head_type=head_type, steps=steps, lr=lr)
    head = copy.deepcopy(pop_head)
    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)
    return head, info
