"""Laplace posterior over the adapted soft-CORN head — UQ Layer 1.

Strictly post-hoc: nothing in training or fine-tuning changes. L2-SP head
fine-tuning is MAP estimation under a Gaussian prior centred on the population
head, so after adaptation the posterior is approximated as a Gaussian at the
trained head with the EXACT Hessian of the negative log-posterior (each CORN
unit is a logistic regression with linear logits, so no GGN approximation is
needed and the posterior is provably log-concave/unimodal).

Why soft targets change nothing structural. For BCE with a linear logit the
second derivative is sigma*(1-sigma) times the per-sample WEIGHT, independent
of the target value — the canonical-link GLM property (Immer, Korzepa & Bauer,
AISTATS 2021, is the reference for why this is the load-bearing assumption).
Soft-CORN only moves the targets and the weights, so the Hessian keeps its
exact closed form; what generalizes is the conditional subset. Hard CORN counts
sample i in unit j iff y_i >= j; soft-CORN counts it with WEIGHT
w_ij = P_i(y >= j), which is exactly ``levels_to_subset_weights``. A single
marked level makes w_ij the old 0/1 indicator, so this reduces to the previous
implementation exactly.

Prior-precision scaling. ``fine_tune_XLSTM.py`` minimizes
``soft_corn_loss(...) + lam * ||theta - theta_pop||^2`` where
``soft_corn_loss`` is normalized by N = the number of training segments (it
sums over units and averages over the batch). Rescaling by N gives the
un-normalized objective ``NLL(theta) + N*lam*||theta - theta_pop||^2``, i.e.
MAP under the prior N(theta_pop, (2*N*lam)^{-1} I). The Hessian of that
negative log-posterior, per CORN unit j, is exact and block-diagonal across
units (units share no parameters):

    H_j = sum_i w_ij * sigma_ij (1 - sigma_ij) * z~_i z~_i^T  +  2*N*lam * I

with z~ = [z; 1] the bias-augmented embedding and sigma_ij the unit's sigmoid
at the MAP logits. Units with no weight at all (labels never reach level j)
keep the pure prior H_j = 2*N*lam*I: no data, prior-width uncertainty.

NOTE the normalizer changed from M (hard CORN's total subset memberships) to N
with the switch to ``soft_corn_loss``; posteriors fitted by earlier revisions
used tau = 2*M*lam and are not comparable.

Both study arms (L2-SP baseline and ANIL) end in the same head fine-tune, so
fitting here applies the identical UQ mechanism to both arms by construction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ProVoice.models.xlstm_model import logits_to_probs, levels_to_subset_weights

# Checkpoint key under which the serialized posterior rides alongside "arch"
# and "state_dict" in the .pt file (absent on checkpoints fit without UQ).
LAPLACE_KEY = "laplace"


class LaplacePosterior:
    """Exact Gaussian posterior over a CORN head's [weight; bias] parameters.

    Block-diagonal across the K-1 CORN units. Internally float64 (65x65
    Cholesky factors; negligible cost, no conditioning surprises).

    Attributes:
        mean:            (K-1, E+1) MAP head, row j = [w_j; b_j]
        chol_prec:       (K-1, E+1, E+1) lower Cholesky L of the Hessian H=LL^T
        n_classes:       K (PMF length after CORN decoding)
        prior_precision: tau = 2*N*lam actually used in H
        l2sp:            the lambda the head was fine-tuned with
        n_examples:      N, the soft_corn_loss normalizer (defines tau)
        n_cond_examples: sum of the soft subset weights — a diagnostic of how
                         much data each threshold actually saw (equals hard
                         CORN's M when every segment marks one level)
    """

    def __init__(
        self,
        mean: torch.Tensor,
        chol_prec: torch.Tensor,
        n_classes: int,
        prior_precision: float,
        l2sp: float,
        n_examples: int,
        n_cond_examples: float,
    ):
        self.mean = mean
        self.chol_prec = chol_prec
        self.n_classes = int(n_classes)
        self.prior_precision = float(prior_precision)
        self.l2sp = float(l2sp)
        self.n_examples = int(n_examples)
        self.n_cond_examples = float(n_cond_examples)

    # ------------------------------------------------------------------ fit
    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        head: nn.Linear,
        Z: torch.Tensor,
        levels: torch.Tensor,
        l2sp: float,
        n_classes: int = 5,
    ) -> "LaplacePosterior":
        """Fit the posterior at the adapted head on the ADAPTATION embeddings.

        Args:
            head:      the fine-tuned CORN head, ``Linear(E, n_classes-1)``.
                       Must be the head being shipped (best checkpoint), not an
                       intermediate epoch — the expansion is only valid at the
                       mode of the objective.
            Z:         (N, E) precomputed embeddings the head was trained on
            levels:    (N, n_classes) multi-hot marked levels — the same target
                       ``soft_corn_loss`` was trained on. A one-hot row is the
                       single-label case and reproduces hard-CORN subsets.
            l2sp:      the L2-SP strength lambda used during fine-tuning. Must
                       be > 0: the Laplace layer is defined by the L2-SP prior;
                       without it the prior precision is 0 and units with little
                       (or no) conditional weight have a singular Hessian.
            n_classes: K (5 LoA levels)
        """
        # verify inputs
        if l2sp <= 0.0:
            raise ValueError(
                f"l2sp must be > 0 to define the prior precision, got {l2sp}. "
                "Fit the head with an L2-SP anchor before applying Laplace."
            )
        K1 = n_classes - 1
        if head.out_features != K1:
            raise ValueError(
                f"Expected a CORN head with {K1} outputs, got {head.out_features} "
                "(softmax heads are not supported: units share parameters through "
                "the softmax, so the per-unit GLM decomposition does not apply)."
            )
        Z = Z.detach().to("cpu", torch.float64)
        levels = levels.detach().to("cpu", torch.float64)
        if Z.ndim != 2 or levels.ndim != 2 or Z.shape[0] != levels.shape[0]:
            raise ValueError(
                f"Shape mismatch: Z {tuple(Z.shape)} vs levels {tuple(levels.shape)}")
        if levels.shape[1] != n_classes:
            raise ValueError(
                f"levels has {levels.shape[1]} columns, expected n_classes={n_classes}")
        if Z.shape[1] != head.in_features:
            raise ValueError(f"Z dim {Z.shape[1]} != head.in_features {head.in_features}")

        N = Z.shape[0]
        if N == 0:
            raise ValueError("No training examples — cannot fit a posterior.")

        # weights
        W = head.weight.detach().to("cpu", torch.float64)          # (K-1, E)
        b = head.bias.detach().to("cpu", torch.float64)            # (K-1,)
        mean = torch.cat([W, b.unsqueeze(1)], dim=1)               # (K-1, E+1)

        # Soft conditional-subset weights: w_ij = P_i(y >= j). Reduces to the
        # hard indicator 1[y_i >= j] when exactly one level is marked.
        Wt = levels_to_subset_weights(levels)                      # (N, K-1)
        soft_M = float(Wt.sum())
        # soft_corn_loss averages over the batch, so the un-normalized objective
        # is recovered by multiplying by N — that N is the prior scale.
        tau = 2.0 * N * float(l2sp)

        P = Z.shape[1] + 1
        eye = torch.eye(P, dtype=torch.float64)
        Zt = torch.cat([Z, torch.ones(N, 1, dtype=torch.float64)], dim=1)   # (N, E+1)
        H = torch.empty(K1, P, P, dtype=torch.float64)
        for j in range(K1):
            Hj = tau * eye # prior precision
            wj = Wt[:, j]
            if bool((wj > 0).any()): # update prior with likelihood
                sig = torch.sigmoid(Zt @ mean[j])                  # (N,)
                w = wj * sig * (1.0 - sig)                         # weighted GLM curvature
                Hj = Hj + (Zt * w.unsqueeze(1)).T @ Zt
            H[j] = Hj
        chol_prec = torch.linalg.cholesky(H)                       # PD: tau > 0

        return cls(
            mean=mean,
            chol_prec=chol_prec,
            n_classes=n_classes,
            prior_precision=tau,
            l2sp=l2sp,
            n_examples=N,
            n_cond_examples=soft_M,
        )

    # ------------------------------------------------------------- sampling
    def sample_heads(
        self, n_samples: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Draw head samples ~ N(mean, H^{-1}); returns (n_samples, K-1, E+1).

        With H = L L^T, x = mean + L^{-T} eps has covariance H^{-1}.
        """
        K1, P = self.mean.shape
        eps = torch.randn(n_samples, K1, P, 1, dtype=torch.float64, generator=generator)
        delta = torch.linalg.solve_triangular(
            self.chol_prec.transpose(-1, -2), eps, upper=True
        )
        return self.mean + delta.squeeze(-1)

    def pmf_samples(
        self,
        Z: torch.Tensor,
        n_samples: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Per-sample decoded PMFs, shape (n_samples, B, n_classes).

        Each sampled head is decoded through the shared ``logits_to_probs``
        CORN decoding, so posterior samples and the deterministic path cannot
        diverge. Use the spread across dim 0 for contraction/width curves.
        """
        Z64 = Z.detach().to("cpu", torch.float64)
        Zt = torch.cat([Z64, torch.ones(Z64.shape[0], 1, dtype=torch.float64)], dim=1)
        thetas = self.sample_heads(n_samples, generator=generator)  # (n, K-1, E+1)
        logits = torch.einsum("bp,skp->sbk", Zt, thetas)            # (n, B, K-1)
        return logits_to_probs(logits, "corn")

    def predictive_pmf(
        self,
        Z: torch.Tensor,
        n_samples: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Posterior-averaged PMF, shape (B, n_classes), float32.

        Drop-in replacement for ``logits_to_probs(head(Z), 'corn')`` wherever a
        calibrated PMF is wanted (quantile decoding, conformal scores).
        """
        return self.pmf_samples(Z, n_samples, generator).mean(dim=0).to(torch.float32)

    def predictive_loa_std(
        self,
        Z: torch.Tensor,
        n_samples: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Posterior std of the expected LoA per segment, shape (B,), float32.

        The posterior-width quantity for the convergence criterion (research
        question b): plot its mean vs. K next to the MAE/QWK learning curve.
        """
        pmfs = self.pmf_samples(Z, n_samples, generator)            # (n, B, K)
        k = torch.arange(self.n_classes, dtype=pmfs.dtype)
        eloa = (pmfs * k).sum(dim=-1)                               # (n, B)
        return eloa.std(dim=0).to(torch.float32)

    # -------------------------------------------------------- serialization
    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "chol_prec": self.chol_prec,
            "n_classes": self.n_classes,
            "prior_precision": self.prior_precision,
            "l2sp": self.l2sp,
            "n_examples": self.n_examples,
            "n_cond_examples": self.n_cond_examples,
        }

    @classmethod
    def from_state_dict(cls, sd: Dict[str, Any]) -> "LaplacePosterior":
        return cls(**sd)


# ------------------------------------------------------- checkpoint helpers
def attach_laplace_to_checkpoint(path: str, posterior: LaplacePosterior) -> None:
    """Store the posterior in an existing .pt checkpoint under LAPLACE_KEY.

    Read-modify-write on the whole file so ``xlstm_model.save_checkpoint`` /
    ``load_checkpoint`` stay untouched and older checkpoints (no posterior)
    remain valid.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt[LAPLACE_KEY] = posterior.state_dict()
    torch.save(ckpt, path)


def load_laplace_from_checkpoint(path: str) -> Optional[LaplacePosterior]:
    """Return the stored posterior, or None if the checkpoint has none."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get(LAPLACE_KEY)
    return None if sd is None else LaplacePosterior.from_state_dict(sd)
