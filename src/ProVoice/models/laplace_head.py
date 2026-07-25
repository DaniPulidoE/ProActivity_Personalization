"""Laplace posterior over the adapted CORN head — UQ Layer 1.

Strictly post-hoc: nothing in training or fine-tuning changes. L2-SP head
fine-tuning is MAP estimation under a Gaussian prior centred on the population
head, so after adaptation the posterior is approximated as a Gaussian at the
trained head with the EXACT Hessian of the negative log-posterior (each CORN
unit is a logistic regression with linear logits, so no GGN approximation is
needed and the posterior is provably log-concave/unimodal).

Prior-precision scaling. ``fine_tune_XLSTM.py`` minimizes
``corn_loss(...) + lam * ||theta - theta_pop||^2`` where ``corn_loss`` is
normalized by M = the total number of conditional training examples
(sum over units of |S_j|). Rescaling by M gives the un-normalized objective
``NLL(theta) + M*lam*||theta - theta_pop||^2``, i.e. MAP under the prior
N(theta_pop, (2*M*lam)^{-1} I). The Hessian of that negative log-posterior,
per CORN unit j, is exact and block-diagonal across units (units share no
parameters):

    H_j = sum_{i in S_j} sigma_i (1 - sigma_i) * z~_i z~_i^T  +  2*M*lam * I

with z~ = [z; 1] the bias-augmented embedding, sigma_i the unit's sigmoid at
the MAP logits, and S_j = {i : y_i >= j} the conditional subset unit j is
trained on (target 1[y > j]) — the same subsets ``coral_pytorch.losses
.corn_loss`` uses. Units whose subset is empty (labels never reach level j)
keep the pure prior H_j = 2*M*lam*I: no data, prior-width uncertainty.

Both study arms (L2-SP baseline and ANIL) end in the same head fine-tune, so
fitting here applies the identical UQ mechanism to both arms by construction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ProVoice.models.xlstm_model import logits_to_probs

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
        prior_precision: tau = 2*M*lam actually used in H
        l2sp:            the lambda the head was fine-tuned with
        n_cond_examples: M, the corn_loss normalizer
    """

    def __init__(
        self,
        mean: torch.Tensor,
        chol_prec: torch.Tensor,
        n_classes: int,
        prior_precision: float,
        l2sp: float,
        n_cond_examples: int,
    ):
        self.mean = mean
        self.chol_prec = chol_prec
        self.n_classes = int(n_classes)
        self.prior_precision = float(prior_precision)
        self.l2sp = float(l2sp)
        self.n_cond_examples = int(n_cond_examples)

    # ------------------------------------------------------------------ fit
    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        head: nn.Linear,
        Z: torch.Tensor,
        y: torch.Tensor,
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
            y:         (N,) integer labels 0..n_classes-1
            l2sp:      the L2-SP strength lambda used during fine-tuning. Must
                       be > 0: the Laplace layer is defined by the L2-SP prior;
                       without it the prior precision is 0 and units with few
                       (or no) conditional examples have a singular Hessian.
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
        y = y.detach().to("cpu", torch.long)
        if Z.ndim != 2 or Z.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch: Z {tuple(Z.shape)} vs y {tuple(y.shape)}")
        if Z.shape[1] != head.in_features:
            raise ValueError(f"Z dim {Z.shape[1]} != head.in_features {head.in_features}")

        # weights
        W = head.weight.detach().to("cpu", torch.float64)          # (K-1, E)
        b = head.bias.detach().to("cpu", torch.float64)            # (K-1,)
        mean = torch.cat([W, b.unsqueeze(1)], dim=1)               # (K-1, E+1)

        # Conditional subsets of corn_loss: unit j trains on {y >= j}.
        subsets = [y >= j for j in range(K1)]
        M = int(sum(int(s.sum()) for s in subsets))
        if M == 0:
            raise ValueError("No training examples — cannot fit a posterior.")
        tau = 2.0 * M * float(l2sp)

        P = Z.shape[1] + 1
        eye = torch.eye(P, dtype=torch.float64)
        H = torch.empty(K1, P, P, dtype=torch.float64)
        for j in range(K1):
            Hj = tau * eye # prior covariance
            s = subsets[j]
            if bool(s.any()): # update prior with likelihood
                Zj = torch.cat([Z[s], torch.ones(int(s.sum()), 1, dtype=torch.float64)], dim=1)
                sig = torch.sigmoid(Zj @ mean[j])                  # (|S_j|,)
                w = sig * (1.0 - sig)
                Hj = Hj + (Zj * w.unsqueeze(1)).T @ Zj
            H[j] = Hj
        chol_prec = torch.linalg.cholesky(H)                       # PD: tau > 0

        return cls(
            mean=mean,
            chol_prec=chol_prec,
            n_classes=n_classes,
            prior_precision=tau,
            l2sp=l2sp,
            n_cond_examples=M,
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
