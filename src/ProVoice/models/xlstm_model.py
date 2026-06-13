"""Shared xLSTM model definition and feature schema.

SINGLE source of truth imported by BOTH the trainer (``train_XLSTM.py``) and the
inference strategy (``decision_engine.StateXLSTMLoAStrategy``). Because the model
architecture and the feature encoding live here, the train/serve mismatch that
previously existed (custom exp-gated LSTM at train time vs. stock ``nn.LSTM`` at
inference time, loaded with ``strict=False``) is structurally impossible.

Uses the OFFICIAL nx-ai/xlstm library (``xlstm==2.0.5``), classic
``xLSTMBlockStack`` with mLSTM-only blocks (pure PyTorch, no triton/CUDA).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function

# Guarded import of the heavy xlstm symbols. The verified import path is pure
# torch (no triton/ninja/CUDA), but we still guard so this module imports even if
# the dependency is missing -- callers can inspect XLSTM_AVAILABLE.
try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
    )
    XLSTM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when xlstm is absent
    xLSTMBlockStack = None
    xLSTMBlockStackConfig = None
    mLSTMBlockConfig = None
    mLSTMLayerConfig = None
    XLSTM_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Canonical feature schema (ONE fixed order, used everywhere).
# --------------------------------------------------------------------------- #
STATE_NUM = ['drowsiness_alert', 'gaze_distracted', 'heart_rate']
STATE_CAT = ['environment', 'secondary_task', 'lab', 'emotion']

# FCD values, then each NUM (1 value), then each CAT (2 values).
D_IN = len(FCD_NAMES) + len(STATE_NUM) + 2 * len(STATE_CAT)

DEFAULT_CONTEXT_LENGTH = 256


def _as01(x: Any) -> float:
    """Coerce a loosely-typed truthiness/value into a float in a robust way.

    true/1/yes -> 1.0 ; false/0/no/""/nan/none -> 0.0 ; else float(x) ; on
    failure 0.0.
    """
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    s = str(x).strip().lower()
    if s in ('true', '1', 't', 'yes', 'y'):
        return 1.0
    if s in ('false', '0', 'f', 'no', 'n', '', 'nan', 'none', 'null'):
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def encode_frame(functionname: str, row: Dict[str, Any]) -> np.ndarray:
    """Encode a single timestep into a ``float32`` vector of length ``D_IN``.

    Layout: FCD values (via ``get_fcd_for_function``), then each NUM via
    ``_as01``, then for each CAT two values ``[1.0 if non-empty else 0.0,
    min(len(str(v))/16.0, 1.0)]``.
    """
    fcd = get_fcd_for_function(functionname or "")
    fcd_vec = [float(fcd[k]) for k in FCD_NAMES]
    num = [_as01(row.get(k)) for k in STATE_NUM]
    catv = []
    for k in STATE_CAT:
        v = row.get(k, "")
        c = "" if v is None else str(v)
        catv.extend([1.0 if c != "" else 0.0, min(len(c) / 16.0, 1.0)])
    return np.asarray([*fcd_vec, *num, *catv], dtype=np.float32)


def _stack_cfg(embedding_dim: int, num_blocks: int, num_heads: int, context_length: int):
    """Build the validated classic mLSTM-only xLSTMBlockStack config."""
    return xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=num_heads,
            )
        ),
        slstm_block=None,
        context_length=context_length,
        num_blocks=num_blocks,
        embedding_dim=embedding_dim,
        add_post_blocks_norm=True,
        bias=False,
        dropout=0.0,
        slstm_at=[],
    )


class XLSTMSequenceClassifier(nn.Module):
    """Input proj -> xLSTMBlockStack -> last-step pool -> linear classifier."""

    def __init__(
        self,
        d_in: int = D_IN,
        n_classes: int = 5,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        pool: str = 'last',
    ):
        super().__init__()
        if not XLSTM_AVAILABLE:
            raise ImportError(
                "xlstm is not available; cannot build XLSTMSequenceClassifier."
            )
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.d_in = d_in
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.context_length = context_length
        self.pool = pool

        self.in_proj = nn.Linear(d_in, embedding_dim)
        self.backbone = xLSTMBlockStack(
            _stack_cfg(embedding_dim, num_blocks, num_heads, context_length)
        )
        self.head = nn.Linear(embedding_dim, n_classes)
        self.backbone.reset_parameters()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.to(torch.float32)
        h = self.in_proj(x)
        h = self.backbone(h)
        # Sequences are LEFT-padded, so the last timestep is always valid.
        pooled = h[:, -1, :]
        return self.head(pooled)


def save_checkpoint(model: XLSTMSequenceClassifier, path: str, arch: Dict[str, Any]) -> None:
    """Persist model weights plus the exact kwargs needed to rebuild it."""
    torch.save(
        {
            "format_version": 1,
            "xlstm_version": "2.0.5",
            "arch": arch,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, map_location: str = 'cpu') -> Tuple[XLSTMSequenceClassifier, Dict[str, Any]]:
    """Load a checkpoint, rebuild the model, and strict-load its weights."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    assert ckpt.get("format_version") == 1, (
        f"Unsupported checkpoint format_version: {ckpt.get('format_version')!r}"
    )
    arch = ckpt["arch"]
    model = XLSTMSequenceClassifier(**arch)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, arch
