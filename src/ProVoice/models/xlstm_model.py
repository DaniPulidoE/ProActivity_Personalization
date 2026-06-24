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

import ast
import json
from typing import Any, Dict, List, Optional, Tuple

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
STATE_NUM = ['perclos', 'gaze_score', 'hr_delta', 'rr_delta', 'blink_rate', 'yawn_rate']
STATE_CARLA = ['speed_ratio_max', 'speed_ratio_limit', 'brake', 'steer', 'precipitation', 'is_night', 'is_junction']
STATE_CAT_LEN = ['environment', 'secondary_task']
STATE_CAT_ONE_HOT = ['emotion', 'lab']
STATE_CAT = STATE_CAT_LEN + STATE_CAT_ONE_HOT


# one-hot encoding for emotion and lab categories
EMOTION_VOCAB = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
LAB_VOCAB = ['face', 'phone', 'drink', 'smoke', 'distracted', 'safe'] # leave only phone and drink if we use the detection model !

# FCD values, then each NUM (1 value), then each CAT (2 values).
#D_IN = len(FCD_NAMES) + len(STATE_NUM) + 2 * len(STATE_CAT)
# with one-hot encoding for emotion and lab categories
D_IN = len(FCD_NAMES) + len(STATE_NUM) + len(STATE_CARLA) + len(STATE_CAT_LEN) + len(EMOTION_VOCAB) + len(LAB_VOCAB) 

DEFAULT_CONTEXT_LENGTH = 400

# Canonical feature names — same order as the 40-dim vector produced by encode_frame().
# Used to label log entries so they can be read without the source code.
FEATURE_NAMES: List[str] = (
    list(FCD_NAMES)                                       # 12  FCD dims (normalised [1,5]→[0,1])
    + list(STATE_NUM)                                     #  6  driver state numerics
    + list(STATE_CARLA)                                   #  7  CARLA vehicle/world
    + ["environment_len", "secondary_task_len"]           #  2  string-length context encoding
    + [f"emotion_{e}" for e in EMOTION_VOCAB]             #  7  one-hot emotion
    + [f"lab_{l}"     for l in LAB_VOCAB]                 #  6  multi-hot distraction
)
assert len(FEATURE_NAMES) == D_IN, f"FEATURE_NAMES length {len(FEATURE_NAMES)} != D_IN {D_IN}"


def log_encoded_frames(
    fh,
    mode: str,
    tag: str,
    frames: np.ndarray,
    label: Optional[int] = None,
) -> None:
    """Append one JSONL line per frame to an open file handle.

    Args:
        fh:     open writable text file handle
        mode:   "train", "val", or "infer"
        tag:    segment_id (train/val) or timestamp string (infer)
        frames: float32 array shape (T, D_IN) — only real frames, no zero padding
        label:  integer LoA class 0-4 (train/val only; omit for infer)
    """
    for i in range(len(frames)):
        vec = frames[i]
        entry: Dict[str, Any] = {"mode": mode, "tag": tag, "frame_idx": i}
        if label is not None:
            entry["label"] = label
        for j, name in enumerate(FEATURE_NAMES):
            entry[name] = round(float(vec[j]), 6)
        fh.write(json.dumps(entry) + "\n")
    fh.flush()


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

    Layout: 12 FCD values (normalised [1,5]→[0,1]), 6 driver-state NUM values,
    7 CARLA vehicle/world values (speed_ratio_max, speed_ratio_limit, brake,
    steer, precipitation, is_night, is_junction), 2 string-length CAT values
    (environment, secondary_task), 7 emotion one-hot values, 6 lab multi-hot
    values. Total: D_IN = 40.

    ``lab`` can be a Python list (from live DataCollector or JSONL), a
    string-encoded list (``"['face', 'phone']"`` — produced when pandas
    stringifies the column), or a plain string (``"phone"``). All forms are
    normalised to a list before building the multi-hot vector.
    """
    fcd = get_fcd_for_function(functionname or "")
    fcd_vec = [(float(fcd[k]) - 1.0) / 4.0 for k in FCD_NAMES]   # [1,5] → [0,1]
    num = [_as01(row.get(k)) for k in STATE_NUM]
    num_carla = [_as01(row.get(k)) for k in STATE_CARLA]
    catv = []
    for k in STATE_CAT:
        v = row.get(k, "")
        if k == 'emotion':
            vec = [1.0 if v == e else 0.0 for e in EMOTION_VOCAB]
        elif k == 'lab':
            if isinstance(v, list):
                lab_list = v
            else:
                s = str(v).strip()
                try:
                    parsed = ast.literal_eval(s)
                    lab_list = parsed if isinstance(parsed, list) else ([s] if s else [])
                except Exception:
                    lab_list = [s] if s else []
            vec = [1.0 if label in lab_list else 0.0 for label in LAB_VOCAB]
        else:
            c = "" if v is None else str(v)
            vec = [min(len(c) / 16.0, 1.0)]
        catv.extend(vec)
    vec = np.asarray([*fcd_vec, *num, *num_carla, *catv], dtype=np.float32)
    return vec


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
    """Persist model weights, arch kwargs, and feature normalisation stats."""
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
    if ckpt.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format_version: {ckpt.get('format_version')!r}")

    arch = ckpt["arch"]
    model = XLSTMSequenceClassifier(**arch)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, arch
