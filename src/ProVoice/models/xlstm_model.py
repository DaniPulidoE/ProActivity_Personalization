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
            vec = [1.0 if v.strip().lower() == e else 0.0 for e in EMOTION_VOCAB]
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
    """Input proj -> xLSTMBlockStack -> readout at last real frame -> linear classifier.

    ``head_type`` selects the output parameterization:
      - 'softmax': ``Linear(embedding_dim, n_classes)`` logits for CE/softmax.
      - 'corn':    ``Linear(embedding_dim, n_classes - 1)`` logits, where logit k
        models the conditional P(y > k | y > k-1) (Shi, Cao & Raschka 2023).
        Class probabilities are recovered with :func:`logits_to_probs`; train
        with ``coral_pytorch.losses.corn_loss``.
      - 'coral':   ONE shared weight vector plus ``n_classes - 1`` independent
        biases (Cao, Mirjalili & Raschka 2020), so logit k is ``w·h + b_k`` and
        ``sigmoid`` of it is the unconditional P(y > k). Sharing w is what makes
        the thresholds rank-consistent — an independent Linear per threshold
        (as 'corn' uses) would not be CORAL. Train with
        ``coral_pytorch.losses.coral_loss``, which takes a cumulative target
        vector and therefore accepts SOFT targets: a driver who marks several
        acceptable LoAs becomes q_k = P(y > k) of the uniform distribution over
        that set. Single-label data yields the usual 0/1 vector unchanged.
    """

    def __init__(
        self,
        d_in: int = D_IN,
        n_classes: int = 5,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        #pool: str = 'last',
        head_type: str = 'softmax',
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
        if head_type not in ('softmax', 'corn', 'coral'):
            raise ValueError(
                f"head_type must be 'softmax', 'corn' or 'coral', got {head_type!r}")
        self.d_in = d_in
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.context_length = context_length
        #self.pool = pool
        self.head_type = head_type

        self.in_proj = nn.Linear(d_in, embedding_dim)
        self.backbone = xLSTMBlockStack(
            _stack_cfg(embedding_dim, num_blocks, num_heads, context_length)
        )
        if head_type == 'coral':
            # Shared weight vector, one bias per threshold. This is the whole
            # point of CORAL: identical w across thresholds means their ordering
            # is the same for every input, so the model cannot emit
            # contradictory ranks.
            self.head = nn.Linear(embedding_dim, 1, bias=False)
            self.coral_bias = nn.Parameter(torch.zeros(n_classes - 1))
        else:
            n_out = n_classes - 1 if head_type == 'corn' else n_classes
            self.head = nn.Linear(embedding_dim, n_out)
        self.backbone.reset_parameters()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Batches are RIGHT-padded; ``lengths`` holds the true frame counts.

        The readout is the hidden state at the last REAL frame (index
        ``lengths-1``). The stack is causal, so pad frames sit after the
        readout point and cannot influence the output — padding is exactly
        neutral, not merely learned-to-ignore. Omit ``lengths`` for unpadded
        input (e.g. batch-of-1 inference), where the last timestep is real.
        """
        x = x.to(torch.float32)
        h = self.in_proj(x)
        h = self.backbone(h)
        if lengths is None:
            pooled = h[:, -1, :]
        else:
            idx = (lengths.to(h.device).long() - 1).clamp(min=0)
            pooled = h[torch.arange(h.size(0), device=h.device), idx]
        out = self.head(pooled)
        if self.head_type == 'coral':
            out = out + self.coral_bias   # (B, 1) + (K-1,) -> (B, K-1)
        return out


def logits_to_probs(logits: torch.Tensor, head_type: str = 'softmax') -> torch.Tensor:
    """Convert head logits to a (B, n_classes) probability distribution.

    SINGLE source of truth for decoding, shared by the trainers and the
    inference strategy so train/serve decoding cannot diverge.

    - 'softmax': plain softmax over n_classes logits.
    - 'corn': logits are K-1 conditionals P(y>k | y>k-1). The chain rule gives
      cumulative probs q_k = P(y>k) as a running product of sigmoids (monotone
      non-increasing by construction => rank-consistent), and differencing the
      q_k yields a valid PMF: p_0 = 1-q_1, p_k = q_k - q_{k+1}, p_{K-1} = q_{K-1}.
    """
    if head_type == 'softmax':
        return torch.softmax(logits, dim=-1)
    if head_type in ('corn', 'coral'):
        if head_type == 'corn':
            q = torch.cumprod(torch.sigmoid(logits), dim=-1)  # (B, K-1), q_k = P(y > k-1)
        else:
            # CORAL logits are UNCONDITIONAL P(y > k), so no chain rule. The
            # shared weight vector makes the thresholds rank-consistent, but the
            # learned biases are not constrained to be ordered; take a running
            # minimum so differencing can never produce a negative probability.
            q = torch.cummin(torch.sigmoid(logits), dim=-1).values
        ones = torch.ones_like(q[..., :1])
        upper = torch.cat([ones, q], dim=-1)                  # [1, q_1, ..., q_{K-1}]
        lower = torch.cat([q, torch.zeros_like(q[..., :1])], dim=-1)  # [q_1, ..., q_{K-1}, 0]
        return upper - lower                                  # non-negative because q is monotone
    raise ValueError(f"Unknown head_type: {head_type!r}")


def save_checkpoint(model: XLSTMSequenceClassifier, path: str, arch: Dict[str, Any]) -> None:
    """Persist model weights, arch kwargs."""
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

    arch = dict(ckpt["arch"])
    # Legacy key: older checkpoints stored pool='last'. Pooling is now fixed to
    # last-step (the only behaviour forward() ever implemented), so the key is
    # dropped rather than passed to a constructor that no longer accepts it.
    arch.pop("pool", None)
    # 'window_seconds' is a DATA contract (how segments were cut at training),
    # not a model kwarg: keep it in the returned arch for callers (fine-tuning,
    # sweep, inference) but don't pass it to the constructor.
    ctor_kwargs = {k: v for k, v in arch.items() if k != "window_seconds"}
    model = XLSTMSequenceClassifier(**ctor_kwargs)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, arch
