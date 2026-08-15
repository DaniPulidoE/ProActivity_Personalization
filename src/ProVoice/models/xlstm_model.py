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
import math
import os
import pathlib
import sysconfig
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function

def _ensure_cuda_home() -> None:
    """Make ``import xlstm`` survive a GPU box that has no CUDA *Toolkit*.

    ``xlstm/__init__.py`` imports sLSTM unconditionally, and sLSTM's
    ``src/cuda_init.py`` runs, AT IMPORT TIME and guarded only by
    ``if torch.cuda.is_available():``::

        os.environ["CUDA_LIB"] = os.path.join(os.path.split(
            torch.utils.cpp_extension.include_paths(device_type="cuda")[-1])[0], "lib")

    That helper ends in ``_join_cuda_home``, which raises::

        OSError: CUDA_HOME environment variable is not set.

    whenever the CUDA Toolkit (``nvcc`` + headers) is absent. It normally IS
    absent: the NVIDIA driver ships the CUDA *runtime*, not the SDK, and the
    ``+cu128`` wheel bundles runtime libraries only. The net effect is backwards
    -- ``import xlstm`` works on a CPU-only box and fails on a GPU one, which is
    why this surfaced only after the CUDA overlay was installed.

    Nothing here compiles those kernels: ``_stack_cfg`` builds an mLSTM-ONLY
    stack (pure PyTorch). ``cuda_init`` merely interpolates the path into
    ``os.environ["CUDA_LIB"]`` and never opens it, so pointing CUDA_HOME at the
    CUDA runtime the torch wheel already ships is enough to get past the import.

    Deliberately does nothing when CUDA_HOME is already set: on a machine with a
    real Toolkit, overwriting it would break anything that genuinely compiles.
    """
    try:
        if not torch.cuda.is_available():
            return          # CPU-only: cuda_init's guard skips the whole block
    except Exception:       # pragma: no cover - torch too broken to ask
        return

    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    warn = False
    if not home:
        for cand in (site / "nvidia" / "cuda_runtime", site / "nvidia" / "cuda_nvcc"):
            if (cand / "include").is_dir():
                home = str(cand)
                break
        else:
            # Nothing bundled found. Any existing directory still gets the import
            # through, because the value is only ever string-formatted on this
            # path. If a future change DOES need real kernels, the thing to fix
            # is the missing Toolkit, not this line.
            home, warn = str(site), True
    os.environ["CUDA_HOME"] = home

    # Setting the environment variable is NOT sufficient on its own. torch reads
    # it ONCE, in torch.utils.cpp_extension's module body, and freezes the result
    # in a module-level CUDA_HOME constant; _join_cuda_home then tests that
    # constant, not os.environ. If anything imported cpp_extension before this
    # runs, the constant is already None and the env var is ignored. Assigning it
    # directly covers both orderings, so this cannot depend on import sequence.
    try:
        import torch.utils.cpp_extension as _ce
        if getattr(_ce, "CUDA_HOME", None) is None:
            _ce.CUDA_HOME = home
    except Exception:       # pragma: no cover
        pass

    if warn:
        print("[xlstm][warn] CUDA is available but no CUDA Toolkit was found; CUDA_HOME "
              f"set to {site} purely to get xlstm's unused sLSTM import past "
              "_join_cuda_home. Fine for the mLSTM-only stack used here; install the "
              "real Toolkit before compiling any CUDA kernel.")


_ensure_cuda_home()

# Guarded import of the heavy xlstm symbols. The COMPUTE path is pure torch (no
# triton/ninja/CUDA), but the IMPORT path is not on a GPU box -- see
# _ensure_cuda_home above, which must run first.
#
# OSError is caught alongside ImportError because that is the shape the missing
# CUDA Toolkit takes; without it the failure is an unhandled traceback rather
# than a clean XLSTM_AVAILABLE = False.
try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
    )
    XLSTM_AVAILABLE = True
except (ImportError, OSError) as _xlstm_err:  # pragma: no cover
    print(f"[xlstm][warn] xlstm unavailable: {type(_xlstm_err).__name__}: {_xlstm_err}")
    xLSTMBlockStack = None
    xLSTMBlockStackConfig = None
    mLSTMBlockConfig = None
    mLSTMLayerConfig = None
    XLSTM_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Canonical feature schema (ONE fixed order, used everywhere).
# --------------------------------------------------------------------------- #
# no respiratory rate (pure noise)
STATE_NUM = ['perclos', 'gaze_score', 'hr_delta', 'blink_rate', 'yawn_rate', 'ear', 'mar']
# no speed ratio limit because all map has the same speed limit
# no precipitation or night - no weather in CARLA0.10.0
STATE_CARLA = ['speed_ratio_max', 'brake', 'steer', 'throttle', 'is_junction', 'lead_distance_m']
STATE_CAT_LEN = []
STATE_CAT_ONE_HOT = ['emotion', 'lab']
STATE_CAT =  STATE_CAT_ONE_HOT


# one-hot encoding for emotion and lab categories
EMOTION_VOCAB = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
LAB_VOCAB = ['face'] # discard drinks due to lack of data

# FCD values, then each NUM (1 value), then each CAT (2 values).
#D_IN = len(FCD_NAMES) + len(STATE_NUM) + 2 * len(STATE_CAT)
# with one-hot encoding for emotion and lab categories
D_IN = len(FCD_NAMES) + len(STATE_NUM) + len(STATE_CARLA) + len(EMOTION_VOCAB) + len(LAB_VOCAB)

DEFAULT_CONTEXT_LENGTH = 400

# Canonical feature names — same order as the D_IN-dim vector produced by encode_frame().
# Used to label log entries so they can be read without the source code.
FEATURE_NAMES: List[str] = (
    list(FCD_NAMES)                                       # 12  FCD dims (normalised [1,5]→[0,1])
    + list(STATE_NUM)                                     #  8  driver state numerics
    + list(STATE_CARLA)                                   #  6  CARLA vehicle/world
    + [f"emotion_{e}" for e in EMOTION_VOCAB]             #  7  one-hot emotion
    + [f"lab_{l}"     for l in LAB_VOCAB]                 #  2  multi-hot distraction
)
assert len(FEATURE_NAMES) == D_IN, f"FEATURE_NAMES length {len(FEATURE_NAMES)} != D_IN {D_IN}"


# --------------------------------------------------------------------------- #
# Fixed-grid resampling — a DATA contract, shared by train and serve.
# --------------------------------------------------------------------------- #
# The collection loop is paced by `sampling_interval` but only ever falls BELOW
# its nominal rate (CPU contention, camera stalls), so the achieved rate varies
# per session: 19.6 Hz and 9.3 Hz have both been measured on the same machine on
# the same day. Moving perception onto worker threads has since made the rate
# much more stable -- 18.6 and 17.9 Hz over two 15-min sessions on 2026-08-07,
# median inter-frame gap 51-53 ms, one gap >0.5 s in 32,892 frames -- but that
# is a property of one machine under one load, not a guarantee, and the whole
# point of this contract is that nothing downstream has to trust it.
# `truncate_frames_by_seconds` already pins the WALL-CLOCK span of
# the input, but the number of recurrent steps spanning that span still floats
# with the rate — 20 s is ~396 frames at 19.6 Hz and ~186 at 9.3 Hz. An xLSTM
# learns its time constants per STEP, not per second, so that is a 2x rescaling
# of every learned dynamic from one session to the next. Worse, the achieved
# rate is set by scene load (night, rain, junctions, YOLO), which correlates
# with both the covariates and the labels, so sequence length is a leakage
# channel: the model can read the condition off the step count instead of off
# the driver.
#
# Resampling each window onto a fixed time grid makes the step count a function
# of the window's DURATION alone. 10 Hz is chosen from the physiology, not from
# the camera: the eye-closure indicator has an integrated autocorrelation time
# of ~830 ms, so a 10 Hz grid is ~8 samples per correlation time — oversampled
# for every feature in STATE_NUM, all of which are 30 s- or 180 s-smoothed.
#
# This runs DOWNSTREAM of the detectors, so it fixes the temporal geometry only.
# A feature mismeasured at collection time (see the duration-gated blink counter
# in data_collector._visual_process, whose yield drops once dt approaches the
# 100/500 ms gate) stays mismeasured; that is a separate problem.
DEFAULT_RESAMPLE_HZ = 10.0

# Columns that may be LINEARLY interpolated onto the grid. Everything else is
# carried from the last real frame at or before each grid point.
#
# The criterion is NOT "is this a float" — it is whether the change BETWEEN two
# consecutive real samples is small relative to the feature's own quantum. Where
# it is, interpolation estimates a value the signal genuinely passed through;
# where it is not, interpolation manufactures a value the pipeline never
# produced and the model never saw at training time.
#
# What is held, and why:
#   * emotion one-hot / lab multi-hot — a blend (emotion_happy=0.4 next to
#     emotion_sad=0.6) is a state that never occurred.
#   * is_night / is_junction — binary.
#   * environment_len / secondary_task_len — a CATEGORY encoded as a string
#     length; the midpoint of two lengths is not a category.
#   * the 12 FCD dims — ordinal 1-5 ratings looked up per FUNCTION, not measured
#     over time. Constant while functionname is (the common case, and always so
#     at serving, where one name applies to the whole window), but a segment
#     spanning a function change would otherwise get a blend of two functions'
#     profiles — a rating vector no real function has.
#   * blink_rate — a deterministic function of an INTEGER count over a trailing
#     window, so there is no continuous latent value to estimate. Two regimes,
#     both measured on data/raw_data.jsonl: for the first 30 s of a session
#     `elapsed_s` is still growing, so the rate decays as 60N/elapsed (370
#     distinct values in one session); after that `elapsed_s` pins to 30 and the
#     rate is exactly 2N — a strict lattice of 2 blinks/min (8 distinct values,
#     84% of all frames). Holding costs <=0.13 blinks/min of ramp error in the
#     first regime and zero in the second; interpolating costs up to a full 2.0
#     quantum in the second. Hold dominates, so the ramp needs no special case.
#   * yawn_rate — same, and without even the ramp: the divisor is the fixed
#     180 s window, so the served feature is Anscombe(N) on {0, 1.12, 1.86, ...}
#     for integer N. Anything between those is a fractional yawn.
#   * hr_delta / rr_delta — the rPPG worker emits an estimate only about every
#     6 s and the collection loop reads the held value on every tick, so these
#     are staircases with ~6.7 s plateaus (measured: 25 and 21 distinct values
#     across a 4867 s recording). Heart rate is a continuous latent signal, but
#     the pipeline never observed it between readings; a plateau of 6.7 s is a
#     third of a 20 s window, so interpolating would reshape a 3-step staircase
#     into a smooth ramp built entirely from values the system never had.
#     Holding also matches the carry-forward the collector already applies
#     between readings.
#
# What survives as linear is exactly the set measured at FULL loop rate and
# moving in small per-frame increments: the two visual driver-state features and
# the CARLA vehicle/world channels.
_LINEAR_FEATURES = frozenset(
    ['perclos', 'gaze_score']
    + ['speed_ratio_max', 'brake', 'steer', 'lead_distance_m']
)
assert _LINEAR_FEATURES <= set(FEATURE_NAMES), (
    f"unknown feature name in _LINEAR_FEATURES: {_LINEAR_FEATURES - set(FEATURE_NAMES)}")

# Dims that are continuous EXCEPT for a reserved "missing" marker, mapped to
# that marker's value. They interpolate normally among real readings, but a grid
# point whose two bracketing samples include the marker is HELD instead: a
# blend of a real value and a sentinel reads as a real value and is not one.
#
#   * lead_distance_m — metres to the nearest vehicle ahead in the ego lane, or
#     -1 for "no lead vehicle within the 100 m detection range" (see
#     data_collector._carla_process, which logs that case as null; the null->-1
#     mapping is applied here, in encode_frame, so the raw log keeps saying
#     "nothing there" rather than asserting a distance). A distance is
#     non-negative, so -1 is unambiguous. The marker is NOT interchangeable
#     with 0: 0 m is bumper contact, i.e. the most urgent state the feature can
#     express, and is exactly what `_as01(None)` would otherwise produce.
#     Measured on data/raw_data.jsonl: 73.5 % of frames carry the marker and
#     real readings cluster at both ends (p25 6.6 m, p75 55.8 m), so
#     marker<->real transitions are frequent and large — without this guard each
#     one would be smeared over ~5 grid steps into a closing/receding lead
#     vehicle that never existed.
# Feature names the DataCollector logs under a DIFFERENT key. Read through
# ``_row_get`` so the rename happens in ONE place, on every path into the model.
#
# The bug this exists to prevent: 'ear' is logged as 'eye_ar'. The trainer's
# normalize_row aliased it, encode_frame did not, and the serving path hands
# encode_frame the collector's raw dicts -- so the model TRAINED on real EAR
# (~0.2) and was SERVED a constant 0.0 for that dim, silently, because a missing
# key is indistinguishable from a genuine zero once _as01 has run. Aliasing here
# rather than in each caller is the point: normalize_row WHITELISTS keys, so any
# future feature whose log key differs from its feature name fails the same way.
#
# Prefer the canonical name when both are present; fall through only on
# missing/empty, so a row that already carries 'ear' is untouched.
FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = {'ear': ('eye_ar',)}

SENTINEL_VALUES: Dict[str, float] = {'lead_distance_m': -1.0}
assert set(FEATURE_ALIASES) <= set(FEATURE_NAMES), (
    "FEATURE_ALIASES keys must be real feature names: "
    f"{set(FEATURE_ALIASES) - set(FEATURE_NAMES)}")
assert set(SENTINEL_VALUES) <= _LINEAR_FEATURES, (
    "a sentinel column must also be linear, else the guard is dead code: "
    f"{set(SENTINEL_VALUES) - _LINEAR_FEATURES}")
_SENTINEL_COLS: List[Tuple[int, float]] = [
    (FEATURE_NAMES.index(n), float(v)) for n, v in SENTINEL_VALUES.items()]
LINEAR_COLS = np.array(
    [i for i, n in enumerate(FEATURE_NAMES) if n in _LINEAR_FEATURES], dtype=np.intp)
HOLD_COLS = np.array(
    [i for i, n in enumerate(FEATURE_NAMES) if n not in _LINEAR_FEATURES], dtype=np.intp)

# A hole longer than this means the pipeline stopped observing (face absent,
# loop stall) rather than that the signal moved smoothly between two samples.
# Interpolating across it would invent a ramp that never happened, so every
# column is held instead — the same carry-forward the DataCollector already
# applies while the face is out of frame.
RESAMPLE_GAP_S = 1.0


def secs_of_day(ts: Any) -> Optional[float]:
    """Parse a frame timestamp ('HH:MM:SS[.fff]' or full ISO) to seconds-of-day.

    HOT PATH. Called about 400x per decision at 4 Hz — every frame in the window
    is parsed twice, once by ``truncate_frames_by_seconds`` and once by
    ``_relative_times`` — and once per row over the whole training set.
    ``strptime`` costs ~4.7 us/call and is the bulk of that, so the fixed
    ``HH:MM:SS[.fff]`` shape the DataCollector actually writes (100 % of real
    logged frames) is parsed by hand first at ~0.5 us, and everything else falls
    through to the general parsers.

    The arithmetic mirrors the fallback's ``h*3600 + m*60 + s + micro/1e6``
    term for term, including the split of seconds into integer and fractional
    parts, so the two paths agree BIT-FOR-BIT rather than merely closely — a
    reordered float sum would differ in the last ulp and make the fast path a
    silent source of drift between otherwise identical runs.

    The ISO fallback is not optional. The drive UI writes full ISO datetimes for
    label windows, and returning None for those fails SILENTLY in the worst
    direction: ``_relative_times`` DROPS unparseable frames (so a window quietly
    skips resampling and reaches the model on the wrong time grid) and
    ``truncate_frames_by_seconds`` returns the sequence uncut (so ~40-100 s of
    history reaches a model trained on 20 s windows). Neither raises.
    """
    if not ts:
        return None
    s = str(ts).strip()
    # Fast path, guarded on the exact shape: colons in place, all-digit fields,
    # and either nothing or a '.' after the seconds. Anything else — a sign, a
    # space-padded hour, an ISO date, a timezone suffix — drops through to the
    # general parsers rather than being silently mis-read.
    if (len(s) >= 8 and s[2] == ':' and s[5] == ':'
            and (len(s) == 8 or s[8] == '.')
            and s[0:2].isdigit() and s[3:5].isdigit() and s[6:8].isdigit()):
        try:
            return (int(s[0:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
                    + (float(s[8:]) if len(s) > 8 else 0.0))
        except ValueError:
            pass
    t = None
    try:
        t = datetime.fromisoformat(s).time()
    except Exception:
        for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
            try:
                t = datetime.strptime(s, fmt).time()
                break
            except Exception:
                continue
    if t is None:
        return None
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def _relative_times(seq: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """``(kept_indices, seconds_since_first_kept_frame)``, monotone non-decreasing.

    Frames whose timestamp cannot be parsed are DROPPED — resampling needs a time
    for every sample, unlike ``truncate_frames_by_seconds`` which keeps them
    because their age is merely unknowable. Handles midnight wrap-around, since
    frame timestamps are time-of-day only.
    """
    idx: List[int] = []
    secs: List[float] = []
    for i, fr in enumerate(seq):
        t = secs_of_day(fr.get("timestamp")) if isinstance(fr, dict) else None
        if t is not None:
            idx.append(i)
            secs.append(t)
    if len(idx) < 2:
        return np.asarray(idx, dtype=np.intp), np.zeros(len(idx), dtype=np.float64)
    out = np.empty(len(secs), dtype=np.float64)
    out[0] = 0.0
    for k in range(1, len(secs)):
        d = secs[k] - secs[k - 1]
        if d < -43200.0:      # more than half a day backwards → midnight wrap
            d += 86400.0
        elif d < 0.0:         # small backward clock jitter → treat as same instant
            d = 0.0
        out[k] = out[k - 1] + d
    return np.asarray(idx, dtype=np.intp), out


def resample_encoded(
    X: np.ndarray,
    seq: List[Dict[str, Any]],
    resample_hz: Optional[float],
    window_seconds: Optional[float] = None,
    gap_s: float = RESAMPLE_GAP_S,
) -> np.ndarray:
    """Resample encoded frames onto a fixed ``resample_hz`` time grid.

    Args:
        X:              (T, D_IN) encoded frames, one row per entry in ``seq``.
        seq:            the source frame dicts — read only for ``timestamp``.
        resample_hz:    grid rate; None/0 returns ``X`` unchanged (legacy path).
        window_seconds: caps the grid length, so the result is at most
                        ``round(window_seconds * resample_hz)`` steps.
        gap_s:          holes longer than this are held, never interpolated.

    Returns (n, D_IN) where n depends ONLY on the window's duration, never on the
    rate at which it was sampled. Falls back to ``X`` unchanged whenever the
    timestamps cannot support resampling (fewer than 2 parseable), so a
    malformed log degrades to the old behaviour instead of raising.
    """
    if not resample_hz or float(resample_hz) <= 0.0:
        return X
    if X.ndim != 2 or X.shape[0] != len(seq):
        raise ValueError(
            f"resample_encoded: X has {X.shape[0]} rows but seq has {len(seq)} frames")
    keep, t = _relative_times(seq)
    if len(keep) < 2:
        return X
    Xk = X[keep]
    span = float(t[-1] - t[0])
    dt = 1.0 / float(resample_hz)
    if span < dt:
        # Shorter than a single grid step: one sample, the most recent frame.
        return Xk[-1:].astype(np.float32, copy=True)
    n = int(np.floor(span / dt)) + 1
    if window_seconds and float(window_seconds) > 0.0:
        n = min(n, max(1, int(round(float(window_seconds) * float(resample_hz)))))
    # Anchor the grid at the NEWEST frame and step backwards: forward() reads the
    # hidden state at the last timestep, so the most recent observation has to
    # land exactly on it. Stepping back from t[-1] also guarantees grid[0] >=
    # t[0], so the grid never extrapolates before the first real frame.
    grid = t[-1] - dt * np.arange(n - 1, -1, -1, dtype=np.float64)

    # The two real samples bracketing each grid point. `prev` is the last sample
    # at or before it (causal — never a future frame); `nxt` is the one after,
    # clamped at the end where grid[-1] coincides with t[-1].
    prev = np.clip(np.searchsorted(t, grid, side='right') - 1, 0, len(t) - 1)
    nxt = np.minimum(prev + 1, len(t) - 1)

    out = np.empty((n, X.shape[1]), dtype=np.float32)
    for c in LINEAR_COLS:
        out[:, c] = np.interp(grid, t, Xk[:, c])
    if HOLD_COLS.size:
        out[:, HOLD_COLS] = Xk[np.ix_(prev, HOLD_COLS)]
    # Sentinel guard: revert to a hold wherever either bracketing sample carries
    # the reserved "missing" marker, so no grid point lands between a real
    # reading and a sentinel. Interpolation among real readings is untouched.
    for c, sentinel in _SENTINEL_COLS:
        col = Xk[:, c]
        touching = np.isclose(col[prev], sentinel) | np.isclose(col[nxt], sentinel)
        if touching.any():
            out[touching, c] = col[prev[touching]]
    # Long holes: hold every column, including the interpolatable ones.
    stale = (t[nxt] - t[prev]) > float(gap_s)
    if stale.any():
        out[stale] = Xk[prev[stale]]
    return out


def encode_and_resample(
    rows: List[Dict[str, Any]],
    resample_hz: Optional[float] = None,
    window_seconds: Optional[float] = None,
    functionname: Optional[str] = None,
) -> np.ndarray:
    """Encode ``rows`` and resample them — the ONE path train and serve share.

    ``functionname=None`` takes the name per row (the trainer's JSONL carries it
    on every frame); passing it explicitly applies one name to the whole window,
    which is what the decision engine does.
    """
    if not rows:
        raise ValueError("encode_and_resample got an empty frame list")
    xs = [
        encode_frame(
            functionname if functionname is not None else (r.get("functionname") or ""),
            r,
        )
        for r in rows
    ]
    X = np.stack(xs, axis=0).astype(np.float32)
    return resample_encoded(X, rows, resample_hz, window_seconds)


def log_encoded_frames(
    fh,
    mode: str,
    tag: str,
    frames: np.ndarray,
    levels: Optional[np.ndarray] = None,
) -> None:
    """Append one JSONL line per frame to an open file handle.

    Args:
        fh:     open writable text file handle
        mode:   "train", "val", or "infer"
        tag:    segment_id (train/val) or timestamp string (infer)
        frames: float32 array shape (T, D_IN) — only real frames, no zero padding
        levels: multi-hot (K,) mark vector, logged as the LIST of LoA levels the
                driver marked acceptable (train/val only; omit for infer). A
                list, not a scalar: a window may mark several levels, and
                collapsing that to one integer is exactly the bias this
                pipeline no longer carries.
    """
    marked: Optional[List[int]] = None
    if levels is not None:
        marked = [int(k) for k, v in enumerate(np.asarray(levels).ravel()) if v]
    for i in range(len(frames)):
        vec = frames[i]
        entry: Dict[str, Any] = {"mode": mode, "tag": tag, "frame_idx": i}
        if marked is not None:
            entry["levels"] = marked
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


def _row_get(row: Dict[str, Any], name: str) -> Any:
    """``row[name]``, falling back to the log-key aliases in FEATURE_ALIASES.

    Returns None for missing/empty so ``_as01`` and ``_as_sentinel`` apply their
    own defaults exactly as before -- this only widens WHERE the value is looked
    up, it does not change how a genuinely absent value is encoded.
    """
    v = row.get(name)
    if v is not None and v != "":
        return v
    for alt in FEATURE_ALIASES.get(name, ()):
        v = row.get(alt)
        if v is not None and v != "":
            return v
    return None


def _as_sentinel(x: Any, sentinel: float) -> float:
    """Like :func:`_as01`, but ABSENT/None/NaN maps to ``sentinel`` — not 0.0.

    For the columns in :data:`SENTINEL_VALUES` the missing state is a real,
    meaningful state of the world ("no lead vehicle in range"), and 0.0 is a
    legitimate — in fact extreme — value of the same feature. Routing those
    through ``_as01`` would silently encode "the road ahead is clear" as "a
    vehicle is touching the bumper", which is the worst possible confusion for
    a feature the model may lean on to justify a high LoA.

    Note the empty string is treated as missing too: pandas writes NaN out as
    "" through several of the CSV/JSONL paths that feed the trainers.
    """
    if x is None:
        return float(sentinel)
    if isinstance(x, float) and math.isnan(x):
        return float(sentinel)
    if str(x).strip().lower() in ('', 'nan', 'none', 'null'):
        return float(sentinel)
    return _as01(x)


@lru_cache(maxsize=256)
def _fcd_vec_normalized(functionname: str) -> Tuple[float, ...]:
    """The 12 FCD dims for ``functionname``, normalised [1,5] -> [0,1].

    Cached because it is a pure function of the name and the name is CONSTANT
    across a window at serving (``encode_and_resample`` is handed one name for
    the whole sequence) — yet it used to be recomputed per frame: a dict copy
    out of ``get_fcd_for_function`` plus twelve float ops, then discarded.

    Returns a TUPLE, not a list: immutable, so handing the same object to every
    caller cannot let one of them corrupt the cache. ``encode_frame`` only
    splats it, so nothing downstream needs a mutable copy.
    """
    fcd = get_fcd_for_function(functionname)
    return tuple((float(fcd[k]) - 1.0) / 4.0 for k in FCD_NAMES)


def encode_frame(functionname: str, row: Dict[str, Any]) -> np.ndarray:
    """Encode a single timestep into a ``float32`` vector of length ``D_IN``.

    Layout: 12 FCD values (normalised [1,5]→[0,1]), 8 driver-state NUM values,
    6 CARLA vehicle/world values (speed_ratio_max, brake, steer, throttle,
    is_junction, lead_distance_m), 7 emotion one-hot values, 2 lab multi-hot
    values. Total: D_IN = 35. ``FEATURE_NAMES`` is the authoritative order.

    Columns listed in ``SENTINEL_VALUES`` take their reserved marker when the
    field is absent or null instead of ``_as01``'s 0.0 — see ``_as_sentinel``.

    ``lab`` can be a Python list (from live DataCollector or JSONL), a
    string-encoded list (``"['face', 'phone']"`` — produced when pandas
    stringifies the column), or a plain string (``"phone"``). All forms are
    normalised to a list before building the multi-hot vector.
    """
    fcd_vec = _fcd_vec_normalized(functionname or "")            # [1,5] → [0,1]
    num = [_as01(_row_get(row, k)) for k in STATE_NUM]
    num_carla = [_as_sentinel(_row_get(row, k), SENTINEL_VALUES[k]) if k in SENTINEL_VALUES
                 else _as01(_row_get(row, k))
                 for k in STATE_CARLA]
    catv = []
    for k in STATE_CAT:
        v = row.get(k, "")
        if k == 'emotion':
            # An ALL-ZERO one-hot is the encoding of "no emotion reading". It is
            # not a class and must not be confused with one: the DataCollector
            # logs emotion=null whenever the classifier produced nothing (no
            # face, empty crop, recognizer down), and asserting a class there --
            # 'neutral', as this pipeline used to -- tells the model the driver
            # was calm during exactly the frames where their face was not
            # visible. Zeros say "no evidence either way", which is true.
            #
            # This costs no input dimension: the 7-dim block simply sums to 0
            # instead of 1, so D_IN is unchanged and existing checkpoints load.
            # An unrecognised label falls through to the same all-zero vector
            # rather than being silently rounded to a neighbouring class.
            # None also reaches here from pandas via the string 'None'/'nan',
            # which is why the check is on the normalised string, not `is None`.
            es = "" if v is None else str(v).strip().lower()
            vec = ([0.0] * len(EMOTION_VOCAB) if es in ('', 'none', 'null', 'nan')
                   else [1.0 if es == e else 0.0 for e in EMOTION_VOCAB])
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


def _stack_cfg(embedding_dim: int, num_blocks: int, num_heads: int,
               context_length: int, dropout: float = 0.0):
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
        dropout=dropout,
        slstm_at=[],
    )


class XLSTMSequenceClassifier(nn.Module):
    """Input proj -> xLSTMBlockStack -> readout at last real frame -> linear classifier.

    ``head_type`` selects the output parameterization:
      - 'softmax': ``Linear(embedding_dim, n_classes)`` logits for CE/softmax.
      - 'corn':    ``Linear(embedding_dim, n_classes - 1)`` logits, where logit k
        models the conditional P(y > k | y > k-1) (Shi, Cao & Raschka 2023).
        Class probabilities are recovered with :func:`logits_to_probs`; train
        with :func:`soft_corn_loss`, which accepts a driver's SET of acceptable
        LoAs and reduces exactly to the original CORN loss on single-label data.

    ``dropout`` is applied INSIDE the block stack (``xLSTMBlockStackConfig``),
    not to the readout. It is a model kwarg, so it is stored in the checkpoint
    ``arch`` and rebuilt by ``load_checkpoint``; ``model.eval()`` disables it, so
    validation, fine-tuning's frozen-backbone ``embed_all`` pass and serving all
    see the deterministic network. Checkpoints written before it existed lack
    the key and default to 0.0 (the previous behaviour).
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
        dropout: float = 0.0,
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
        if head_type not in ('softmax', 'corn'):
            raise ValueError(
                f"head_type must be 'softmax' or 'corn', got {head_type!r}")
        self.d_in = d_in
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.context_length = context_length
        #self.pool = pool
        self.head_type = head_type
        self.dropout = dropout

        self.in_proj = nn.Linear(d_in, embedding_dim)
        self.backbone = xLSTMBlockStack(
            _stack_cfg(embedding_dim, num_blocks, num_heads, context_length, dropout)
        )
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
        return self.head(pooled)


def logits_to_probs(logits: torch.Tensor, head_type: str = 'softmax') -> torch.Tensor:
    """Convert head logits to a (B, n_classes) probability distribution.

    SINGLE source of truth for decoding, shared by the trainers and the
    inference strategy so train/serve decoding cannot diverge.

    - 'softmax': plain softmax over n_classes logits.
    - 'corn': logits are K-1 conditionals P(y>k | y>k-1). The chain rule gives
      cumulative probs q_k = P(y>k) as a running product of sigmoids (monotone
      non-increasing by construction => rank-consistent), and differencing the
      q_k yields a valid PMF. With q 0-indexed (q_k = P(y>k), k = 0..K-2) and
      the conventions q_{-1} = 1, q_{K-1} = 0, every class is one difference:
      p_k = q_{k-1} - q_k, i.e. p_0 = 1-q_0 and p_{K-1} = q_{K-2}.

    To go all the way to a single LoA, pass the result to :func:`probs_to_label`
    (or use :func:`logits_to_label`) — argmax is NOT the right rule for a CORN
    head.
    """
    if head_type == 'softmax':
        return torch.softmax(logits, dim=-1)
    if head_type == 'corn':
        q = torch.cumprod(torch.sigmoid(logits), dim=-1)       # (B, K-1), q_k = P(y > k)
        ones = torch.ones_like(q[..., :1])
        upper = torch.cat([ones, q], dim=-1)                   # [1, q_0, ..., q_{K-2}]
        lower = torch.cat([q, torch.zeros_like(q[..., :1])], dim=-1)  # [q_0, ..., q_{K-2}, 0]
        return upper - lower                                   # non-negative because q is monotone
    raise ValueError(f"Unknown head_type: {head_type!r}")


# --------------------------------------------------------------------------- #
# PMF -> single LoA. The SECOND half of the decoding contract (logits_to_probs
# is the first), shared by the trainers, the sweep and the decision engine so
# train and serve cannot pick different rules.
# --------------------------------------------------------------------------- #
CORN_RANK_THRESHOLD = 0.5


def probs_to_label(probs: torch.Tensor, head_type: str = 'softmax') -> torch.Tensor:
    """Decode a (B, K) PMF to (B,) integer LoA using each head's canonical rule.

    - **'softmax': argmax** (the mode). A softmax head is nominal — its outputs
      carry no order — so the mode is the only rule its parameterization
      justifies.
    - **'corn': the rank rule of Shi, Cao & Raschka (2023)**,
      ``y_hat = sum_k 1[q_k > 0.5]`` with ``q_k = P(y > k)``: walk up the
      thresholds while the model still believes the label is above them. This
      is CORN's *published* prediction rule and the one its rank-consistency
      guarantee is stated for — `q` is monotone by construction, so the
      indicator flips at most once and ``y_hat`` is exactly the index of that
      flip. Bounded in [0, K-1] with no clamping: all thresholds below 0.5
      gives 0, all above gives K-1.

    Why this is not argmax. ``sum_k 1[q_k > 0.5]`` is the **median** of the
    decoded PMF, whereas argmax is its **mode**. They are optimal for different
    losses — the median minimizes expected absolute error, the mode minimizes
    0/1 errorç. LoA is ordinal and the design selects models on set-MAE, so the
    median is both the better-matched estimator and the one the CORN literature
    reports; argmax was the previous behaviour and is kept available as
    ``logits_to_probs(...).argmax(-1)`` for ablations.

    Note for a CE-vs-CORN comparison: this changes head AND decoder together.
    To isolate the head, decode both arms with argmax explicitly.

    Works on a posterior-AVERAGED PMF too (the Laplace layer's
    ``predictive_pmf``): ``q_k`` is a linear functional of the PMF, so the
    ``q`` of an averaged PMF is the average of the samples' ``q``.
    """
    if head_type == 'softmax':
        return probs.argmax(dim=-1)
    if head_type == 'corn':
        # q_k = P(y > k) = sum_{j > k} p_j — reverse cumulative sum, dropping
        # the k = -1 entry (which is identically 1).
        q = probs.flip(-1).cumsum(-1).flip(-1)[..., 1:]         # (B, K-1)
        return (q > CORN_RANK_THRESHOLD).sum(dim=-1)
    raise ValueError(f"Unknown head_type: {head_type!r}")


def logits_to_label(logits: torch.Tensor, head_type: str = 'softmax') -> torch.Tensor:
    """``probs_to_label(logits_to_probs(logits, ht), ht)`` — the usual entry point."""
    return probs_to_label(logits_to_probs(logits, head_type), head_type)


# --------------------------------------------------------------------------- #
# Ordinal targets and the soft-CORN loss.
#
# The driver may mark SEVERAL acceptable LoAs per 20 s window, so the label is a
# multi-hot vector over the K levels rather than one integer. These three
# functions are the single home for turning that vector into training signal;
# `logits_to_probs` above is their decoding counterpart.
#
# Derivation of soft_corn_loss.
# CORN (Shi, Cao & Raschka 2023) trains unit k as a logistic regression on the
# HARD conditional subset {i : y_i >= k} with binary target 1[y_i > k]. Taking
# the expectation of that per-unit BCE under the label distribution -- weight
# each sample's membership in unit k's subset by P(y >= k) and use the
# conditional target P(y > k | y >= k) -- makes the weights telescope, and the
# whole thing collapses to
#
#     L = - sum_k [ q_k * log sigma(z_k)  +  p_k * log(1 - sigma(z_k)) ]
#
# with q_k = P(y > k) and p_k = P(y = k). It is a strict generalization: when a
# single level is marked, q_k and p_k are 0/1 and this IS the CORN loss, up to
# the normalizer (CORN divides by total subset memberships, this by batch size
# -- a constant factor, i.e. an effective learning-rate rescaling).
# --------------------------------------------------------------------------- #
def levels_to_distribution(levels: torch.Tensor) -> torch.Tensor:
    """Multi-hot (B, K) -> PMF (B, K), uniform over the marked levels.

    A single marked level yields the usual one-hot, so single-label data is
    numerically unchanged.

    Raises on rows that are unmarked or carry negative entries.
    """
    total = levels.sum(dim=-1, keepdim=True)
    # One fused predicate so this costs a single device->host sync per call,
    # negligible at this head size. The per-condition split below runs only on
    # the error path, where extra syncs do not matter.
    if bool(((total <= 0) | (levels < 0).any(dim=-1, keepdim=True)).any()):
        n_empty = int((total <= 0).sum())
        n_neg = int((levels < 0).any(dim=-1).sum())
        raise ValueError(
            f"levels_to_distribution: {n_empty} row(s) do not sum to a positive "
            f"value and {n_neg} row(s) contain negative entries. Every labelled "
            f"window must mark at least one LoA.")
    return levels / total


def levels_to_cumulative(levels: torch.Tensor) -> torch.Tensor:
    """Multi-hot (B, K) -> complementary CDF (B, K-1) with ``q_k = P(y > k)``.

    Monotone non-increasing by construction, and the mapping is invertible, so
    nothing about which levels were marked is lost.
    """
    p = levels_to_distribution(levels)
    return p.flip(-1).cumsum(-1).flip(-1)[..., 1:]


def levels_to_subset_weights(levels: torch.Tensor) -> torch.Tensor:
    """Multi-hot (B, K) -> (B, K-1) soft conditional-subset weights ``P(y >= k)``.

    How much sample i counts toward CORN unit k. In hard CORN this is the
    indicator ``1[y_i >= k]`` of the subset unit k is trained on; under a label
    distribution it is the probability of that event, ``q_{k-1}`` (with
    ``q_{-1} = 1``). This is exactly the per-unit weight that appears in the
    soft-CORN Hessian, so :mod:`ProVoice.models.laplace_head` consumes it too.
    """
    q = levels_to_cumulative(levels)                           # (B, K-1) = q_0..q_{K-2}
    ones = torch.ones_like(q[..., :1])
    return torch.cat([ones, q], dim=-1)[..., :-1]              # [1, q_0, ..., q_{K-3}]


# Saturation bound guarding `0 * (-inf) = NaN`: q_k is exactly 0 above the marked
# level, and that product with a non-finite logit is NaN, not the correct 0 — one
# inf would poison the whole batch. Inert on the loss VALUE (logsigmoid is exactly
# linear below ~-20 in float32) but NOT on the backward pass: like any logistic
# BCE this loss is asymptotically LINEAR in the logit, so its gradient is bounded
# by 1 yet never decays to 0, and `clamp` zeroes a live gradient (-q_k) past the
# bound. The unit then gets no signal from the loss until weight decay (train) or
# the L2-SP anchor (fine-tune) pulls it back — both loss-independent, so the
# freeze is self-healing. By the same mechanism (`clamp`'s backward indicator is
# False for NaN) a NaN logit gives a NaN loss but a ZERO gradient, not a NaN one.
_LOGIT_SATURATION = 1e4


def soft_corn_loss(logits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """CORN loss generalized to a SET of acceptable levels.

    Args:
        logits: (B, K-1) CORN conditional logits.
        levels: (B, K) multi-hot mark vector (a single mark => original CORN).

    Returns the batch-mean loss. Unlike ``coral_pytorch.losses.corn_loss`` this
    needs no hard integer label, so multi-label windows train on what the driver
    actually marked instead of being rejected or silently thresholded.

    Numerics: ``logsigmoid`` rather than ``log(sigmoid(.))`` — the latter
    underflows to ``log(0) = -inf`` around a logit of -100 in float32. The
    gradient is ``q_{k-1}*sigma(z_k) - q_k``, bounded by 1 in absolute value, so
    this loss cannot itself explode gradients regardless of how extreme the
    logits get.
    """
    if logits.shape[-1] != levels.shape[-1] - 1:
        raise ValueError(
            f"soft_corn_loss expects (B, K-1) logits and (B, K) levels, got "
            f"{tuple(logits.shape)} and {tuple(levels.shape)}")
    # Batch dims must match EXACTLY, not merely broadcast. Checking only the
    # last dim above lets a (1, K) or (K,) label through, and torch then
    # broadcasts it over the whole batch — every segment silently trains
    # against one driver's label, with no error and a perfectly plausible loss
    # value. Only non-broadcastable batch sizes (e.g. 3 vs 8) fail loudly, so
    # the dangerous case is exactly the one that used to pass.
    if logits.shape[:-1] != levels.shape[:-1]:
        raise ValueError(
            f"soft_corn_loss batch dims must match exactly: logits "
            f"{tuple(logits.shape)} vs levels {tuple(levels.shape)}. A (1, K) "
            f"or (K,) label would broadcast over the batch instead of raising.")
    # Build the targets in at least float32 even under autocast, so the PMF and
    # its CDF are not quantized to fp16 before they are used as regression
    # weights. (The degenerate all-zero row this also used to guard against is
    # now rejected outright by levels_to_distribution.)
    levels = levels.to(torch.promote_types(levels.dtype, torch.float32))
    p = levels_to_distribution(levels).to(logits.dtype)         # (B, K)   P(y = k)
    q = levels_to_cumulative(levels).to(logits.dtype)           # (B, K-1) P(y > k)
    z = logits.clamp(-_LOGIT_SATURATION, _LOGIT_SATURATION)
    return -(q * F.logsigmoid(z)
             + p[..., :-1] * F.logsigmoid(-z)).sum(-1).mean()


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
    # 'window_seconds' and 'resample_hz' are DATA contracts (how segments were
    # cut and onto which time grid they were resampled at training), not model
    # kwargs: keep them in the returned arch for callers (fine-tuning, sweep,
    # inference) but don't pass them to the constructor. Checkpoints written
    # before resampling existed simply lack the key, and .get() yields None →
    # resampling stays off for them, which is the old behaviour.
    _DATA_CONTRACT_KEYS = ("window_seconds", "resample_hz")
    ctor_kwargs = {k: v for k, v in arch.items() if k not in _DATA_CONTRACT_KEYS}
    model = XLSTMSequenceClassifier(**ctor_kwargs)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, arch
