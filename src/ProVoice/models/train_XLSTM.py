# Usage: python -m ProVoice.train_XLSTM --in data/with_segments.jsonl --label-map data/labels.csv --out trained_models/state_xlstm.pt
import argparse, csv, hashlib, json, pathlib, random
from typing import List, Dict, Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function
from ProVoice.models.xlstm_model import (
    encode_and_resample,
    DEFAULT_RESAMPLE_HZ,
    D_IN,
    STATE_CAT,
    STATE_NUM,
    STATE_CARLA,
    SENTINEL_VALUES,
    FEATURE_ALIASES,
    XLSTMSequenceClassifier,
    save_checkpoint,
    DEFAULT_CONTEXT_LENGTH,
    FEATURE_NAMES,
    log_encoded_frames,
    logits_to_probs,
    logits_to_label,
    probs_to_label,
    levels_to_distribution,
    levels_to_cumulative,
    soft_corn_loss,
)
from ProVoice.models.xlstm_model import _as01
from ProVoice.decision_engine import truncate_frames_by_seconds


LEVELS = [f"Level_{i}" for i in range(1, 6)]
SPLIT_VARIABLE = "participantid"


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def iter_jsonl(path: pathlib.Path):
    """Stream parsed JSONL objects one at a time. PREFER THIS over read_jsonl.

    ``data/labeled_data.jsonl`` is 931 MB / 508,282 rows of 73 keys, and the raw
    parsed dicts cost **4.0 GB** — measured, not estimated. Materializing them in
    a list before normalizing means the raw list, the normalized list and the
    DataFrame are all alive at once: 4.57 GB peak, of which 4.03 GB is the raw
    list that is discarded moments later.

    Consuming this generator instead lets each raw dict be freed as soon as
    ``normalize_row`` has copied the ~25 fields it keeps, which drops the peak to
    roughly 0.6 GB. That is the difference between one trainer fitting in RAM
    alongside anything else and a machine that swaps — and the study runs 180 of
    these back to back.
    """
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Eager list form. Kept for callers that genuinely need random access;
    for a full-size file prefer :func:`iter_jsonl` — see its docstring for why."""
    return list(iter_jsonl(path))


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    def pick(*keys, default=""):
        for k in keys:
            if k in row and row[k] not in (None, ""): return row[k]
        return default
    out = {}
    # timestamp is needed by the --window-seconds truncation (and harmless otherwise)
    out['timestamp']       = pick('timestamp', 'ts', 'time')
    out['segment_id']      = pick('segment_id', 'segment', 'trial_id', 'trial', 'block_id')
    out['participantid']   = pick('participantid', 'participant_id', 'participant', 'pid')
    out['functionname']    = pick('functionname', 'function', 'func_name', 'FunctionName')
    out['environment']     = pick('environment', 'env', 'environment_type')
    out['secondary_task']  = pick('secondary_task', 'sec_task', 'secondaryTask')
    out['lab']             = pick('lab', 'lab_state')
    out['emotion']         = pick('emotion', 'affect', 'emo', 'mood', 'Emotion')
    out['drowsiness_alert']= pick('drowsiness_alert', 'drowsy', 'fatigue')
    out['gaze_distracted'] = pick('gaze_distracted', 'gaze', 'distraction')
    out['heart_rate']      = pick('heart_rate', 'hr', 'heartrate', 'bpm')
    # CARLA vehicle/world features — use sentinel defaults matching encode_frame expectations.
    # NOTE: this function WHITELISTS keys; anything not listed here never reaches
    # the trainer, and silently arrives at encode_frame as its default. Every
    # name in STATE_NUM / STATE_CARLA must therefore appear below (asserted at
    # the end of this function).
    out['speed_ratio_max']   = pick('speed_ratio_max',   default=None)
    out['brake']             = pick('brake',             default=None)
    out['steer']             = pick('steer',             default=None)
    out['throttle']          = pick('throttle',          default=None)
    out['is_junction']       = pick('is_junction',       default=None)
    # null = "no vehicle ahead within the 100 m detection range", a real state of
    # the world, NOT a missing measurement -- it maps to the reserved marker, not
    # to 0.0 (which would mean a lead vehicle at zero metres). Same treatment
    # encode_frame applies at serving time; SENTINEL_VALUES is the shared source.
    out['lead_distance_m']   = pick('lead_distance_m',
                                    default=SENTINEL_VALUES['lead_distance_m'])
    out['perclos']       = pick('perclos',       default=0.0)
    out['gaze_score']    = pick('gaze_score',    default=0.0)
    out['hr_delta']      = pick('hr_delta',      default=0.0)
    out['rr_delta']      = pick('rr_delta',      default=0.0)
    out['blink_rate']    = pick('blink_rate',    default=0.0)
    out['yawn_rate']     = pick('yawn_rate',     default=0.0)
    # The DataCollector logs EAR under 'eye_ar'; the model feature is named 'ear'.
    # Without the alias this column is silently all-zeros. The alias list is
    # xlstm_model.FEATURE_ALIASES, NOT a literal here: encode_frame applies the
    # same map at serving time, and the two drifting apart is exactly the
    # train/serve skew this used to cause.
    out['ear']           = pick('ear', *FEATURE_ALIASES['ear'], default=0.0)
    out['mar']           = pick('mar',            default=0.0)

    for k in LEVELS:
        if k in row and row[k] not in (None, ""):
            out[k] = int(float(row[k]))
    return out


# A model feature missing from normalize_row's whitelist does not raise -- it
# arrives at encode_frame as a constant default (0.0, or the sentinel), so the
# column is dead weight and training still "succeeds". Checked once at import,
# against the schema itself, so adding a feature to STATE_NUM / STATE_CARLA
# without wiring it up here fails loudly instead of quietly zeroing a channel.
_NORMALIZE_ROW_KEYS = set(normalize_row({}))
assert set(STATE_NUM) | set(STATE_CARLA) <= _NORMALIZE_ROW_KEYS, (
    "normalize_row does not emit these model features: "
    f"{sorted((set(STATE_NUM) | set(STATE_CARLA)) - _NORMALIZE_ROW_KEYS)}")


def load_label_map(path: str | None) -> Dict[str, List[int]]: # NOT USED !!!
    if not path: return {}
    p = pathlib.Path(path)
    if not p.exists(): return {}
    df = pd.read_csv(p)
    miss = [k for k in (["segment_id"] + LEVELS) if k not in df.columns]
    if miss:
        raise ValueError(f"--label-map missing columns: {miss}; required: ['segment_id'] + Level_1..Level_5")
    m = {}
    for _, r in df.iterrows():
        sid = str(r['segment_id']).strip()
        if not sid: continue
        vec = [int(float(r[k])) for k in LEVELS]
        vec = [1 if v >= 1 else 0 for v in vec]
        m[sid] = vec
    return m


# --------------------------------------------------------------------------- #
# Encoded-segment cache.
#
# Every run re-parses the same 971 MB JSONL to produce the same ~1,470 encoded
# segments: 24 s of single-threaded CPU per subprocess, and the pipeline launches
# 420 of them. The encoding depends only on (source file, window_seconds,
# resample_hz) — NOT on the train/val split, the seed, dropout or lr — so the
# whole sweep can read one cache. At (100, 33) float32 per segment that is ~19 MB
# against 971 MB, which is also what makes running many trainers concurrently
# affordable: per-process peak drops from ~0.6 GB to a few tens of MB.
#
# THE CACHE IS KEYED BY A FINGERPRINT, NOT BY A FILENAME. `D_IN` went 35 -> 33 on
# 2026-08-14 and invalidated every checkpoint; a cache keyed by name alone would
# reintroduce exactly that failure, silently, as training data. `load_segment_cache`
# refuses anything whose source file, window, grid or feature contract has moved.
# --------------------------------------------------------------------------- #
SEGMENT_CACHE_VERSION = 1


def cache_name(window_seconds: float, resample_hz: float) -> str:
    """Canonical filename for a cache. Lives here, next to the format, so the
    builder and every consumer derive it from one definition."""
    return f"segments_w{window_seconds:g}_hz{resample_hz:g}.npz"


def _file_fingerprint(path: str | pathlib.Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = pathlib.Path(path)
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime)}


def cache_meta(in_jsonl: str, label_map: str | None,
               window_seconds: float | None, resample_hz: float | None) -> Dict[str, Any]:
    """The identity of a cache: everything the encoding depends on.

    `features_sha` covers the feature CONTRACT (names and order), so a schema
    change invalidates the cache instead of feeding a stale 35-dim encoding to a
    33-dim model — the failure that deleted the last population checkpoint.
    """
    return {
        "version": SEGMENT_CACHE_VERSION,
        "source": _file_fingerprint(in_jsonl),
        "label_map": _file_fingerprint(label_map),
        "window_seconds": float(window_seconds) if window_seconds else 0.0,
        "resample_hz": float(resample_hz) if resample_hz else 0.0,
        "d_in": int(D_IN),
        "features_sha": hashlib.sha256(",".join(FEATURE_NAMES).encode()).hexdigest()[:16],
    }


def save_segment_cache(path: pathlib.Path, groups: List[Tuple[np.ndarray, np.ndarray]],
                       segment_ids: List[str], participant_ids: List[str],
                       pid_order: List[str], seg_order: List[str],
                       meta: Dict[str, Any]) -> None:
    """Write the encoded segments.

    Segments are stored CONCATENATED with offsets rather than padded to a common
    length: `make_collate` derives each item's `lengths` entry from its own T, so
    padding here would either have to be undone exactly or would silently extend
    every sequence to the longest one in the file.
    """
    x_flat = (np.concatenate([g[0] for g in groups], axis=0) if groups
              else np.zeros((0, D_IN), dtype=np.float32))
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    if groups:
        offsets[1:] = np.cumsum([g[0].shape[0] for g in groups])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_flat=x_flat.astype(np.float32),
        offsets=offsets,
        levels=(np.stack([g[1] for g in groups], axis=0) if groups
                else np.zeros((0, len(LEVELS)), dtype=np.float32)).astype(np.float32),
        segment_id=np.asarray(segment_ids, dtype=np.str_),
        participantid=np.asarray(participant_ids, dtype=np.str_),
        # First-appearance order in the SOURCE file, not the sorted groupby order.
        # The seeded 80/20 participant split shuffles this list, so storing the
        # sorted order instead would give cached and uncached runs different
        # splits at the same --seed.
        pid_order=np.asarray(pid_order, dtype=np.str_),
        seg_order=np.asarray(seg_order, dtype=np.str_),
        meta=np.asarray(json.dumps(meta), dtype=np.str_),
    )


def load_segment_cache(path: str | pathlib.Path, expect: Dict[str, Any],
                       strict: bool = True) -> Optional[Dict[str, Any]]:
    """Load a cache, or return None if it is absent or stale.

    `strict` raises on a mismatch instead of returning None. The sweep wants that:
    silently falling back to a 24 s re-parse across 420 runs would look like the
    cache is working while delivering none of the speedup.
    """
    p = pathlib.Path(path)
    if not p.exists():
        if strict:
            raise SystemExit(f"--cache {p} does not exist. Build it with:\n"
                             f"    python -m scripts.build_segment_cache --in <jsonl> "
                             f"--window-seconds {expect.get('window_seconds')} "
                             f"--out {p}")
        return None
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        diff = {k: (meta.get(k), expect[k]) for k in expect if meta.get(k) != expect[k]}
        if diff:
            msg = (f"cache {p.name} is STALE — rebuild it. Mismatched: "
                   + "; ".join(f"{k}: cache={c!r} run={r!r}" for k, (c, r) in diff.items()))
            if strict:
                raise SystemExit(msg)
            print(f"[cache][warn] {msg}")
            return None
        return {
            "x_flat": z["x_flat"], "offsets": z["offsets"], "levels": z["levels"],
            "segment_id": [str(s) for s in z["segment_id"]],
            "participantid": [str(s) for s in z["participantid"]],
            "pid_order": [str(s) for s in z["pid_order"]],
            "seg_order": [str(s) for s in z["seg_order"]],
            "meta": meta,
        }


class SeqDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        split: str = "train",
        log_fh=None,
        window_seconds: float | None = None,
        resample_hz: float | None = None,
    ):
        assert 'segment_id' in df.columns and df['segment_id'].astype(bool).any(), "segment_id is required"
        self.context_length = context_length
        self.groups: List[Tuple[np.ndarray, np.ndarray]] = []
        # Parallel to `groups`, one participantid per segment. Needed by the
        # PER-DRIVER constant floor, which is the binding baseline under
        # --split-mode within-driver: there every driver is in both halves, so
        # "always predict this driver's own favourite level" is available for
        # free and a model has to beat THAT, not the global constant.
        self.pids: List[str] = []
        skipped = []
        for gid, g in df.groupby('segment_id'):
            g = g.reset_index(drop=True)
            if not all(k in g.columns for k in LEVELS):
                continue
            level_vec = g[LEVELS].iloc[0].astype(float).values
            if np.isnan(level_vec).any() or level_vec.sum() <= 0:
                skipped.append(gid)
                continue
            # The multi-hot mark vector is the ONLY label representation. Both
            # losses consume it directly and every metric is set-aware, so there
            # is nothing left that needs a collapsed integer — and no argmax to
            # silently pick the driver's lowest acceptable level.
            lvl = (level_vec > 0).astype(np.float32)
            rows = g.to_dict("records")
            # Keep only the LAST window_seconds of the segment (frames are
            # chronological within a segment). None/0 = use the full segment.
            rows = truncate_frames_by_seconds(rows, window_seconds)
            # Then put every segment on the same time grid, so T depends on the
            # segment's DURATION and not on the rate the session happened to
            # achieve. None/0 = encode the raw frames (pre-resampling behaviour).
            X = encode_and_resample(rows, resample_hz, window_seconds)
            self.groups.append((X, lvl))
            self.pids.append(str(g[SPLIT_VARIABLE].iloc[0]) if SPLIT_VARIABLE in g.columns else "")
            if log_fh is not None:
                log_encoded_frames(log_fh, split, str(gid), X, levels=lvl)
        if skipped:
            print(f"[warn] skipped {len(skipped)} segment(s) with missing/empty Level_* labels: "
                f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")


    @classmethod
    def from_cache(cls, cache: Dict[str, Any], context_length: int,
                   pids: Optional[set] = None,
                   segment_ids: Optional[set] = None) -> "SeqDataset":
        """Build from a pre-encoded cache instead of a DataFrame.

        Bypasses __init__ deliberately: the encoding it would redo is exactly what
        the cache holds, and `groups` is the only attribute anything downstream
        reads (`make_collate`, the model, both losses and every metric take items,
        never the frame). Selection is by participant (the normal split) or by
        segment id (the single-participant fallback).
        """
        obj = cls.__new__(cls)
        obj.context_length = context_length
        obj.groups = []
        obj.pids = []
        x_flat, off, lv = cache["x_flat"], cache["offsets"], cache["levels"]
        for i, (sid, pid) in enumerate(zip(cache["segment_id"], cache["participantid"])):
            if pids is not None and str(pid) not in pids:
                continue
            if segment_ids is not None and str(sid) not in segment_ids:
                continue
            obj.groups.append((x_flat[off[i]:off[i + 1]], lv[i]))
            obj.pids.append(str(pid))
        return obj

    def __len__(self): return len(self.groups)
    def __getitem__(self, i): return self.groups[i]


def within_driver_temporal_split(seg_order: Sequence[str], seg_pids: Sequence[str],
                                 val_frac: float) -> Tuple[set, set]:
    """Per driver: earliest ``1-val_frac`` of their segments train, latest tail val.

    THIS IS A SUBJECT-DEPENDENT SPLIT and answers a different question from the
    participant split. Every driver appears in BOTH halves, so a model may
    legitimately exploit driver identity — which on this cohort is worth far more
    MAE than the task itself (``constant_baseline``: knowing the driver buys
    0.23, knowing the function 0.06) and is ~68 % recoverable from the state
    features alone. Do NOT select the deployed configuration on it: the
    population model is only ever served to drivers absent from its training set,
    and ``sweep_population_hparams`` is the estimator for that.

    What it IS for: telling apart "the features carry no LoA signal" from "the
    signal is real but driver-specific and does not transfer". The cross-driver
    sweep cannot separate those — both produce a model that loses to a constant —
    and the answer decides whether the personalization arms are worth running.

    Ordering is FIRST-APPEARANCE order in the source file, not a sort of
    ``segment_id``: ids are ``<session_uuid>|win<idx>p<prompt>``, so sorting them
    orders a driver's two sessions by UUID rather than by time. File order is
    chronological within each driver (verified against the session start times in
    ``user_loa_labels.csv``), which is what makes the tail a genuine FUTURE tail.

    No ``seed`` argument, deliberately: the split is a deterministic function of
    the data, so seeds vary only initialization, dropout and batch order. Every
    seed of every configuration sees the identical split.
    """
    if len(seg_order) != len(seg_pids):
        raise ValueError(f"seg_order ({len(seg_order)}) and seg_pids ({len(seg_pids)}) "
                         "must be parallel; the caller built them from different sources.")
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"--val-frac must be in (0, 1), got {val_frac}")
    by_pid: Dict[str, List[str]] = {}
    for sid, pid in zip(seg_order, seg_pids):
        by_pid.setdefault(str(pid), []).append(str(sid))

    tr_segs, te_segs = set(), set()
    thin = []
    for pid, segs in by_pid.items():
        n = len(segs)
        # Both halves must be non-empty or the driver silently leaves one side of
        # the split, which is exactly the kind of per-driver imbalance that makes
        # a pooled metric mean something other than it appears to.
        n_val = min(max(1, int(round(val_frac * n))), n - 1) if n >= 2 else 0
        if n_val == 0:
            thin.append((pid, n))
            tr_segs.update(segs)
            continue
        tr_segs.update(segs[:n - n_val])
        te_segs.update(segs[n - n_val:])
    if thin:
        print(f"[split][warn] driver(s) with <2 segments got no validation tail: {thin}")
    print(f"[split] WITHIN-DRIVER temporal: {len(by_pid)} driver(s), "
          f"first {1 - val_frac:.0%} of each driver's segments train "
          f"({len(tr_segs)}), last {val_frac:.0%} val ({len(te_segs)}). "
          f"SUBJECT-DEPENDENT — diagnostic only, not a deployment estimate.")
    return tr_segs, te_segs


def choose_split(pid_order: List[str], seg_order: List[str], no_val: bool,
                 val_pids_arg: str, seed: int,
                 split_mode: str = "participant", val_frac: float = 0.2,
                 seg_pids: Optional[Sequence[str]] = None,
                 ) -> Tuple[Optional[set], Optional[set],
                            Optional[set], Optional[set]]:
    """Decide the train/val split. Returns (tr_pids, te_pids, tr_segs, te_segs).

    Exactly one pair is non-None: participant-level normally, segment-level in
    the single-participant fallback and under ``--split-mode within-driver``.
    Shared by the JSONL and cache paths so a cached run and an uncached one at
    the same --seed cannot land on different splits — which is why the cache
    stores FIRST-APPEARANCE order rather than the sorted groupby order it would
    otherwise be natural to write. That same ordering is what
    ``within_driver_temporal_split`` relies on to build a temporal tail.

    ``split_mode`` defaults to the participant behaviour this function has always
    had, so every existing caller and command line is unaffected.
    """
    if not pid_order:
        raise ValueError(f"Split variable '{SPLIT_VARIABLE}' is missing from all rows.")
    if split_mode not in ("participant", "within-driver"):
        raise ValueError(f"unknown --split-mode {split_mode!r}")
    if split_mode == "within-driver":
        # --val-pids means "hold out THESE drivers", which within-driver
        # contradicts by construction. Failing is better than silently honouring
        # one and ignoring the other.
        if val_pids_arg:
            raise ValueError("--val-pids is a cross-driver hold-out and cannot be combined "
                             "with --split-mode within-driver.")
        if no_val:
            print(f"[split] --no-val: training on ALL {len(pid_order)} participant(s), "
                  f"no validation set, no epoch selection")
            return set(str(p) for p in pid_order), set(), None, None
        if seg_pids is None:
            raise ValueError("--split-mode within-driver needs per-segment participant ids; "
                             "the caller did not supply seg_pids.")
        tr_segs, te_segs = within_driver_temporal_split(seg_order, seg_pids, val_frac)
        return None, None, tr_segs, te_segs
    if no_val:
        print(f"[split] --no-val: training on ALL {len(pid_order)} participant(s), "
              f"no validation set, no epoch selection")
        return set(str(p) for p in pid_order), set(), None, None

    have = set(str(p) for p in pid_order)
    if val_pids_arg:
        want = [p.strip() for p in val_pids_arg.split(",") if p.strip()]
        missing = [p for p in want if p not in have]
        if missing:
            raise ValueError(
                f"--val-pids names participant(s) not present in the data: {missing}. "
                f"Available: {sorted(have)}")
        te_pids = set(want)
        tr_pids = have - te_pids
        if not tr_pids:
            raise ValueError("--val-pids holds out every participant; nothing left to train on.")
        print(f"[split] EXPLICIT val participants={sorted(te_pids)}  "
              f"train participants={sorted(tr_pids)}")
        return tr_pids, te_pids, None, None

    rng = pd.Series(list(pid_order)).sample(frac=1.0, random_state=seed).values
    if len(rng) >= 2:
        ntr = max(1, int(0.8 * len(rng)))
        tr_pids, te_pids = set(str(p) for p in rng[:ntr]), set(str(p) for p in rng[ntr:])
        print(f"[split] train participants={sorted(tr_pids)}  val participants={sorted(te_pids)}")
        return tr_pids, te_pids, None, None

    print(f"[split] only {len(rng)} participant(s) — falling back to segment-level 80/20 split")
    gids = pd.Series(list(seg_order)).sample(frac=1.0, random_state=seed).values
    ntr = max(1, int(0.8 * len(gids)))
    tr_segs, te_segs = set(str(g) for g in gids[:ntr]), set(str(g) for g in gids[ntr:])
    print(f"[split] train segments={len(tr_segs)}  val segments={len(te_segs)}")
    return None, None, tr_segs, te_segs


def make_collate(context_length: int):
    """Collate ``(X, levels)`` items into ``(frames, lengths, levels)`` batches.

    Items carry the multi-hot mark vector and nothing else — there is no
    collapsed integer label anywhere in the pipeline, so a caller cannot
    accidentally reintroduce one by unpacking the wrong element.
    """
    def collate(batch):
        if len(batch) == 0:
            return (torch.empty(0, context_length, D_IN),
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, len(LEVELS)))
        xs, ls, lvls = [], [], []
        for X, lvl in batch:
            T = X.shape[0]
            if T > context_length:
                X = X[-context_length:]
            pad = context_length - X.shape[0]
            if pad > 0:
                # RIGHT-pad with zero vectors. forward() reads the hidden state
                # at index length-1 (the last real frame); the stack is causal,
                # so the pad frames after it have exactly zero influence.
                X = np.concatenate([X, np.zeros((pad, X.shape[1]), dtype=X.dtype)], axis=0)
            xs.append(torch.from_numpy(X))
            ls.append(min(T, context_length))
            lvls.append(torch.from_numpy(np.asarray(lvl, dtype=np.float32)))
        return (torch.stack(xs, 0),
                torch.tensor(ls, dtype=torch.long),
                torch.stack(lvls, 0))
    return collate


# --------------------------------------------------------------------------- #
# Metrics.
#
# A window's label is the SET of LoAs the driver marked acceptable (~a third of
# real windows mark more than one, and a third of THOSE are non-contiguous, e.g.
# {L1, L5}). Every metric below therefore takes the multi-hot `levels` vector,
# never a collapsed integer.
#
# There used to be an `int(np.argmax(level_vec))` pseudo-label threaded through
# the datasets and into these metrics. np.argmax returns the FIRST maximal index,
# so on a multi-hot vector it always resolved to the driver's LOWEST acceptable
# level — a systematic downward bias in the reference used for model selection,
# meta-validation early stopping, and the published learning curves. The
# `resolve_targets` below replaces it.
# --------------------------------------------------------------------------- #
def resolve_targets(levels: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-row effective ground truth: the marked level the prediction is judged against.

    Every metric here compares two point values, but the label is a set. This
    resolves the set to the single marked level CLOSEST to the prediction — the
    level the driver would plausibly have named had they been forced to name
    one — so a prediction inside the set is exactly right, and one outside it is
    scored against the nearest acceptable alternative rather than against an
    arbitrary member.

    Properties that make this safe to build every metric on:

    * **Exact reduction.** On a single-label row the only marked level is
      returned regardless of the prediction, so every metric below collapses to
      its ordinary single-label form. Existing single-label results are
      numerically unchanged.
    * **Deterministic ties.** A prediction equidistant from two marked levels
      (e.g. marks {1, 3}, prediction 2) resolves to the LOWER one, via argmin's
      first-match rule. Arbitrary, but fixed — never data-dependent.
    * **Empty rows are inert.** A row with no marked level returns the
      prediction itself, contributing zero error instead of an invented label.
      Callers upstream already reject such rows; this is belt-and-braces.

    CAVEAT, state it when reporting: the resolved target depends on the
    prediction, so these are best-match (oracle-favourable) scores. That is the
    right convention for a set-valued label scored against a point prediction,
    but it means the chance-corrected metric (QWK) is an UPPER BOUND on
    multi-label rows — its "true" marginal shifts with the model. Single-label
    rows are unaffected.
    """
    y_pred = np.asarray(y_pred)
    if len(y_pred) == 0:
        return y_pred.astype(int)
    levels = np.asarray(levels)
    idx = np.arange(levels.shape[1])
    out = y_pred.astype(int).copy()
    for i, (row, p) in enumerate(zip(levels, y_pred)):
        marked = idx[row.astype(bool)]
        if marked.size:
            out[i] = int(marked[np.abs(marked - int(p)).argmin()])
    return out


# --- single-label primitives -------------------------------------------------
# Kept as the building blocks the set-aware metrics delegate to. Call these
# DIRECTLY only on data you know is single-label; otherwise use the set_* forms.
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0: return 0.0
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """Macro-F1 averaged over LoA levels WITH SUPPORT (matches sklearn's
    ``f1_score(..., labels=present, average='macro', zero_division=0)``).

    A level absent from both the labels and the predictions is skipped, not
    scored 0. Scoring it 0 capped the achievable value at (levels present)/5 —
    0.6 on a per-driver validation tail covering 3 of the 5 levels — which read
    as a failure to personalize when the model was in fact perfect.

    Consequence to keep in mind: the denominator now varies with the tail, so
    values are not comparable across tails with different level coverage.
    """
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1s.append(2.0 * tp / denom)
    # No level had support (empty input, or labels outside [0, n_classes)):
    # np.mean([]) is nan, which would propagate silently into the metrics CSV.
    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in LoA levels — ordinal metric: off-by-1 < off-by-4."""
    if len(y_true) == 0: return 0.0
    return float(np.abs(y_true.astype(float) - y_pred.astype(float)).mean())


def qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """Quadratic weighted kappa — chance-corrected ordinal agreement in [-1, 1].

    Undefined (0/0) when labels and predictions are both constant, e.g. a
    single-segment validation tail; defined here as 1.0 on exact agreement
    and 0.0 otherwise, so degenerate tails don't produce NaNs in logs/CSVs.
    """
    if len(y_true) == 0: return 0.0
    from sklearn.metrics import cohen_kappa_score
    k = cohen_kappa_score(y_true, y_pred, labels=list(range(n_classes)), weights="quadratic")
    if np.isfinite(k):
        return float(k)
    return 1.0 if np.array_equal(y_true, y_pred) else 0.0


# --- set-aware metrics (THE ones to report and select on) --------------------
def set_accuracy(levels: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions the driver marked as acceptable.

    Reduces exactly to :func:`accuracy` when every row marks one level.
    """
    if len(y_pred) == 0: return 0.0
    return float((resolve_targets(levels, y_pred) == np.asarray(y_pred)).mean())


def set_mae(levels: np.ndarray, y_pred: np.ndarray) -> float:
    """Distance to the NEAREST marked level; 0 when the prediction is accepted.

    Generalises :func:`mae` to multi-label rows without punishing a model for
    picking one acceptable level over another. Reduces exactly to ``mae`` when
    every row marks a single level. **This is the model-selection metric.**
    """
    if len(y_pred) == 0: return 0.0
    return mae(resolve_targets(levels, y_pred), np.asarray(y_pred))


def set_macro_f1(levels: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """Macro-F1 against the nearest marked level. Reduces exactly to :func:`macro_f1`.

    Averages only over LoA levels with support (see :func:`macro_f1`), so a
    validation tail covering 3 of the 5 levels can still reach 1.0. Scores are
    therefore NOT comparable across tails that cover different numbers of
    levels — report the level coverage alongside, or prefer set-MAE, which has
    no such dependence.
    """
    if len(y_pred) == 0: return 0.0
    return macro_f1(resolve_targets(levels, y_pred), np.asarray(y_pred), n_classes)


def set_qwk(levels: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """QWK against the nearest marked level. Reduces exactly to :func:`qwk`.

    Upper bound on multi-label rows — see the caveat in :func:`resolve_targets`.
    """
    if len(y_pred) == 0: return 0.0
    return qwk(resolve_targets(levels, y_pred), np.asarray(y_pred), n_classes)


def constant_baseline(levels: np.ndarray, n_classes: int = 5) -> Dict[str, float]:
    """Best set-MAE and set-accuracy reachable by predicting ONE fixed LoA.

    **The reference any model has to beat**, and the one this pipeline was
    missing. Measured on the collected cohort, the best global constant (LoA 1)
    scores set-MAE 1.321 — so a population model reporting 1.4-1.5 is losing to
    "always say 1", which is a fact about the task and not about the run. Without
    this line printed next to it, a validation curve whose minimum is at epoch 0
    looks like a mystery instead of a diagnosis.

    Why LoA preference is this hard to predict across drivers: knowing the
    FUNCTION is worth 0.06 MAE (1.321 -> 1.259 with an oracle per-function
    constant), while knowing the DRIVER is worth 0.23 (-> 1.090). Drivers 004
    and 008 never mark LoA 3-4; 005 and 011 mark them ~65 % of the time. A
    population model must sit between those, which is wrong for both.

    The MAE-optimal and accuracy-optimal constants need NOT be the same level
    (MAE punishes distance, accuracy does not), so each metric gets its own
    optimum rather than one level being forced to serve both.
    """
    empty = {"const_loa_mae": -1, "const_set_mae": float("nan"),
             "const_loa_acc": -1, "const_set_acc": float("nan")}
    levels = np.asarray(levels)
    if levels.size == 0:
        return empty
    n = levels.shape[0]
    maes = [set_mae(levels, np.full(n, c, dtype=int)) for c in range(n_classes)]
    accs = [set_accuracy(levels, np.full(n, c, dtype=int)) for c in range(n_classes)]
    c_mae, c_acc = int(np.argmin(maes)), int(np.argmax(accs))
    return {"const_loa_mae": c_mae, "const_set_mae": float(maes[c_mae]),
            "const_loa_acc": c_acc, "const_set_acc": float(accs[c_acc])}


def per_driver_constant_baseline(train_levels: np.ndarray, train_pids: Sequence[str],
                                 val_levels: np.ndarray, val_pids: Sequence[str],
                                 n_classes: int = 5) -> Dict[str, float]:
    """The floor under a SUBJECT-DEPENDENT split: "always predict this driver's
    own favourite level", where the favourite is fitted on that driver's TRAIN
    prefix and scored on their val tail.

    ``constant_baseline`` — one level for the whole cohort — is the right floor
    when the validation drivers are unseen, because their favourite level is not
    knowable. Under ``--split-mode within-driver`` it is far too weak: the driver
    IS in the training set, so a per-driver constant is available for free, needs
    no features at all, and already buys ~0.23 MAE on this cohort. A model that
    beats the global constant but not this one has learned WHO is driving and
    nothing about WHEN they want autonomy — which is precisely the confusion the
    within-driver split exists to expose, so it must be reported next to it.

    Fitted on train and applied to val, NOT fitted on val: the latter is an
    oracle no deployable rule could match. It is returned too, as
    ``pdconst_oracle_set_mae``, because the gap between the two is how much of a
    driver's preference is stable across their own session — a quantity the
    personalization arms are ultimately betting on.
    """
    empty = {"pdconst_set_mae": float("nan"), "pdconst_set_acc": float("nan"),
             "pdconst_oracle_set_mae": float("nan"), "pdconst_n_drivers": 0}
    val_levels = np.asarray(val_levels)
    train_levels = np.asarray(train_levels)
    if val_levels.size == 0 or train_levels.size == 0:
        return empty
    train_pids = np.asarray([str(p) for p in train_pids])
    val_pids = np.asarray([str(p) for p in val_pids])
    if len(train_pids) != len(train_levels) or len(val_pids) != len(val_levels):
        raise ValueError("levels and pids must be parallel")

    def best_const(lv: np.ndarray) -> int:
        n = lv.shape[0]
        return int(np.argmin([set_mae(lv, np.full(n, c, dtype=int)) for c in range(n_classes)]))

    # Fallback for a driver with no train rows: the global train constant, which
    # is what a deployed system would have for a driver it has never adapted to.
    global_c = best_const(train_levels)
    pred = np.empty(len(val_levels), dtype=int)
    pred_oracle = np.empty(len(val_levels), dtype=int)
    for pid in np.unique(val_pids):
        m_val = val_pids == pid
        m_trn = train_pids == pid
        pred[m_val] = best_const(train_levels[m_trn]) if m_trn.any() else global_c
        pred_oracle[m_val] = best_const(val_levels[m_val])
    return {"pdconst_set_mae": float(set_mae(val_levels, pred)),
            "pdconst_set_acc": float(set_accuracy(val_levels, pred)),
            "pdconst_oracle_set_mae": float(set_mae(val_levels, pred_oracle)),
            "pdconst_n_drivers": int(len(np.unique(val_pids)))}


def prepare_frame(df: pd.DataFrame, label_map: str | None) -> pd.DataFrame:
    """Apply the label map and coerce every model-input column to its own dtype.

    Shared by the trainer and scripts/build_segment_cache.py, so a cache cannot
    be encoded from differently-coerced columns than a direct run would use.
    """
    if label_map:
        lm = pd.read_csv(label_map)
        miss = [k for k in (["segment_id"] + LEVELS) if k not in lm.columns]
        if miss:
            raise ValueError(f"--label-map missing columns: {miss}")
        df = df.merge(lm, on="segment_id", how="left", suffixes=("", "_map"))
        for k in LEVELS:
            if k not in df.columns or df[k].isna().all():
                df[k] = df.get(k + "_map")
            df[k] = df[k].fillna(0).astype(int)
            if k + "_map" in df.columns: df.drop(columns=[k + "_map"], inplace=True)

    if 'segment_id' not in df.columns or df['segment_id'].eq("").all():
        raise ValueError("Missing segment_id; cannot build sequences.")
    for k in STATE_CAT:
        if k not in df.columns: df[k] = ""
        df[k] = df[k].fillna("").astype(str)
    for k in STATE_NUM:
        if k not in df.columns: df[k] = 0.0
        df[k] = df[k].apply(_as01)
    for k in STATE_CARLA:
        # SENTINEL_VALUES is the single source of truth for which columns have a
        # reserved "missing" marker; encode_frame applies the same mapping at
        # serving time. Hard-coding the name here would let the two drift.
        default = SENTINEL_VALUES.get(k, 0.0)
        if k not in df.columns: df[k] = default
        df[k] = df[k].fillna(default)
    return df


def datasets_from_jsonl(args, resample_hz) -> Tuple[Any, Any]:
    """Parse, encode and split the source JSONL — the original path.

    ~24 s of single-threaded CPU on the 971 MB file. `--cache` skips it; see
    datasets_from_cache and scripts/build_segment_cache.py.
    """
    rows = [normalize_row(r) for r in iter_jsonl(pathlib.Path(args.in_jsonl))]
    if not rows:
        raise ValueError("JSONL is empty or contains no valid rows.")
    df = prepare_frame(pd.DataFrame(rows), args.label_map)
    if df[SPLIT_VARIABLE].eq("").all():
        raise ValueError(f"Split variable '{SPLIT_VARIABLE}' is missing from all rows.")

    # First-appearance order, which is what choose_split shuffles. The cache
    # stores the same order so the two paths split identically at a given --seed.
    # segment_id and participantid come from ONE de-duplication so they stay
    # parallel — within_driver_temporal_split reads them as a pair and asserts it.
    seg_first = df.drop_duplicates('segment_id')
    tr_pids, te_pids, tr_segs, te_segs = choose_split(
        [str(p) for p in df[SPLIT_VARIABLE].drop_duplicates()],
        [str(g) for g in seg_first['segment_id']],
        args.no_val, args.val_pids, args.seed,
        split_mode=args.split_mode, val_frac=args.val_frac,
        seg_pids=[str(p) for p in seg_first[SPLIT_VARIABLE]])

    def sel(pids, segs):
        if pids is not None:
            return df[df[SPLIT_VARIABLE].astype(str).isin(pids)].reset_index(drop=True)
        return df[df['segment_id'].astype(str).isin(segs)].reset_index(drop=True)

    tr_df = sel(tr_pids, tr_segs)
    te_df = df.iloc[0:0] if args.no_val else sel(te_pids, te_segs)

    log_fh = open(args.log_path, "w", encoding="utf-8") if args.log_path else None
    if log_fh:
        print(f"[log] writing feature log → {args.log_path}")
    try:
        train_ds = SeqDataset(tr_df, context_length=args.context_length, split="train",
                              log_fh=log_fh, window_seconds=args.window_seconds,
                              resample_hz=resample_hz)
        # Under --no-val there is no validation frame to build a dataset from, and
        # SeqDataset rightly refuses an empty one (its segment_id assert). Skip the
        # construction rather than weakening that guard, which exists to catch a
        # genuinely empty split in the normal path.
        test_ds = ([] if args.no_val else
                   SeqDataset(te_df, context_length=args.context_length, split="val",
                              log_fh=log_fh, window_seconds=args.window_seconds,
                              resample_hz=resample_hz))
    finally:
        if log_fh:
            log_fh.close()
    return train_ds, test_ds


def datasets_from_cache(args, resample_hz) -> Tuple[Any, Any]:
    """Build the split from pre-encoded segments — skips the 24 s parse entirely.

    The cache is fingerprinted against (source file, label map, window, grid,
    feature contract), so a mismatch is a hard error rather than a silent
    fallback: a cache that quietly re-parsed would look like it was working while
    delivering none of the speedup, and one that quietly loaded a stale encoding
    would be the D_IN 35->33 failure again, this time as training data.
    """
    cache = load_segment_cache(
        args.cache, cache_meta(args.in_jsonl, args.label_map,
                               args.window_seconds, resample_hz))
    if args.log_path:
        print("[log][warn] --log is ignored with --cache: the per-frame feature log is "
              "written during encoding, which the cache skips. Re-run without --cache.")
    # seg_order is FIRST-APPEARANCE order; cache["segment_id"] is the sorted
    # groupby order the encoder wrote. They are not the same list, so the pid
    # lookup goes through a dict rather than positional zip.
    pid_by_seg = {str(sid): str(pid) for sid, pid
                  in zip(cache["segment_id"], cache["participantid"])}
    tr_pids, te_pids, tr_segs, te_segs = choose_split(
        cache["pid_order"], cache["seg_order"], args.no_val, args.val_pids, args.seed,
        split_mode=args.split_mode, val_frac=args.val_frac,
        seg_pids=[pid_by_seg.get(str(sid), "") for sid in cache["seg_order"]])
    train_ds = SeqDataset.from_cache(cache, args.context_length,
                                     pids=tr_pids, segment_ids=tr_segs)
    test_ds = ([] if args.no_val else
               SeqDataset.from_cache(cache, args.context_length,
                                     pids=te_pids, segment_ids=te_segs))
    print(f"[cache] {pathlib.Path(args.cache).name}: {len(train_ds)} train + "
          f"{len(test_ds)} val segments")
    return train_ds, test_ds


def main():
    ap = argparse.ArgumentParser(description="Train official xLSTM (single-label 5-class).")
    ap.add_argument("--in",        dest="in_jsonl", required=True)
    ap.add_argument("--out",       dest="out_pt",   default="trained_models/state_xlstm.pt")
    ap.add_argument("--log",       dest="log_path", default="",
                    help="Optional path for a JSONL log of the exact features fed to the "
                         "xLSTM (one line per frame). Off by default — it writes one JSON "
                         "object per frame (~thousands of large lines on a full dataset). "
                         "Pass a path to enable for debugging.")
    ap.add_argument("--label-map", dest="label_map", default=None, help="CSV with columns: segment_id, Level_1..Level_5")
    ap.add_argument("--cache", default="",
                    help="Pre-encoded segment cache (.npz) from scripts/build_segment_cache.py. "
                         "Skips the ~24 s parse+encode of the source JSONL, which is otherwise "
                         "paid once per process — 420 times across the population pipeline — "
                         "and drops per-process peak RSS from ~0.6 GB to a few tens of MB, "
                         "which is what makes --jobs > 1 in the sweep affordable. The cache is "
                         "fingerprinted against the source file, label map, window_seconds, "
                         "resample_hz and the feature contract; a mismatch is a hard error, "
                         "never a silent re-parse. Encoding does not depend on the split, seed, "
                         "dropout or lr, so one cache per (window, grid) serves the whole sweep.")
    ap.add_argument("--val-pids", dest="val_pids", default="",
                    help="Comma-separated participantids to use as the validation set, e.g. "
                         "'001,005'. Overrides the default seeded 80/20 participant split, "
                         "which cannot express a FIXED fold — needed by "
                         "ProVoice.training_scripts.sweep_population_hparams, where the same "
                         "6 folds must be reused across every config and seed or the "
                         "comparison is confounded by which drivers each config was "
                         "validated on. Ignored when --no-val is set.")
    ap.add_argument("--split-mode", dest="split_mode",
                    choices=["participant", "within-driver"], default="participant",
                    help="WHICH GENERALIZATION QUESTION the validation set asks. "
                         "'participant' (default, unchanged): held-out DRIVERS — the "
                         "deployment condition, since the population model is only ever "
                         "served to drivers absent from its training set. This is what "
                         "sweep_population_hparams uses and what the configuration, window "
                         "and loss must be SELECTED on. "
                         "'within-driver': every driver appears in both halves, split "
                         "temporally at --val-frac (earliest segments train, latest tail "
                         "val). SUBJECT-DEPENDENT and deliberately leaky — driver identity "
                         "is ~68 % recoverable from the state features and is worth ~4x "
                         "more MAE than the task, so a model here may score well by "
                         "learning who is driving. Its purpose is DIAGNOSTIC: it separates "
                         "'the features carry no LoA signal' from 'the signal is real but "
                         "does not transfer across drivers', which the cross-driver sweep "
                         "cannot do because both look identical there. Never select a "
                         "shipped configuration on it.")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.2,
                    help="Validation tail fraction per driver under --split-mode "
                         "within-driver. Ignored in 'participant' mode, where the "
                         "validation set is a set of DRIVERS, not a fraction of each "
                         "driver's timeline.")
    ap.add_argument("--no-val", dest="no_val", action="store_true",
                    help="Train on EVERY participant in the input with no validation set, no "
                         "per-epoch evaluation and no best-epoch selection; the final-epoch "
                         "weights are saved. This is the LODO population-training mode: the "
                         "held-out driver is absent from the input file entirely and is scored "
                         "afterwards by the caller, so no signal from it can reach the "
                         "checkpoint. Requires a pre-chosen --epochs (E*); --patience is "
                         "ignored because there is nothing to be patient about.")
    ap.add_argument("--min-select-epoch", dest="min_select_epoch", type=int, default=3,
                    help="Earliest epoch that may be CHECKPOINTED or counted toward "
                         "--patience. Earlier epochs still train and are still logged, so "
                         "the full curve survives for diagnosis; they just cannot win. "
                         "Set >= --warmup-epochs: epochs inside the warmup run at a "
                         "reduced LR, so selecting among them selects a schedule artifact. "
                         "The floor also blocks a silent failure — on this cohort the "
                         "unconstrained argmin often lands on epoch 0, and E*=0 makes the "
                         "LODO runner train for zero epochs and ship 12 random inits as "
                         "population models. 0 disables.")
    ap.add_argument("--decode", choices=["canonical", "argmax", "median"], default="canonical",
                    help="Which rule turns the decoded PMF into a single LoA for the "
                         "REPORTED and SELECTED-ON metrics. 'canonical' (default) uses each "
                         "head's own rule — argmax for softmax, Shi et al.'s rank rule "
                         "sum_k 1[q_k>0.5] for CORN.\n"
                         "This matters for a CE-vs-CORN comparison. argmax is the MODE and "
                         "is optimal for 0/1 loss (accuracy); the rank rule is the MEDIAN "
                         "and is optimal for absolute error. Since this pipeline selects on "
                         "set-MAE, 'canonical' gives the CORN arm the decoder matched to "
                         "the metric and the CE arm one that is not — so a naive comparison "
                         "measures head AND decoder together. Pass --decode argmax (or "
                         "median) to hold the decoder fixed and isolate the head.\n"
                         "Both rules are computed and logged every epoch regardless, so the "
                         "full 2x2 is available without re-running; this flag only picks "
                         "which one drives checkpoint selection and the [BEST] line.")
    ap.add_argument("--metrics-csv", dest="metrics_csv", default="",
                    help="Write one row per epoch (epoch, set_acc, macro_f1, set_mae, qwk, lr, "
                         "val_n) to this CSV. The sweep reads these curves rather than parsing "
                         "stdout, and needs the full curve — not just the [BEST] line — to "
                         "extract E* from a smoothed minimum.")
    ap.add_argument("--epochs", type=int, default=100,
                    help="MAX epochs. Training stops earlier once --patience epochs pass "
                         "with no set-MAE improvement, so this is a ceiling, not a "
                         "duration. Raised from 30 together with the addition of real "
                         "regularization (--dropout): before that, best-epoch selection "
                         "was the ONLY thing standing between 63k parameters and ~1.1k "
                         "training segments, and more epochs would only have widened the "
                         "gap it had to paper over.")
    ap.add_argument("--patience", type=int, default=20,
                    help="Stop after this many consecutive epochs with no improvement "
                         "(see --min-delta). 0 disables. Purely a compute saving: the "
                         "BEST epoch is checkpointed as it happens, so stopping early "
                         "never changes which weights ship — it only decides how long "
                         "to keep looking for a better one.")
    ap.add_argument("--min-delta", dest="min_delta", type=float, default=0.0,
                    help="Minimum set-MAE decrease that counts as an improvement — it "
                         "gates BOTH the checkpoint save and the patience counter. "
                         "0 (default) keeps the previous behaviour: any decrease saves. "
                         "Consider ~0.02: with a validation set of ~3 participants "
                         "(~360 segments) the standard error on set-MAE is about "
                         "0.9/sqrt(360) ~ 0.05, so improvements below that are noise, "
                         "and taking the min over many epochs biases the reported value "
                         "downward. Raising this makes early stopping more aggressive as "
                         "well as the saves rarer, which is why it is opt-in.")
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--lr",     type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4,
                    help="AdamW decoupled weight decay, applied to weight matrices and "
                         "(residual-parameterized) norm gains only — never to biases or "
                         "learnable_skip. See the param-group split below for why. NOTE "
                         "this is near-inert at the default: decoupled decay shrinks by "
                         "(1 - lr*wd) per step = 1 - 2e-7, i.e. ~0.04 % over 2,000 steps. "
                         "--dropout is what actually regularizes this model.")
    ap.add_argument("--dropout", type=float, default=0.15,
                    help="Dropout inside the xLSTM block stack. This is the model's only "
                         "active regularizer (see --weight_decay). Stored in the "
                         "checkpoint arch and rebuilt on load; disabled by model.eval() "
                         "for validation, fine-tuning embeddings and serving.")
    ap.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0,
                    help="Max global grad-norm; 0 disables. soft-CORN's own gradient is "
                         "bounded by 1 per unit, so the LOSS cannot explode — this "
                         "guards the 200-step recurrent unroll behind it, which can. "
                         "Cheap insurance that also keeps the L2-SP vs. ANIL comparison "
                         "from turning on one unlucky batch.")
    ap.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=3.0,
                    help="Linear LR warmup from 0 to --lr over this many epochs, then "
                         "CONSTANT (no decay). 0 disables. lr=2e-3 at batch 16 is high "
                         "for a 2-block recurrent stack, and the warmup is the cheap half "
                         "of the fix. Cosine decay is deliberately NOT applied: "
                         "docs/meta_optimization_options.md rejected it for the outer "
                         "meta-loop on the grounds that best-checkpoint selection plus "
                         "early stopping already neutralize it, and the same argument "
                         "holds here.")
    ap.add_argument("--context-length", dest="context_length", type=int, default=None,
                    help="Max sequence length. Defaults to window_seconds * resample_hz "
                         "(the exact grid length, so the frame cap never binds), or "
                         f"{DEFAULT_CONTEXT_LENGTH} when resampling is disabled.")
    ap.add_argument("--embedding-dim", dest="embedding_dim", type=int, default=64)
    ap.add_argument("--num-blocks", dest="num_blocks", type=int, default=2)
    ap.add_argument("--num-heads", dest="num_heads", type=int, default=4)
    ap.add_argument("--window-seconds", dest="window_seconds", type=float, default=10.0,
                    help="Truncate each segment to its LAST k seconds before encoding "
                         "(by frame timestamps, so it is robust to the actual sampling "
                         "rate). Default 10 = the second HALF of the 20 s label window; "
                         "20 would be the whole window. 0 disables. Stored in the "
                         "checkpoint so fine-tuning and inference inherit it. "
                         "NOTE this is the model's INPUT window, not the label cadence: "
                         "the drive UI still asks once per 20 s, so one segment is still "
                         "one label and scripts/sweep_train_frac.SEGMENT_SECONDS stays 20. "
                         "At the default resample_hz this makes context_length 100, so a "
                         "checkpoint trained at 20 s is not interchangeable with one "
                         "trained at 10 s — the stored arch keeps them apart.")
    ap.add_argument("--resample-hz", dest="resample_hz", type=float, default=DEFAULT_RESAMPLE_HZ,
                    help="Resample every segment onto a fixed time grid at this rate "
                         "AFTER the window_seconds truncation, so the number of "
                         "timesteps depends on a window's duration and not on the "
                         "sampling rate the session achieved (which varies ~2x between "
                         "sessions and correlates with scene load). Continuous dims are "
                         "linearly interpolated, one-hot/binary dims are held from the "
                         "last real frame, and holes longer than "
                         "xlstm_model.RESAMPLE_GAP_S are held rather than interpolated. "
                         "0 disables. Stored in the checkpoint so fine-tuning and "
                         "inference inherit it.")
    ap.add_argument("--loss", choices=["ce", "corn"], default="corn",
                    help="DEFAULT 'corn': rank-consistent ordinal head (K-1 conditional "
                         "logits) trained with SOFT-CORN (Shi et al. 2023, generalized to a "
                         "SET of marked LoAs; see docs/soft_corn_and_oldl.md). This is the "
                         "thesis path — LoA is ordinal, the design selects on set-MAE, and "
                         "the Laplace UQ layer REFUSES a non-CORN head — so it is the "
                         "default rather than something every caller has to remember. "
                         "'ce' (softmax + cross-entropy) is kept as the nominal-loss "
                         "ablation; it is blind to ordinal distance. Both accept "
                         "multi-label windows. The choice is baked into the checkpoint as "
                         "head_type and picked up automatically by fine_tune_XLSTM.py, the "
                         "sweep, xlstm_maml.py and the decision engine.")
    args = ap.parse_args()
    head_type = "corn" if args.loss == "corn" else "softmax"

    # Derive the sequence cap from the grid unless it was given explicitly. With
    # resampling on, window_seconds * resample_hz IS the sequence length, so any
    # other value either truncates real history or pads that never fills. The old
    # default (400) happened to equal 20 s x 20 Hz, which is why the frame cap and
    # the time cap used to coincide exactly at the nominal rate — and why the
    # effective history silently shrank whenever a session ran faster than that.
    resample_hz = args.resample_hz if args.resample_hz and args.resample_hz > 0 else None
    if args.context_length is None:
        if resample_hz and args.window_seconds and args.window_seconds > 0:
            args.context_length = int(round(args.window_seconds * resample_hz))
        else:
            args.context_length = DEFAULT_CONTEXT_LENGTH
    if resample_hz:
        print(f"[data] resampling to {resample_hz:g} Hz over {args.window_seconds:g} s "
              f"→ context_length={args.context_length}")
    else:
        print(f"[data] resampling DISABLED — sequence length follows the achieved "
              f"sampling rate (context_length={args.context_length})")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Say which device, ALWAYS. A CPU-only wheel falls back silently, and the
    # symptom (a slow run) is indistinguishable from a slow GPU -- which cost a
    # multi-hour sweep before this line existed. When CUDA is absent, name the
    # reason: `torch.version.cuda is None` means the +cpu wheel is installed,
    # which is what a plain `uv run` re-sync reinstates.
    if device == "cuda":
        print(f"[device] cuda — {torch.cuda.get_device_name(0)} "
              f"(torch {torch.__version__})", flush=True)
    else:
        why = ("CPU-ONLY torch build; re-run scripts/setup_cuda_torch.py and launch "
               "with `uv run --no-sync`" if torch.version.cuda is None
               else "CUDA build present but no device visible (driver?)")
        print(f"[device] cpu — {why} (torch {torch.__version__})", flush=True)

    train_ds, test_ds = (datasets_from_cache(args, resample_hz) if args.cache
                         else datasets_from_jsonl(args, resample_hz))
    if len(train_ds) == 0 or (len(test_ds) == 0 and not args.no_val):
        raise ValueError(f"Insufficient segments: train={len(train_ds)}, val={len(test_ds)}. Ensure Level_* labels exist.")
    collate = make_collate(args.context_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate)
    test_dl  = (DataLoader(test_ds, batch_size=max(8, args.batch), shuffle=False, collate_fn=collate)
                if len(test_ds) else None)

    model = XLSTMSequenceClassifier(
        d_in=D_IN,
        n_classes=5,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        context_length=args.context_length,
        #pool='last',
        head_type=head_type,
        dropout=args.dropout,
    ).to(device)

    # --- weight-decay param groups -----------------------------------------
    # Decaying EVERY parameter is wrong for this backbone, and not for the usual
    # reasons:
    #
    #   * fgate.bias initializes to a per-head linspace(3, 6) — deliberately
    #     large so the forget gates start near-open, i.e. so the cell starts with
    #     a LONG memory. Decaying it toward 0 shortens the memory horizon, which
    #     is precisely the thing a recurrent model over a 20 s window exists to
    #     learn. Same for igate.bias.
    #   * learnable_skip initializes to 1.0; decaying it toward 0 attenuates the
    #     residual path.
    #   * The LayerNorm gains are the EXCEPTION to the usual "never decay norms"
    #     advice. nx-ai's LayerNorm uses residual_weight=True, so the effective
    #     gain is (1 + weight) and the weight initializes to 0.0 — decay pulls it
    #     toward gain 1, i.e. toward the identity. That is a sensible shrinkage,
    #     so norm gains stay in the decay group.
    #
    # At the default weight_decay this changes almost nothing (see the flag's
    # help — decoupled decay is ~0.04 % over a full run). It matters the moment
    # anyone raises it to a value that actually regularizes, which is exactly
    # when the wrong grouping would quietly cost the model its memory horizon.
    decay, no_decay = [], []
    for _pname, _p in model.named_parameters():
        if not _p.requires_grad:
            continue
        (no_decay if _pname.endswith(".bias") or "learnable_skip" in _pname
         else decay).append(_p)
    opt = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    print(f"[optim] AdamW lr={args.lr:g} dropout={args.dropout:g} | weight_decay="
          f"{args.weight_decay:g} on {sum(p.numel() for p in decay)} params, "
          f"0.0 on {sum(p.numel() for p in no_decay)} (biases + learnable_skip)")

    # --- linear warmup, then constant --------------------------------------
    # Scheduler steps per BATCH, not per epoch, so the ramp is unaffected by how
    # many segments a dataset happens to hold.
    steps_per_epoch = max(1, len(train_dl))
    warmup_steps = int(round(max(0.0, args.warmup_epochs) * steps_per_epoch))
    sched = None
    if warmup_steps > 0:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup_steps))
        print(f"[optim] linear warmup over {warmup_steps} steps "
              f"({args.warmup_epochs:g} epochs x {steps_per_epoch} steps), then constant")
    multi = int(sum(1 for _, lvl in train_ds.groups if float(np.sum(lvl)) > 1))
    if multi:
        print(f"[info] {multi}/{len(train_ds.groups)} training segment(s) mark several "
              f"acceptable LoAs; targets become a distribution over them.")

    if head_type == "corn":
        # soft-CORN: each of the K-1 logits models P(y>k | y>k-1), trained on
        # its conditional subset weighted by P(y >= k). A single marked level
        # recovers the original CORN loss exactly (up to the normalizer).
        loss_fn = lambda logits, lvl: soft_corn_loss(logits, lvl)
    else:
        _ce = nn.CrossEntropyLoss()
        # Soft (B, K) targets — supported since torch 1.10 and numerically
        # identical to the integer form when exactly one level is marked.
        loss_fn = lambda logits, lvl: _ce(logits, levels_to_distribution(lvl))

    best = float("inf")  # select on MAE (lower is better); +inf ensures the first epoch always saves
    outp = pathlib.Path(args.out_pt); outp.parent.mkdir(parents=True, exist_ok=True)
    arch = dict(
        d_in=D_IN, n_classes=5, embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks, num_heads=args.num_heads,
        context_length=args.context_length,
        head_type=head_type,
        dropout=args.dropout,
        # Data contracts, not model kwargs: the time window segments were cut to
        # and the grid they were resampled onto. load_checkpoint() keeps them in
        # arch but does not pass them to the constructor; fine-tuning and
        # inference inherit them from here, so train and serve cannot end up on
        # different grids.
        window_seconds=(args.window_seconds if args.window_seconds and args.window_seconds > 0 else None),
        resample_hz=resample_hz,
    )

    # The constant-prediction floor for THIS validation set. It does not depend
    # on the model or the epoch, so it is computed once and printed before
    # training rather than recomputed per epoch — but it is written into every
    # metrics row so a downstream reader never has to re-derive it.
    base = {"const_loa_mae": -1, "const_set_mae": float("nan"),
            "const_loa_acc": -1, "const_set_acc": float("nan")}
    # Only meaningful when the val drivers are also train drivers, i.e. under
    # --split-mode within-driver. Left as NaN in participant mode, where a
    # per-driver constant is not knowable for an unseen driver.
    pdbase = {"pdconst_set_mae": float("nan"), "pdconst_set_acc": float("nan"),
              "pdconst_oracle_set_mae": float("nan"), "pdconst_n_drivers": 0}
    # The TRAINING drivers' label marginal — the distribution the model is
    # rewarded for reproducing. Printed next to the validation marginal because
    # the gap between them is the whole story on this cohort: a model whose
    # predictions track `train` while `val` wants something else has learned its
    # training drivers correctly and discovered they do not generalize, which is
    # a different failure from collapsing onto one level.
    train_hist = (np.stack([lvl for _, lvl in train_ds.groups], axis=0).sum(axis=0).astype(int)
                  if len(train_ds) else np.zeros(5, dtype=int))
    if test_dl is not None and len(test_ds):
        val_levels = np.stack([lvl for _, lvl in test_ds.groups], axis=0)
        base = constant_baseline(val_levels, 5)
        print(f"[labels] marks[LoA0..4]  train={train_hist.tolist()}  "
              f"val={val_levels.sum(axis=0).astype(int).tolist()}")
        print(f"[baseline] best CONSTANT prediction on this val set: "
              f"set-MAE={base['const_set_mae']:.3f} @LoA{base['const_loa_mae']} | "
              f"set-acc={base['const_set_acc']:.3f} @LoA{base['const_loa_acc']} "
              f"(val_n={len(val_levels)}) — a model above that MAE is losing to "
              f"'always predict one level'")
        if args.split_mode == "within-driver":
            train_levels = np.stack([lvl for _, lvl in train_ds.groups], axis=0)
            pdbase = per_driver_constant_baseline(
                train_levels, train_ds.pids, val_levels, test_ds.pids, 5)
            print(f"[baseline] best PER-DRIVER constant (fitted on each driver's train "
                  f"prefix, scored on their val tail): set-MAE={pdbase['pdconst_set_mae']:.3f} | "
                  f"set-acc={pdbase['pdconst_set_acc']:.3f} "
                  f"({pdbase['pdconst_n_drivers']} drivers; oracle-on-val "
                  f"{pdbase['pdconst_oracle_set_mae']:.3f}) — THIS is the floor that binds "
                  f"under --split-mode within-driver. Beating the global constant but not "
                  f"this one means the model learned WHO is driving, not WHEN they want "
                  f"autonomy.")

    metrics_fh = None
    metrics_writer = None
    if args.metrics_csv:
        mp = pathlib.Path(args.metrics_csv); mp.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = mp.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_fh)
        metrics_writer.writerow(["epoch", "set_acc", "macro_f1", "set_mae", "qwk", "lr", "val_n",
                                 "const_set_mae", "const_loa_mae",
                                 "const_set_acc", "const_loa_acc",
                                 # NaN unless --split-mode within-driver; see
                                 # per_driver_constant_baseline for why this is
                                 # the binding floor there and the global
                                 # constant is not.
                                 "pdconst_set_mae", "pdconst_set_acc",
                                 "pdconst_oracle_set_mae", "pdconst_n_drivers",
                                 # Both decoders every epoch: the CE-vs-CORN
                                 # comparison is only interpretable if the
                                 # decoder can be held fixed post hoc.
                                 "mae_argmax", "acc_argmax", "mae_median", "acc_median",
                                 "pred_loa0", "pred_loa1", "pred_loa2", "pred_loa3", "pred_loa4",
                                 # Constant per run; repeated per row so a reader
                                 # of one row can answer "did training help?"
                                 "init_set_mae", "init_set_acc", "selectable"])

    @torch.no_grad()
    def evaluate_val():
        """Set-aware metrics on the validation set under BOTH decoders."""
        model.eval(); y_arg = []; y_med = []; y_lvl = []
        for xb, lb, vb in test_dl:
            xb, lb = xb.to(device), lb.to(device)
            probs = logits_to_probs(model(xb, lengths=lb), head_type)
            y_arg.append(probs_to_label(probs, 'softmax').cpu().numpy())
            y_med.append(probs_to_label(probs, 'corn').cpu().numpy())
            y_lvl.append(vb.numpy())
        Ya, Ym, Yl_ = np.concatenate(y_arg), np.concatenate(y_med), np.concatenate(y_lvl)
        return Ya, Ym, Yl_

    # UNTRAINED REFERENCE. Evaluated before a single gradient step, so it is the
    # honest answer to "does training this model achieve anything at all?" —
    # stronger than the epoch-0 number, which already includes a full pass over
    # the data (at warmup LR, so at a fraction of the intended step size).
    # A run whose best epoch does not clearly beat this line has not learned;
    # one that does not beat the constant floor has learned something that does
    # not transfer to held-out drivers. They are different diagnoses.
    init = {}
    if test_dl is not None:
        Ya0, Ym0, Yl0 = evaluate_val()
        c0 = Ya0 if head_type == 'softmax' else Ym0
        init = {"mae": set_mae(Yl0, c0), "acc": set_accuracy(Yl0, c0)}
        print(f"[init] UNTRAINED model: set-MAE={init['mae']:.3f} set-acc={init['acc']:.3f} "
              f"(vs constant floor {base['const_set_mae']:.3f}) — training must beat THIS "
              f"line to have learned anything")

    bad_epochs = 0   # consecutive epochs without an improvement (see --patience)
    saved_any = False
    for ep in range(args.epochs):
        model.train()
        for xb, lb, vb in train_dl:
            xb, lb, vb = xb.to(device), lb.to(device), vb.to(device)
            logits = model(xb, lengths=lb)
            loss = loss_fn(logits, vb)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                # BEFORE opt.step(), AFTER backward() — clipping the accumulated
                # .grad is the whole mechanism; anywhere else it is a no-op.
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()   # per BATCH: the warmup is defined in steps

        if test_dl is None:
            # --no-val: no evaluation, no selection. The checkpoint is written
            # once after the loop, from the final epoch.
            print(f"[epoch {ep:02d}] (no validation set; lr={opt.param_groups[0]['lr']:.2e})")
            continue

        # The PMF is head-specific (softmax vs the CORN chain rule), but the RULE
        # that collapses it to one LoA is a separate choice, so both are
        # computed: argmax (mode) vs sum_k 1[q_k>0.5] (median).
        Yarg, Ymed, Yl = evaluate_val()
        canonical = Yarg if head_type == 'softmax' else Ymed
        Yp = {"canonical": canonical, "argmax": Yarg, "median": Ymed}[args.decode]
        # All four metrics are set-aware: they credit any level the driver
        # marked acceptable, and reduce EXACTLY to their single-label forms on
        # rows that mark one level.
        sacc = set_accuracy(Yl, Yp); mf1 = set_macro_f1(Yl, Yp, 5)
        err = set_mae(Yl, Yp); kappa = set_qwk(Yl, Yp, 5)
        cur_lr = opt.param_groups[0]["lr"]
        # "vs const" is the number to watch: negative means the model is worse
        # than predicting a single fixed LoA on this fold.
        d_mae = err - base["const_set_mae"]
        d_acc = sacc - base["const_set_acc"]
        print(f"[epoch {ep:02d}] set-acc={sacc:.3f} macro-F1={mf1:.3f} "
              f"set-MAE={err:.3f} QWK={kappa:.3f} (val_n={len(Yp)}, lr={cur_lr:.2e}) "
              f"| vs const: MAE {d_mae:+.3f} acc {d_acc:+.3f}"
              f"{'' if d_mae < 0 else '  <-- WORSE THAN A CONSTANT'}")
        # PREDICTION HISTOGRAM: how the epoch's predictions spread over the five
        # LoAs, against the label marginal on the same segments. A model whose
        # counts pile onto one or two levels is reproducing the marginal, and its
        # set-MAE is the constant baseline wearing a different hat — which the
        # aggregate metrics alone cannot distinguish from genuine learning.
        pred_hist = np.bincount(Yp.astype(int), minlength=5)[:5]
        lbl_hist = np.asarray(Yl).sum(axis=0).astype(int)[:5]
        # Both decoders, so the CE-vs-CORN comparison never needs a re-run.
        mae_arg, mae_med = set_mae(Yl, Yarg), set_mae(Yl, Ymed)
        acc_arg, acc_med = set_accuracy(Yl, Yarg), set_accuracy(Yl, Ymed)
        def _pct(v):
            t = int(np.sum(v)) or 1
            return "[" + " ".join(f"{100 * x / t:4.1f}" for x in v) + "]"
        print(f"           %LoA0..4  pred {_pct(pred_hist)}  val {_pct(lbl_hist)}  "
              f"train {_pct(train_hist)}   (counts pred={pred_hist.tolist()})")
        print(f"           argmax MAE={mae_arg:.3f} acc={acc_arg:.3f}"
              f"  |  median MAE={mae_med:.3f} acc={acc_med:.3f}")
        if metrics_writer is not None:
            metrics_writer.writerow([ep, sacc, mf1, err, kappa, cur_lr, len(Yp),
                                     base["const_set_mae"], base["const_loa_mae"],
                                     base["const_set_acc"], base["const_loa_acc"],
                                     pdbase["pdconst_set_mae"], pdbase["pdconst_set_acc"],
                                     pdbase["pdconst_oracle_set_mae"], pdbase["pdconst_n_drivers"],
                                     mae_arg, acc_arg, mae_med, acc_med,
                                     *pred_hist.tolist(),
                                     init.get("mae", float("nan")),
                                     init.get("acc", float("nan")),
                                     int(ep >= args.min_select_epoch)])
            metrics_fh.flush()   # the sweep reads these while the run is in flight

        # Select on set-MAE: LoA is ordinal, so off-by-1 << off-by-4, and
        # accuracy is blind to error distance (and to majority-class collapse
        # under the class imbalance). MAE is the design-doc primary metric, and
        # the set form is the one that does not silently score a multi-label
        # window against the driver's lowest acceptable level.
        #
        # ONE definition of "improved", used for both the save and the patience
        # counter, so the two cannot disagree about whether an epoch helped.
        # EPOCH FLOOR. Epochs below --min-select-epoch are trained, evaluated and
        # logged, but cannot be selected or checkpointed. Two reasons:
        #   * The first --warmup-epochs run at a reduced LR, so choosing among
        #     them is choosing an artifact of the schedule, not a better model.
        #   * On a cohort where held-out-driver performance degrades with
        #     training, the unconstrained argmin lands on epoch 0 — and E*=0
        #     makes run_lodo_population execute `for ep in range(0)`, i.e. no
        #     training at all, saving 12 randomly-initialized checkpoints as the
        #     population models. That failure is silent.
        # The floor makes the comparison one between TRAINED models. If the best
        # such model still loses to [init] or to the constant floor, that is the
        # finding — and it is now visible rather than hidden behind an epoch-0
        # selection.
        if ep < args.min_select_epoch:
            print(f"           (epoch < --min-select-epoch {args.min_select_epoch}: "
                  f"logged, not selectable)")
        elif err < best - args.min_delta:
            best = err
            bad_epochs = 0
            saved_any = True
            save_checkpoint(model, str(outp), arch=arch)
            print(f"[OK] saved -> {outp} (set-MAE={err:.3f})")
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"[early-stop] no set-MAE improvement > {args.min_delta:g} for "
                      f"{bad_epochs} epochs; stopping at epoch {ep} of {args.epochs}. "
                      f"The best epoch is already on disk.")
                break

    if metrics_fh is not None:
        metrics_fh.close()

    if test_dl is None:
        # --no-val: nothing was selected, so the FINAL epoch is the model. Saved
        # once, here, rather than inside the loop.
        save_checkpoint(model, str(outp), arch=arch)
        print(f"[OK] saved -> {outp} (final epoch {args.epochs - 1}, no validation, "
              f"no epoch selection)")
        return

    # This is the MINIMUM over epochs of a quantity estimated on ~360 segments,
    # so it is optimistically biased by roughly 1-2 standard errors (~0.05 each)
    # — the winner's curse of selecting on the same set you report. Quote it as
    # the selection criterion it is, not as the population model's expected
    # performance on an unseen driver; the LODO folds are what estimates that.
    if not saved_any:
        # --epochs never reached --min-select-epoch, so nothing was ever
        # eligible. Saving the final epoch beats leaving no checkpoint at all
        # (the caller expects a file), but the run cannot be treated as selected.
        save_checkpoint(model, str(outp), arch=arch)
        print(f"[warn] no epoch was selectable: --epochs {args.epochs} never reached "
              f"--min-select-epoch {args.min_select_epoch}. Saved the FINAL epoch "
              f"unselected — do not treat this checkpoint as chosen.")

    print(f"[BEST] set-MAE={best:.3f} (selection minimum over epochs >= "
          f"{args.min_select_epoch} — optimistically biased, see note in train_XLSTM.py)")
    if init:
        print(f"[BEST] vs untrained init {init['mae']:.3f} -> {best - init['mae']:+.3f}"
              f"   vs constant floor {base['const_set_mae']:.3f} -> "
              f"{best - base['const_set_mae']:+.3f}")


if __name__ == "__main__":
    main()
