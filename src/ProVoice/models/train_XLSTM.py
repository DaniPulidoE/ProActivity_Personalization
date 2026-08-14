# Usage: python -m ProVoice.train_XLSTM --in data/with_segments.jsonl --label-map data/labels.csv --out trained_models/state_xlstm.pt
import argparse, csv, json, pathlib, random
from typing import List, Dict, Any, Tuple

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
            if log_fh is not None:
                log_encoded_frames(log_fh, split, str(gid), X, levels=lvl)
        if skipped:
            print(f"[warn] skipped {len(skipped)} segment(s) with missing/empty Level_* labels: "
                f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")


    def __len__(self): return len(self.groups)
    def __getitem__(self, i): return self.groups[i]


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
    ap.add_argument("--val-pids", dest="val_pids", default="",
                    help="Comma-separated participantids to use as the validation set, e.g. "
                         "'001,005'. Overrides the default seeded 80/20 participant split, "
                         "which cannot express a FIXED fold — needed by "
                         "ProVoice.training_scripts.sweep_population_hparams, where the same "
                         "6 folds must be reused across every config and seed or the "
                         "comparison is confounded by which drivers each config was "
                         "validated on. Ignored when --no-val is set.")
    ap.add_argument("--no-val", dest="no_val", action="store_true",
                    help="Train on EVERY participant in the input with no validation set, no "
                         "per-epoch evaluation and no best-epoch selection; the final-epoch "
                         "weights are saved. This is the LODO population-training mode: the "
                         "held-out driver is absent from the input file entirely and is scored "
                         "afterwards by the caller, so no signal from it can reach the "
                         "checkpoint. Requires a pre-chosen --epochs (E*); --patience is "
                         "ignored because there is nothing to be patient about.")
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
    ap.add_argument("--window-seconds", dest="window_seconds", type=float, default=20.0,
                    help="Truncate each segment to its LAST k seconds before encoding "
                         "(by frame timestamps, so it is robust to the actual sampling "
                         "rate). Default 20 = the full label window. 0 disables. "
                         "Stored in the checkpoint so fine-tuning and inference inherit it.")
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
    ap.add_argument("--loss", choices=["ce", "corn"], default="ce",
                    help="'ce': softmax head + cross-entropy (nominal — blind to ordinal "
                         "distance). 'corn': rank-consistent ordinal head (K-1 conditional "
                         "logits) trained with soft-CORN (Shi et al. 2023, generalized to a "
                         "SET of marked LoAs; see docs/soft_corn_and_oldl.md). Both accept "
                         "multi-label windows. The choice is baked into the checkpoint and "
                         "picked up automatically by fine_tune_XLSTM.py and the decision "
                         "engine; only 'corn' supports the Laplace UQ layer.")
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

    rows = [normalize_row(r) for r in iter_jsonl(pathlib.Path(args.in_jsonl))]
    if not rows:
        raise ValueError("JSONL is empty or contains no valid rows.")
    df = pd.DataFrame(rows)

    if args.label_map:
        lm = pd.read_csv(args.label_map)
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
    
    """ Original case: train-val split done randomly, not by participant
    gids = df['segment_id'].drop_duplicates().sample(frac=1.0, random_state=args.seed).values
    ntr = max(1, int(0.8 * len(gids)))
    tr_ids, te_ids = set(gids[:ntr]), set(gids[ntr:])
    tr_df = df[df['segment_id'].isin(tr_ids)].reset_index(drop=True)
    te_df = df[df['segment_id'].isin(te_ids)].reset_index(drop=True)
    """
    # Train-validation split: by participant when ≥2 participants, else by segment.
    if df[SPLIT_VARIABLE].eq("").all():
        raise ValueError(f"Split variable '{SPLIT_VARIABLE}' is missing from all rows.")
    pids = df[SPLIT_VARIABLE].drop_duplicates().sample(frac=1.0, random_state=args.seed).values
    if args.no_val:
        # LODO population training: every provided driver trains, nothing is held
        # back, no epoch is selected. The held-out driver is not in this file at
        # all, and is scored by the caller AFTER training — so there is no path
        # by which a validation signal could pick the checkpoint.
        print(f"[split] --no-val: training on ALL {len(pids)} participant(s), "
              f"no validation set, no epoch selection")
        tr_df = df.reset_index(drop=True)
        te_df = df.iloc[0:0]
    elif args.val_pids:
        want = [p.strip() for p in args.val_pids.split(",") if p.strip()]
        have = set(str(p) for p in pids)
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
        tr_df = df[df[SPLIT_VARIABLE].astype(str).isin(tr_pids)].reset_index(drop=True)
        te_df = df[df[SPLIT_VARIABLE].astype(str).isin(te_pids)].reset_index(drop=True)
    elif len(pids) >= 2:
        ntr = max(1, int(0.8 * len(pids)))
        tr_pids, te_pids = set(pids[:ntr]), set(pids[ntr:])
        print(f"[split] train participants={sorted(tr_pids)}  val participants={sorted(te_pids)}")
        tr_df = df[df[SPLIT_VARIABLE].isin(tr_pids)].reset_index(drop=True)
        te_df = df[df[SPLIT_VARIABLE].isin(te_pids)].reset_index(drop=True)
    else:
        print(f"[split] only {len(pids)} participant(s) — falling back to segment-level 80/20 split")
        gids = df['segment_id'].drop_duplicates().sample(frac=1.0, random_state=args.seed).values
        ntr = max(1, int(0.8 * len(gids)))
        tr_ids, te_ids = set(gids[:ntr]), set(gids[ntr:])
        tr_df = df[df['segment_id'].isin(tr_ids)].reset_index(drop=True)
        te_df = df[df['segment_id'].isin(te_ids)].reset_index(drop=True)
        print(f"[split] train segments={len(tr_ids)}  val segments={len(te_ids)}")

    log_fh = open(args.log_path, "w", encoding="utf-8") if args.log_path else None
    if log_fh:
        print(f"[log] writing feature log → {args.log_path}")
    try:
      train_ds = SeqDataset(tr_df, context_length=args.context_length, split="train", log_fh=log_fh,
                            window_seconds=args.window_seconds, resample_hz=resample_hz)
      # Under --no-val there is no validation frame to build a dataset from, and
      # SeqDataset rightly refuses an empty one (its segment_id assert). Skip the
      # construction rather than weakening that guard, which exists to catch a
      # genuinely empty split in the normal path.
      test_ds  = ([] if args.no_val else
                  SeqDataset(te_df, context_length=args.context_length, split="val", log_fh=log_fh,
                             window_seconds=args.window_seconds, resample_hz=resample_hz))
    finally:
      if log_fh:
          log_fh.close()
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

    metrics_fh = None
    metrics_writer = None
    if args.metrics_csv:
        mp = pathlib.Path(args.metrics_csv); mp.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = mp.open("w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_fh)
        metrics_writer.writerow(["epoch", "set_acc", "macro_f1", "set_mae", "qwk", "lr", "val_n"])

    bad_epochs = 0   # consecutive epochs without an improvement (see --patience)
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

        model.eval(); y_pred = []; y_lvl = []
        with torch.no_grad():
            for xb, lb, vb in test_dl:
                xb, lb = xb.to(device), lb.to(device)
                logits = model(xb, lengths=lb)
                # Each head decoded by its canonical rule: argmax for softmax,
                # Shi et al.'s rank rule sum_k 1[q_k > 0.5] for CORN.
                pred = logits_to_label(logits, head_type)
                y_pred.append(pred.cpu().numpy())
                y_lvl.append(vb.numpy())
        Yp = np.concatenate(y_pred, 0)
        Yl = np.concatenate(y_lvl, 0)
        # All four metrics are set-aware: they credit any level the driver
        # marked acceptable, and reduce EXACTLY to their single-label forms on
        # rows that mark one level.
        sacc = set_accuracy(Yl, Yp); mf1 = set_macro_f1(Yl, Yp, 5)
        err = set_mae(Yl, Yp); kappa = set_qwk(Yl, Yp, 5)
        cur_lr = opt.param_groups[0]["lr"]
        print(f"[epoch {ep:02d}] set-acc={sacc:.3f} macro-F1={mf1:.3f} "
              f"set-MAE={err:.3f} QWK={kappa:.3f} (val_n={len(Yp)}, lr={cur_lr:.2e})")
        if metrics_writer is not None:
            metrics_writer.writerow([ep, sacc, mf1, err, kappa, cur_lr, len(Yp)])
            metrics_fh.flush()   # the sweep reads these while the run is in flight

        # Select on set-MAE: LoA is ordinal, so off-by-1 << off-by-4, and
        # accuracy is blind to error distance (and to majority-class collapse
        # under the class imbalance). MAE is the design-doc primary metric, and
        # the set form is the one that does not silently score a multi-label
        # window against the driver's lowest acceptable level.
        #
        # ONE definition of "improved", used for both the save and the patience
        # counter, so the two cannot disagree about whether an epoch helped.
        if err < best - args.min_delta:
            best = err
            bad_epochs = 0
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
    print(f"[BEST] set-MAE={best:.3f} (selection minimum — optimistically biased, "
          f"see note in train_XLSTM.py)")


if __name__ == "__main__":
    main()
