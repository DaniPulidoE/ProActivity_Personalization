# Usage: python -m ProVoice.train_XLSTM --in data/with_segments.jsonl --label-map data/labels.csv --out trained_models/state_xlstm.pt
import argparse, json, pathlib, random
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function
from ProVoice.models.xlstm_model import (
    encode_frame,
    D_IN,
    STATE_CAT,
    STATE_NUM,
    STATE_CARLA,
    XLSTMSequenceClassifier,
    save_checkpoint,
    DEFAULT_CONTEXT_LENGTH,
    FEATURE_NAMES,
    log_encoded_frames,
    logits_to_probs,
)
from ProVoice.models.xlstm_model import _as01
from ProVoice.decision_engine import truncate_frames_by_seconds
from coral_pytorch.losses import coral_loss, corn_loss


LEVELS = [f"Level_{i}" for i in range(1, 6)]


def levels_to_distribution(lvl: torch.Tensor) -> torch.Tensor:
    """Multi-hot (B, K) -> PMF (B, K), uniform over the marked levels.

    A single marked level yields the usual one-hot, so single-label data is
    numerically unchanged.
    """
    return lvl / lvl.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def levels_to_cumulative(lvl: torch.Tensor) -> torch.Tensor:
    """Multi-hot (B, K) -> CORAL target (B, K-1) with q_k = P(y > k).

    This is what lets CORAL take more than one label: the driver's marked set
    becomes a distribution, and its complementary CDF is a valid soft target for
    coral_loss (which is a sum of BCEs, so targets need only lie in [0, 1]).
    The mapping is invertible, so nothing about which levels were marked is lost.
    """
    p = levels_to_distribution(lvl)
    return p.flip(-1).cumsum(-1).flip(-1)[..., 1:]
SPLIT_VARIABLE = "participantid"


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
    return rows


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
    # CARLA vehicle/world features — use sentinel defaults matching encode_frame expectations
    out['speed_ratio_max']   = pick('speed_ratio_max',   default=None)
    out['speed_ratio_limit'] = pick('speed_ratio_limit', default=-1)
    out['brake']             = pick('brake',             default=None)
    out['steer']             = pick('steer',             default=None)
    out['precipitation']     = pick('precipitation',     default=None)
    out['is_night']          = pick('is_night',          default=None)
    out['is_junction']       = pick('is_junction',       default=None)
    out['perclos']       = pick('perclos',       default=0.0)
    out['gaze_score']    = pick('gaze_score',    default=0.0)
    out['hr_delta']      = pick('hr_delta',      default=0.0)
    out['rr_delta']      = pick('rr_delta',      default=0.0)
    out['blink_rate']    = pick('blink_rate',    default=0.0)
    out['yawn_rate']     = pick('yawn_rate',     default=0.0)

    for k in LEVELS:
        if k in row and row[k] not in (None, ""):
            out[k] = int(float(row[k]))
    return out


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
    ):
        assert 'segment_id' in df.columns and df['segment_id'].astype(bool).any(), "segment_id is required"
        self.context_length = context_length
        self.groups: List[Tuple[np.ndarray, int]] = []
        skipped = []
        for gid, g in df.groupby('segment_id'):
            g = g.reset_index(drop=True)
            if not all(k in g.columns for k in LEVELS):
                continue
            level_vec = g[LEVELS].iloc[0].astype(float).values
            if np.isnan(level_vec).any() or level_vec.sum() <= 0:
                skipped.append(gid)
                continue
            # Keep BOTH representations: the multi-hot drives CORAL/CE (which
            # accept soft targets), while the argmax int is what CORN and the
            # legacy single-label metrics need.
            lvl = (level_vec > 0).astype(np.float32)
            y = int(np.argmax(level_vec))
            rows = [g.iloc[i].to_dict() for i in range(len(g))]
            # Keep only the LAST window_seconds of the segment (frames are
            # chronological within a segment). None/0 = use the full segment.
            rows = truncate_frames_by_seconds(rows, window_seconds)
            xs = [encode_frame(r.get('functionname') or "", r) for r in rows]
            X = np.stack(xs, axis=0).astype(np.float32)
            self.groups.append((X, y, lvl))
            if log_fh is not None:
                log_encoded_frames(log_fh, split, str(gid), X, label=y)
        if skipped:
            print(f"[warn] skipped {len(skipped)} segment(s) with missing/empty Level_* labels: "
                f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")


    def __len__(self): return len(self.groups)
    def __getitem__(self, i): return self.groups[i]


def make_collate(context_length: int):
    def collate(batch):
        if len(batch) == 0:
            return (torch.empty(0, context_length, D_IN),
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, len(LEVELS)))
        xs, ys, ls, lvls = [], [], [], []
        for X, y, lvl in batch:
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
            ys.append(int(y))
            ls.append(min(T, context_length))
            lvls.append(torch.from_numpy(np.asarray(lvl, dtype=np.float32)))
        return (torch.stack(xs, 0),
                torch.tensor(ys, dtype=torch.long),
                torch.tensor(ls, dtype=torch.long),
                torch.stack(lvls, 0))
    return collate


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0: return 0.0
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in LoA levels — ordinal metric: off-by-1 < off-by-4."""
    if len(y_true) == 0: return 0.0
    return float(np.abs(y_true.astype(float) - y_pred.astype(float)).mean())


def set_accuracy(levels: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions the driver marked as acceptable.

    Identical to :func:`accuracy` when every row marks exactly one level.
    """
    if len(y_pred) == 0: return 0.0
    return float(levels[np.arange(len(y_pred)), y_pred].astype(bool).mean())


def set_mae(levels: np.ndarray, y_pred: np.ndarray) -> float:
    """Distance to the NEAREST marked level; 0 when the prediction is accepted.

    Generalises :func:`mae` to multi-label rows without punishing a model for
    picking one acceptable level over another. Reduces exactly to ``mae`` when
    every row marks a single level.
    """
    if len(y_pred) == 0: return 0.0
    idx = np.arange(levels.shape[1])
    dists = []
    for row, p in zip(levels, y_pred):
        marked = idx[row.astype(bool)]
        dists.append(float(np.abs(marked - p).min()) if marked.size else 0.0)
    return float(np.mean(dists))


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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--lr",     type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--context-length", dest="context_length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    ap.add_argument("--embedding-dim", dest="embedding_dim", type=int, default=64)
    ap.add_argument("--num-blocks", dest="num_blocks", type=int, default=2)
    ap.add_argument("--num-heads", dest="num_heads", type=int, default=4)
    ap.add_argument("--window-seconds", dest="window_seconds", type=float, default=20.0,
                    help="Truncate each segment to its LAST k seconds before encoding "
                         "(by frame timestamps, so it is robust to the actual sampling "
                         "rate). Default 20 = the full label window. 0 disables. "
                         "Stored in the checkpoint so fine-tuning and inference inherit it.")
    ap.add_argument("--loss", choices=["ce", "corn", "coral"], default="ce",
                    help="'ce': softmax head + cross-entropy (nominal). 'corn': rank-consistent "
                         "ordinal head (K-1 conditional logits) + CORN loss (Shi et al. 2023). "
                         "'coral': ordinal head with ONE shared weight vector + K-1 biases + "
                         "CORAL loss (Cao et al. 2020) — the only option that accepts more than "
                         "one marked LoA per window, since its target is a cumulative vector. "
                         "The choice is baked into the checkpoint and picked up automatically "
                         "by fine_tune_XLSTM.py and the decision engine.")
    args = ap.parse_args()
    head_type = {"corn": "corn", "coral": "coral"}.get(args.loss, "softmax")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [normalize_row(r) for r in read_jsonl(pathlib.Path(args.in_jsonl))]
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
        default = -1 if k == 'speed_ratio_limit' else 0.0
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
    if len(pids) >= 2:
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
                            window_seconds=args.window_seconds)
      test_ds  = SeqDataset(te_df, context_length=args.context_length, split="val",   log_fh=log_fh,
                            window_seconds=args.window_seconds)
    finally:
      if log_fh:
          log_fh.close()
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Insufficient segments: train={len(train_ds)}, val={len(test_ds)}. Ensure Level_* labels exist.")
    collate = make_collate(args.context_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=max(8, args.batch), shuffle=False, collate_fn=collate)

    model = XLSTMSequenceClassifier(
        d_in=D_IN,
        n_classes=5,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        context_length=args.context_length,
        #pool='last',
        head_type=head_type,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Reject multi-label data on the CORN path rather than training on a
    # silently-thresholded label: corn_loss partitions samples into HARD
    # conditional subsets, and it does not error on a soft target — it just
    # rounds it, which would corrupt the run without any warning.
    multi = int(sum(1 for _, _, lvl in train_ds.groups if float(np.sum(lvl)) > 1))
    if multi and head_type == "corn":
        raise SystemExit(
            f"--loss corn cannot represent multiple marked LoAs, but {multi} training "
            f"segment(s) mark more than one. Use --loss coral (ordinal, accepts a set) "
            f"or --loss ce (nominal, treats the set as a uniform distribution)."
        )
    if multi:
        print(f"[info] {multi}/{len(train_ds.groups)} training segment(s) mark several "
              f"acceptable LoAs; targets become a distribution over them.")

    if head_type == "corn":
        # CORN trains each of the K-1 logits as P(y>k | y>k-1) on its
        # conditional subset; corn_loss builds those subsets internally.
        loss_fn = lambda logits, yb, lvl: corn_loss(logits, yb, num_classes=5)
    elif head_type == "coral":
        # CORAL's target is the complementary CDF of the marked set, so a
        # single mark gives the standard 0/1 extended label and several marks
        # give intermediate values. No chain rule, no hard subsets.
        loss_fn = lambda logits, yb, lvl: coral_loss(logits, levels_to_cumulative(lvl))
    else:
        _ce = nn.CrossEntropyLoss()
        # Soft (B, K) targets — supported since torch 1.10 and numerically
        # identical to the integer form when exactly one level is marked.
        loss_fn = lambda logits, yb, lvl: _ce(logits, levels_to_distribution(lvl))

    best = float("inf")  # select on MAE (lower is better); +inf ensures the first epoch always saves
    outp = pathlib.Path(args.out_pt); outp.parent.mkdir(parents=True, exist_ok=True)
    arch = dict(
        d_in=D_IN, n_classes=5, embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks, num_heads=args.num_heads,
        context_length=args.context_length,
        head_type=head_type,
        # Data contract, not a model kwarg: the time window segments were cut
        # to. load_checkpoint() keeps it in arch but does not pass it to the
        # constructor; fine-tuning and inference inherit it from here.
        window_seconds=(args.window_seconds if args.window_seconds and args.window_seconds > 0 else None),
    )

    for ep in range(args.epochs):
        model.train()
        for xb, yb, lb, vb in train_dl:
            xb, yb, lb, vb = xb.to(device), yb.to(device), lb.to(device), vb.to(device)
            logits = model(xb, lengths=lb)
            loss = loss_fn(logits, yb, vb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); y_true = []; y_pred = []; y_lvl = []
        with torch.no_grad():
            for xb, yb, lb, vb in test_dl:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                logits = model(xb, lengths=lb)
                # logits_to_probs maps every head type to a 5-class PMF, so
                # argmax decoding is identical for CE, CORN and CORAL.
                pred = logits_to_probs(logits, head_type).argmax(dim=-1)
                y_true.append(yb.cpu().numpy()); y_pred.append(pred.cpu().numpy())
                y_lvl.append(vb.numpy())
        Yt = np.concatenate(y_true, 0); Yp = np.concatenate(y_pred, 0)
        Yl = np.concatenate(y_lvl, 0)
        acc = accuracy(Yt, Yp); mf1 = macro_f1(Yt, Yp, 5)
        kappa = qwk(Yt, Yp, 5)
        # Set-aware: credit any level the driver marked acceptable. Both reduce
        # to the plain single-label metrics when one level is marked per row.
        err = set_mae(Yl, Yp); sacc = set_accuracy(Yl, Yp)
        print(f"[epoch {ep:02d}] acc={acc:.3f} set-acc={sacc:.3f} macro-F1={mf1:.3f} "
              f"MAE={err:.3f} QWK={kappa:.3f} (val_n={len(Yt)})")

        # Select on MAE: LoA is ordinal, so off-by-1 << off-by-4, and accuracy
        # is blind to error distance (and to majority-class collapse under the
        # class imbalance). MAE is the design-doc primary metric.
        if err < best:
            best = err
            save_checkpoint(model, str(outp), arch=arch)
            print(f"[OK] saved -> {outp} (MAE={err:.3f})")

    print(f"[BEST] MAE={best:.3f}")


if __name__ == "__main__":
    main()
