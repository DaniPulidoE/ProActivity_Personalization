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
    XLSTMSequenceClassifier,
    save_checkpoint,
    DEFAULT_CONTEXT_LENGTH,
)

LEVELS = [f"Level_{i}" for i in range(1, 6)]


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _as01(x: Any) -> float:
    s = str(x).strip().lower()
    if s in ('true', '1', 't', 'yes', 'y'): return 1.0
    if s in ('false', '0', 'f', 'no', 'n', '', 'nan', 'none', 'null'): return 0.0
    try: return float(s)
    except Exception: return 0.0


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
    out['segment_id']      = pick('segment_id', 'segment', 'trial_id', 'trial', 'block_id')
    out['functionname']    = pick('functionname', 'function', 'func_name', 'FunctionName')
    out['environment']     = pick('environment', 'env', 'environment_type')
    out['secondary_task']  = pick('secondary_task', 'sec_task', 'secondaryTask')
    out['lab']             = pick('lab', 'lab_state')
    out['emotion']         = pick('emotion', 'affect', 'emo', 'mood', 'Emotion')
    out['drowsiness_alert']= pick('drowsiness_alert', 'drowsy', 'fatigue')
    out['gaze_distracted'] = pick('gaze_distracted', 'gaze', 'distraction')
    out['heart_rate']      = pick('heart_rate', 'hr', 'heartrate', 'bpm')
    for k in LEVELS:
        if k in row and row[k] not in (None, ""):
            out[k] = int(float(row[k]))
    return out


def load_label_map(path: str | None) -> Dict[str, List[int]]:
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
    def __init__(self, df: pd.DataFrame, context_length: int = DEFAULT_CONTEXT_LENGTH):
        assert 'segment_id' in df.columns and df['segment_id'].astype(bool).any(), "segment_id is required"
        self.context_length = context_length
        self.groups: List[Tuple[np.ndarray, int]] = []
        for gid, g in df.groupby('segment_id'):
            g = g.reset_index(drop=True)
            if not all(k in g.columns for k in LEVELS):
                continue
            level_vec = g[LEVELS].iloc[0].astype(float).values
            y = int(np.argmax(level_vec))  # single-label 5-class target
            xs = [encode_frame(g.iloc[i].get('functionname') or "", g.iloc[i].to_dict()) for i in range(len(g))]
            X = np.stack(xs, axis=0).astype(np.float32)
            self.groups.append((X, y))

    def __len__(self): return len(self.groups)
    def __getitem__(self, i): return self.groups[i]


def make_collate(context_length: int):
    def collate(batch):
        if len(batch) == 0:
            return (torch.empty(0, context_length, D_IN),
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long))
        xs, ys, ls = [], [], []
        for X, y in batch:
            T = X.shape[0]
            if T > context_length:
                X = X[-context_length:]
            pad = context_length - X.shape[0]
            if pad > 0:
                # LEFT-pad with zero vectors so h[:, -1, :] is always valid.
                X = np.concatenate([np.zeros((pad, X.shape[1]), dtype=X.dtype), X], axis=0)
            xs.append(torch.from_numpy(X))
            ys.append(int(y))
            ls.append(min(T, context_length))
        return (torch.stack(xs, 0),
                torch.tensor(ys, dtype=torch.long),
                torch.tensor(ls, dtype=torch.long))
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


def main():
    ap = argparse.ArgumentParser(description="Train official xLSTM (single-label 5-class).")
    ap.add_argument("--in",        dest="in_jsonl", required=True)
    ap.add_argument("--out",       dest="out_pt",   default="trained_models/state_xlstm.pt")
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
    args = ap.parse_args()

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

    gids = df['segment_id'].drop_duplicates().sample(frac=1.0, random_state=args.seed).values
    ntr = max(1, int(0.8 * len(gids)))
    tr_ids, te_ids = set(gids[:ntr]), set(gids[ntr:])
    tr_df = df[df['segment_id'].isin(tr_ids)].reset_index(drop=True)
    te_df = df[df['segment_id'].isin(te_ids)].reset_index(drop=True)

    train_ds = SeqDataset(tr_df, context_length=args.context_length)
    test_ds  = SeqDataset(te_df, context_length=args.context_length)
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
        pool='last',
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0  # ensures the first epoch always saves, so a checkpoint always exists
    outp = pathlib.Path(args.out_pt); outp.parent.mkdir(parents=True, exist_ok=True)
    arch = dict(
        d_in=D_IN, n_classes=5, embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks, num_heads=args.num_heads,
        context_length=args.context_length, pool='last',
    )

    for ep in range(args.epochs):
        model.train()
        for xb, yb, lb in train_dl:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            logits = model(xb, lb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); y_true = []; y_pred = []
        with torch.no_grad():
            for xb, yb, lb in test_dl:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                logits = model(xb, lb)
                pred = logits.argmax(dim=-1)
                y_true.append(yb.cpu().numpy()); y_pred.append(pred.cpu().numpy())
        Yt = np.concatenate(y_true, 0); Yp = np.concatenate(y_pred, 0)
        acc = accuracy(Yt, Yp); mf1 = macro_f1(Yt, Yp, 5)
        print(f"[epoch {ep:02d}] acc={acc:.3f} macro-F1={mf1:.3f} (val_n={len(Yt)})")

        if acc > best:
            best = acc
            save_checkpoint(model, str(outp), arch=arch)
            print(f"[OK] saved -> {outp}")

    print(f"[BEST] acc={best:.3f}")


if __name__ == "__main__":
    main()
