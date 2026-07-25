import argparse
import json
import os
import pathlib
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
)

LEVELS = [f"Level_{i}" for i in range(1, 6)]
CLASS_NAMES = ["LoA1", "LoA2", "LoA3", "LoA4", "LoA5"]  
FCD_NAMES = [
    "Safety Risk", "Increased Safety", "Relevance", "Magicality",
    "Privacy", "Trust", "Time Consumption", "Repetitiveness",
    "Situational Context", "Social Risk", "Urgency", "Complexity",
]


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def extract_frame_truth_pred(row: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[int], Optional[List[float]]]:
    y_true = None
    if all(k in row for k in LEVELS):
        try:
            y_true = np.array([int(float(row[k])) for k in LEVELS], dtype=np.int32)
        except Exception:
            y_true = None

    pred_label = None
    if "LoA" in row and row["LoA"] is not None:
        try:
            pred_label = int(row["LoA"])
        except Exception:
            pass
    if pred_label is None and "loa" in row and row["loa"] is not None:
        try:
            pred_label = int(row["loa"])
        except Exception:
            pass

    probs = None
    if "probs" in row and isinstance(row["probs"], (list, tuple)) and len(row["probs"]) == 5:
        try:
            probs = [float(x) for x in row["probs"]]
        except Exception:
            probs = None

    if pred_label is None or probs is None:
        la = row.get("last_action") or {}
        if isinstance(la, dict):
            if pred_label is None and "LoA" in la and la["LoA"] is not None:
                try:
                    pred_label = int(la["LoA"])
                except Exception:
                    pred_label = None
            if probs is None and "probs" in la and isinstance(la["probs"], (list, tuple)) and len(la["probs"]) == 5:
                try:
                    probs = [float(x) for x in la["probs"]]
                except Exception:
                    probs = None

    return (y_true, pred_label, probs)


def aggregate_by_segment(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for r in rows:
        sid = str(r.get("segment_id", "")).strip()
        if not sid:
            continue
        y_true, y_pred_label, y_pred_probs = extract_frame_truth_pred(r)
        fcd_dict = r.get("FCD") if isinstance(r.get("FCD"), dict) else (r.get("fcd") if isinstance(r.get("fcd"), dict) else None)

        recs.append({
            "segment_id": sid,
            "participantid": str(r.get("participantid", "")),
            "environment": str(r.get("environment", "")),
            "secondary_task": str(r.get("secondary_task", "")),
            "functionname": str(r.get("functionname", "")),
            "y_true_5": y_true,
            "y_pred_label": y_pred_label,
            "y_pred_probs": np.array(y_pred_probs, dtype=np.float64) if y_pred_probs is not None else None,
            "fcd_dict": fcd_dict,
        })

    if not recs:
        raise ValueError("No records available for evaluation (missing segment_id or file is empty)")

    df = pd.DataFrame(recs)

    groups = []
    for sid, g in df.groupby("segment_id"):
        g = g.reset_index(drop=True)
        meta = {
            "segment_id": sid,
            "participantid": g["participantid"].iloc[0],
            "environment": g["environment"].iloc[0],
            "secondary_task": g["secondary_task"].iloc[0],
            "functionname": g["functionname"].iloc[0],
            "n_frames": int(len(g)),
        }

        y_list = [y for y in g["y_true_5"].tolist() if isinstance(y, np.ndarray)]
        if y_list:
            y_sum = np.stack(y_list, axis=0).sum(axis=0)
            y_true_5 = (y_sum >= (len(y_list) / 2.0)).astype(int)
        else:
            continue


        probs_list = [p for p in g["y_pred_probs"].tolist() if isinstance(p, np.ndarray)]
        if probs_list:
            probs = np.mean(np.stack(probs_list, axis=0), axis=0)
        else:
            labels = [l for l in g["y_pred_label"].tolist() if isinstance(l, int)]
            if labels:
                ohs = []
                for l in labels:
                    oh = np.zeros(5, dtype=np.float64)
                    if 0 <= l < 5:
                        oh[l] = 1.0
                    ohs.append(oh)
                probs = np.mean(np.stack(ohs, axis=0), axis=0)
            else:
                probs = np.array([1.0, 0, 0, 0, 0], dtype=np.float64)

        y_pred = int(np.argmax(probs))
        meta.update({
            "y_true_5": y_true_5,
            "y_true": int(np.argmax(y_true_5)), 
            "y_pred_probs": probs,
            "y_pred": y_pred,
        })

        fcd_first = None
        for vv in g["fcd_dict"].tolist():
            if isinstance(vv, dict):
                fcd_first = vv
                break
        meta["fcd_dict"] = fcd_first

        groups.append(meta)

    seg_df = pd.DataFrame(groups)
    if seg_df.empty:
        raise ValueError("No valid segments after aggregation (possibly missing Level_* ground-truth labels)")
    return seg_df


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, out: pathlib.Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(5)); ax.set_yticklabels(CLASS_NAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_prf_bars(p: np.ndarray, r: np.ndarray, f1: np.ndarray, outdir: pathlib.Path, title_prefix: str):
    # Precision
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(5), p)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1); ax.set_title(f"{title_prefix} — Precision")
    fig.tight_layout(); fig.savefig(outdir / "prf_bars_precision.png", dpi=200); plt.close(fig)

    # Recall
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(5), r)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1); ax.set_title(f"{title_prefix} — Recall")
    fig.tight_layout(); fig.savefig(outdir / "prf_bars_recall.png", dpi=200); plt.close(fig)

    # F1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(5), f1)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1); ax.set_title(f"{title_prefix} — F1")
    fig.tight_layout(); fig.savefig(outdir / "prf_bars_f1.png", dpi=200); plt.close(fig)


def save_roc(seg_df: pd.DataFrame, out: pathlib.Path, title: str):
    if not seg_df["y_pred_probs"].notna().any():
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y_true_oh = np.stack(seg_df["y_true_5"].values, axis=0).astype(int)
    y_prob = np.stack(seg_df["y_pred_probs"].values, axis=0).astype(float)
    for k in range(5):
        fpr, tpr, _ = roc_curve(y_true_oh[:, k], y_prob[:, k])
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[k]} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title(title); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def save_pr(seg_df: pd.DataFrame, out: pathlib.Path, title: str):
    if not seg_df["y_pred_probs"].notna().any():
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y_true_oh = np.stack(seg_df["y_true_5"].values, axis=0).astype(int)
    y_prob = np.stack(seg_df["y_pred_probs"].values, axis=0).astype(float)
    for k in range(5):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, k], y_prob[:, k])
        ap = average_precision_score(y_true_oh[:, k], y_prob[:, k])
        ax.plot(recall, precision, label=f"{CLASS_NAMES[k]} AP={ap:.3f}")
    ax.set_title(title); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def save_seg_len_hist(seg_df: pd.DataFrame, out: pathlib.Path, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(seg_df["n_frames"].values, bins=20)
    ax.set_title(title); ax.set_xlabel("Frames per segment"); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def save_acc_by_category(seg_df: pd.DataFrame, cat: str, out: pathlib.Path, title: str, top_k: int = 20):
    g = seg_df.groupby(cat)
    xs, accs = [], []
    for k, dfk in g:
        xs.append(str(k))
        accs.append(accuracy_score(dfk["y_true"].values, dfk["y_pred"].values))
    order = np.argsort(accs)[::-1][:top_k]
    xs = [xs[i] for i in order]; accs = [accs[i] for i in order]
    fig = plt.figure(figsize=(max(6, len(xs) * 0.35), 4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(xs)), accs)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=45, ha="right")
    ax.set_ylim(0, 1); ax.set_title(title); ax.set_ylabel("Accuracy")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def save_fcd_correlation(seg_df: pd.DataFrame, out: pathlib.Path, title: str):
    if "fcd_dict" not in seg_df.columns or seg_df["fcd_dict"].isna().all():
        return
    fcd_rows, y = [], []
    for _, r in seg_df.iterrows():
        f = r["fcd_dict"]
        if not isinstance(f, dict):
            continue
        try:
            v = [float(f.get(k, np.nan)) for k in FCD_NAMES]
        except Exception:
            continue
        if any(np.isnan(v)):
            continue
        fcd_rows.append(v)
        y.append(int(r["y_pred"])) 
    if not fcd_rows:
        return
    X = np.array(fcd_rows, dtype=float) 
    Y = np.array(y, dtype=int)         
    corr = []
    for j in range(X.shape[1]):
        xj = X[:, j]
        if xj.std() < 1e-6:
            corr.append(0.0)
        else:
            c = np.corrcoef(xj, Y)[0, 1]
            corr.append(float(c))
    corr = np.array(corr, dtype=float)

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.reshape(1, -1), aspect="auto")
    ax.set_yticks([0]); ax.set_yticklabels(["corr(FCD, LoA)"])
    ax.set_xticks(range(len(FCD_NAMES))); ax.set_xticklabels(FCD_NAMES, rotation=45, ha="right")
    for j, v in enumerate(corr):
        ax.text(j, 0, f"{v:+.2f}", ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_jsonl", required=True, help="Per-frame JSONL (containing segment_id, Level_1..Level_5, predicted LoA/probs)")
    ap.add_argument("--outdir", default="reports/eval", help="Output directory")
    ap.add_argument("--title", default="LoA Evaluation", help="Chart title prefix")
    args = ap.parse_args()

    in_path = pathlib.Path(args.in_jsonl)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(in_path)
    if not rows:
        raise SystemError(" JSONL empty")

    seg_df = aggregate_by_segment(rows)

    y_true = seg_df["y_true"].values.astype(int)
    y_pred = seg_df["y_pred"].values.astype(int)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(5)), zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    per_seg = pd.DataFrame({
        "segment_id": seg_df["segment_id"],
        "participantid": seg_df["participantid"],
        "environment": seg_df["environment"],
        "secondary_task": seg_df["secondary_task"],
        "functionname": seg_df["functionname"],
        "n_frames": seg_df["n_frames"],
        "y_true": seg_df["y_true"],
        "y_pred": seg_df["y_pred"],
    })
    per_seg.to_csv(outdir / "per_segment.csv", index=False, encoding="utf-8-sig")

    func_acc = seg_df.groupby("functionname").apply(lambda d: accuracy_score(d["y_true"], d["y_pred"])).reset_index(name="accuracy")
    env_acc  = seg_df.groupby("environment").apply(lambda d: accuracy_score(d["y_true"], d["y_pred"])).reset_index(name="accuracy")
    func_acc.to_csv(outdir / "per_function.csv", index=False, encoding="utf-8-sig")
    env_acc.to_csv(outdir / "per_environment.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_segments": int(len(seg_df)),
        "micro": {"precision": float(micro[0]), "recall": float(micro[1]), "f1": float(micro[2])},
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "accuracy": float(acc),
        "per_class": [
            {"class": CLASS_NAMES[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i in range(5)
        ],
    }
    with open(outdir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_confusion(y_true, y_pred, outdir / "confusion_matrix.png", f"{args.title} — Confusion Matrix")
    save_prf_bars(p, r, f1, outdir, f"{args.title} — PRF per class")
    save_roc(seg_df, outdir / "roc_curves.png", f"{args.title} — ROC")
    save_pr(seg_df, outdir / "pr_curves.png", f"{args.title} — Precision-Recall")
    save_seg_len_hist(seg_df, outdir / "seg_len_hist.png", f"{args.title} — Segment length")
    save_acc_by_category(seg_df, "functionname", outdir / "acc_by_function.png", f"{args.title} — Accuracy by Function")
    save_acc_by_category(seg_df, "environment", outdir / "acc_by_environment.png", f"{args.title} — Accuracy by Environment")
    save_fcd_correlation(seg_df, outdir / "fcd_loa_correlation.png", f"{args.title} — FCD–LoA correlation")

    print(f"[OK] eval done. figures & tables saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
