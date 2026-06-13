"""Build a trainable LoA dataset by aligning logged frames to the driver's labels.

This is the missing link in the data pipeline. It joins:

  * per-frame features    -> ``raw_data.jsonl``      (written by ProVoice each cycle)
  * ground-truth LoA      -> ``user_loa_labels.csv`` (written by the drive UI every 20 s)

on ``session_id`` + the frame's wall-clock timestamp falling inside a label
window ``[window_start_timestamp, window_end_timestamp]``. The driver's
``user_selected_loa`` becomes the label (NOT the system's own predicted ``LoA``,
which would be circular).

It writes two files, matching exactly what the trainers expect:

  * ``labeled_data.jsonl`` — every labelled frame + ``segment_id`` + ``Level_1..5``
      (feed to ``python -m ProVoice.train_XLSTM --in labeled_data.jsonl``)
  * ``fcd_out.csv``        — per-segment aggregated FCD features + ``Level_1..5``
      (feed to ``ProVoice.train_fcd_loa`` / ``data/processed_data/fcd_out.csv``)

A "segment" is one 20 s label window (``segment_id = <session_id>|win<idx>``).
``Level_k`` is the one-hot of the driver's LoA: LoA 0..4 -> Level_1..5.

Usage::

    python scripts/build_loa_dataset.py \
        --raw data/raw_data.jsonl \
        --labels data/user_loa_labels.csv \
        --out-jsonl data/labeled_data.jsonl \
        --out-fcd data/processed_data/fcd_out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function  # noqa: E402

LEVELS = [f"Level_{i}" for i in range(1, 6)]
FEATS = [f"Feature_{i}" for i in range(1, 13)]


def _secs_of_day(ts: Optional[str]) -> Optional[float]:
    """Parse a timestamp to seconds-since-midnight, tolerant of several formats.

    Accepts ``HH:MM:SS.ffffff`` (ProVoice frame timestamps) and full ISO
    datetimes ``YYYY-MM-DDTHH:MM:SS.ffffff`` (drive label windows).
    """
    if not ts:
        return None
    ts = str(ts).strip()
    t: Optional[time] = None
    try:
        t = datetime.fromisoformat(ts).time()
    except Exception:
        for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
            try:
                t = datetime.strptime(ts, fmt).time()
                break
            except Exception:
                continue
    if t is None:
        return None
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def _loa_to_levels(loa: int) -> Dict[str, int]:
    levels = {k: 0 for k in LEVELS}
    idx = max(0, min(4, int(loa)))
    levels[f"Level_{idx + 1}"] = 1
    return levels


def load_label_windows(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """session_id -> list of windows (start/end seconds-of-day, loa, context)."""
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    kept = skipped = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("session_id") or "").strip()
            start_s = _secs_of_day(row.get("window_start_timestamp"))
            end_s = _secs_of_day(row.get("window_end_timestamp"))
            loa_raw = (row.get("user_selected_loa") or "").strip()
            if not sid or start_s is None or end_s is None or loa_raw == "":
                skipped += 1
                continue
            try:
                loa = int(float(loa_raw))
            except Exception:
                skipped += 1
                continue
            by_session[sid].append({
                "window_idx": (row.get("window_idx") or "").strip() or str(len(by_session[sid]) + 1),
                "start_s": start_s,
                "end_s": end_s,
                "loa": loa,
            })
            kept += 1
    for sid in by_session:
        by_session[sid].sort(key=lambda w: w["start_s"])
    print(f"[labels] {kept} usable windows across {len(by_session)} sessions ({skipped} skipped)")
    return by_session


def _match_window(windows: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    for w in windows:
        s, e = w["start_s"], w["end_s"]
        inside = (s <= t <= e) if s <= e else (t >= s or t <= e)  # tolerate midnight wrap
        if inside:
            return w
    return None


def iter_raw_frames(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _fcd_vector(frame: Dict[str, Any]) -> List[float]:
    fcd = frame.get("FCD") or frame.get("fcd")
    if not isinstance(fcd, dict):
        fcd = get_fcd_for_function(str(frame.get("functionname", "")))
    out = []
    for name in FCD_NAMES:
        try:
            out.append(float(fcd.get(name)))
        except Exception:
            out.append(float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw_data.jsonl", help="ProVoice per-frame log")
    ap.add_argument("--labels", default="data/user_loa_labels.csv", help="driver LoA labels (drive UI)")
    ap.add_argument("--out-jsonl", default="data/labeled_data.jsonl", help="per-frame labelled output (train_XLSTM)")
    ap.add_argument("--out-fcd", default="data/processed_data/fcd_out.csv", help="per-segment FCD output (train_fcd_loa)")
    args = ap.parse_args()

    raw_path, lbl_path = Path(args.raw), Path(args.labels)
    if not raw_path.exists():
        raise SystemExit(f"raw frames not found: {raw_path}")
    if not lbl_path.exists():
        raise SystemExit(f"labels not found: {lbl_path}")

    windows = load_label_windows(lbl_path)
    if not windows:
        raise SystemExit("no usable label windows — was user_loa_labels.csv populated by the drive UI?")

    out_jsonl = Path(args.out_jsonl); out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_fcd = Path(args.out_fcd); out_fcd.parent.mkdir(parents=True, exist_ok=True)

    seg_frames: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seg_meta: Dict[str, Dict[str, Any]] = {}
    n_total = n_labeled = n_no_session = n_no_window = 0
    loa_hist: Dict[int, int] = defaultdict(int)

    with out_jsonl.open("w", encoding="utf-8") as fo:
        for frame in iter_raw_frames(raw_path):
            n_total += 1
            sid = (frame.get("session_id") or "").strip()
            t = _secs_of_day(frame.get("timestamp"))
            if not sid or sid not in windows or t is None:
                n_no_session += 1
                continue
            w = _match_window(windows[sid], t)
            if w is None:
                n_no_window += 1
                continue
            seg_id = f"{sid}|win{int(float(w['window_idx'])):03d}" if str(w["window_idx"]).replace('.', '', 1).isdigit() else f"{sid}|win{w['window_idx']}"
            row = dict(frame)
            row["segment_id"] = seg_id
            row["user_loa"] = w["loa"]
            row.update(_loa_to_levels(w["loa"]))
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_labeled += 1
            loa_hist[w["loa"]] += 1
            seg_frames[seg_id].append(frame)
            seg_meta[seg_id] = {"loa": w["loa"], "functionname": str(frame.get("functionname", ""))}

    # Per-segment aggregated FCD table for the FCD trainer.
    n_seg = 0
    with out_fcd.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(LEVELS + FEATS + ["function_group", "segment_id", "n_frames"])
        for seg_id, frames in seg_frames.items():
            vecs = [_fcd_vector(fr) for fr in frames]
            agg = []
            for j in range(len(FCD_NAMES)):
                col = [v[j] for v in vecs if v[j] == v[j]]  # drop NaN
                agg.append(int(round(sum(col) / len(col))) if col else 3)
            agg = [max(1, min(5, v)) for v in agg]
            levels = _loa_to_levels(seg_meta[seg_id]["loa"])
            w.writerow([levels[k] for k in LEVELS] + agg + [seg_meta[seg_id]["functionname"], seg_id, len(frames)])
            n_seg += 1

    print(f"[frames] {n_total} read | {n_labeled} labelled | "
          f"{n_no_session} no-session-match | {n_no_window} outside-any-window")
    print(f"[segments] {n_seg} -> {out_fcd}")
    print(f"[labelled frames] -> {out_jsonl}")
    print("[LoA distribution] " + ", ".join(f"LoA{k}={loa_hist[k]}" for k in sorted(loa_hist)))
    if n_labeled == 0:
        raise SystemExit("nothing aligned — check that drive & ProVoice shared the same session_id "
                         "(PV_SESSION_ID / --session-id) and ran on the same machine clock.")


if __name__ == "__main__":
    main()
