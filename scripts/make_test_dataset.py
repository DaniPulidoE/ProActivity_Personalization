"""Generate CLEARLY-LABELLED synthetic TEST data to exercise the dataset pipeline.

This is *fake* data for end-to-end testing of build_loa_dataset.py -> the
trainers. It is NOT real experiment data:

  * session ids are prefixed ``TESTDATA_``
  * participant ids are ``TEST_P*``
  * everything is written under ``data/testdata/`` (never the real data/ files)
  * timestamps use the year 2099 so they can't be confused with real sessions

It mimics the two files a real experiment produces:
  * raw_data.jsonl       (ProVoice per-frame log, face present)
  * user_loa_labels.csv  (drive UI, one LoA selection per 20 s window)

Run::  python scripts/make_test_dataset.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ProVoice.fcd_config import get_fcd_for_function  # noqa: E402

OUT_DIR = Path("data/testdata")
LABEL_COLUMNS = [
    "session_id", "window_idx", "window_start_ms", "window_end_ms",
    "window_start_timestamp", "window_end_timestamp", "selection_timestamp",
    "selection_frame", "selection_sim_time", "selection_speed_kmh",
    "participantid", "environment", "secondary_task", "functionname", "emotion",
    "modeltype", "state_model", "w_fcd", "user_selected_loa", "system_action",
    "system_level", "system_loa", "system_message", "system_probs",
    "system_profile", "system_fallback", "system_fallback_reason", "system_fcd",
]
WINDOW_S = 20
FRAMES_PER_WINDOW = 20
EMOTIONS = ["neutral", "happy", "sad", "surprise", "fear"]

# (session_id, participantid, base_start, [(functionname, environment, secondary_task, user_loa), ...])
SESSIONS = [
    ("TESTDATA_SESSION_A", "TEST_P01", datetime(2099, 1, 1, 10, 0, 0), [
        ("Adjust seat positioning", "city", "none", 0),
        ("Send a text message", "city", "phone", 1),
        ("Overtake vehicle ahead", "highway", "none", 2),
        ("Change song", "city", "none", 3),
        ("Navigation control", "highway", "none", 4),
        ("Start a phone call", "city", "phone", 1),
        ("Provide weather update", "highway", "none", 2),
        ("Adjust in-car temperature", "city", "none", 3),
        ("Change driving mode", "highway", "none", 4),
        ("Select parking space", "city", "none", 0),
        ("Start a movie", "city", "phone", 1),
        ("Provide traffic news", "highway", "none", 2),
    ]),
    ("TESTDATA_SESSION_B", "TEST_P02", datetime(2099, 1, 2, 14, 30, 0), [
        ("Adjust seat positioning", "highway", "none", 4),
        ("Send a text message", "city", "phone", 3),
        ("Overtake vehicle ahead", "highway", "none", 2),
        ("Change song", "city", "none", 1),
        ("Navigation control", "city", "none", 0),
        ("Start a phone call", "highway", "phone", 4),
        ("Provide weather update", "city", "none", 3),
        ("Adjust in-car temperature", "highway", "none", 2),
        ("Change driving mode", "city", "none", 1),
        ("Select parking space", "highway", "none", 0),
        ("Start a movie", "city", "phone", 4),
        ("Provide traffic news", "highway", "none", 2),
    ]),
]


def _ts_frame(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S.%f")[:-3]


def main() -> None:
    random.seed(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "raw_data.jsonl"
    lbl_path = OUT_DIR / "user_loa_labels.csv"

    n_frames = n_windows = 0
    with raw_path.open("w", encoding="utf-8") as fraw, \
         lbl_path.open("w", encoding="utf-8", newline="") as flbl:
        lw = csv.DictWriter(flbl, fieldnames=LABEL_COLUMNS)
        lw.writeheader()
        for sid, pid, base, windows in SESSIONS:
            for widx, (fn, env, task, loa) in enumerate(windows, start=1):
                w_start = base + timedelta(seconds=(widx - 1) * WINDOW_S)
                w_end = w_start + timedelta(seconds=WINDOW_S)
                lw.writerow({
                    "session_id": sid, "window_idx": widx,
                    "window_start_ms": (widx - 1) * WINDOW_S * 1000,
                    "window_end_ms": widx * WINDOW_S * 1000,
                    "window_start_timestamp": w_start.isoformat(),
                    "window_end_timestamp": w_end.isoformat(),
                    "selection_timestamp": (w_end + timedelta(seconds=2)).isoformat(timespec="milliseconds"),
                    "participantid": pid, "environment": env, "secondary_task": task,
                    "functionname": fn, "modeltype": "combined", "state_model": "xlstm",
                    "w_fcd": 0.7, "user_selected_loa": loa,
                })
                n_windows += 1
                fcd = get_fcd_for_function(fn)
                for k in range(FRAMES_PER_WINDOW):
                    # frame timestamp spread across the 20 s window (well inside it)
                    ft = w_start + timedelta(seconds=1 + k * (WINDOW_S - 2) / FRAMES_PER_WINDOW)
                    lab = ["face"] + (["phone"] if (task == "phone" and random.random() < 0.5) else [])
                    drowsy = random.random() < 0.1
                    frame = {
                        "session_id": sid, "participantid": pid, "environment": env,
                        "secondary_task": task, "functionname": fn,
                        "modeltype": "combined", "state_model": "xlstm", "w_fcd": 0.7,
                        "timestamp": _ts_frame(ft),
                        "emotion": random.choice(EMOTIONS),
                        "emotion_prob": round(random.uniform(0.3, 0.9), 3),
                        "eye_ar": round(random.gauss(0.30, 0.03), 3),
                        "mar": round(abs(random.gauss(0.02, 0.01)), 3),
                        "gaze_score": round(abs(random.gauss(0.13, 0.02)), 3),
                        "gaze_distracted": random.random() < 0.2,
                        "drowsiness_alert": drowsy,
                        "blink_count": k, "yawn_count": 0,
                        "perclos": round(random.uniform(0.0, 0.3), 3),
                        "lab": lab,
                        "heart_rate": (round(random.uniform(60, 90), 1) if random.random() < 0.5 else None),
                        "speed": random.randint(0, 90),
                        "FCD": dict(fcd),
                        "LoA": random.randint(0, 4),  # system prediction (ignored by alignment)
                    }
                    fraw.write(json.dumps(frame, ensure_ascii=False) + "\n")
                    n_frames += 1

    (OUT_DIR / "_TESTDATA_README.txt").write_text(
        "SYNTHETIC TEST DATA - not real experiment data.\n"
        "Generated by scripts/make_test_dataset.py to test scripts/build_loa_dataset.py.\n"
        "Session ids are prefixed TESTDATA_, participants are TEST_P*, year is 2099.\n",
        encoding="utf-8",
    )
    print(f"[testdata] wrote {n_frames} frames / {n_windows} windows")
    print(f"  raw    -> {raw_path}")
    print(f"  labels -> {lbl_path}")


if __name__ == "__main__":
    main()
