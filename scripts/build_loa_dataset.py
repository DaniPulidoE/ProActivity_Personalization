"""Build a trainable LoA dataset by aligning logged frames to the driver's labels.

This is the missing link in the data pipeline. It joins:

  * per-frame features    -> ``raw_data.jsonl``      (written by ProVoice each cycle)
  * ground-truth LoA      -> ``user_loa_labels.csv`` (written by the drive UI every 20 s)

on ``session_id`` + the frame's wall-clock timestamp falling inside a label
window ``[window_start_timestamp, window_end_timestamp]``. The driver's
``user_selected_loa`` becomes the label (NOT the system's own predicted ``LoA``,
which would be circular).

A window can carry MORE THAN ONE label. Under ``--random-function`` the drive UI
shows two prompts per 20 s window, about two different functions, and both rows
share the window's index and its start/end. Every frame in such a window is
therefore emitted once PER LABEL: same features, different function and different
LoA. Taking only the first match would silently throw half the labels away.

The function a segment is attributed to comes from the LABEL row, not from the
frame. ProVoice is started with a single ``functionname`` and stamps it (and the
FCD vector it implies) on every frame it logs, which is wrong for every prompt
that drew a different function; the label row records what the driver was
actually asked about.

It writes two files, matching exactly what the trainers expect:

  * ``labeled_data.jsonl`` — every labelled frame + ``segment_id`` + ``Level_1..5``
      (feed to ``python -m ProVoice.train_XLSTM --in labeled_data.jsonl``)

      All raw frame fields are passed through verbatim, including:
        - xLSTM model features: perclos, gaze_score, hr_delta, rr_delta,
          blink_rate, yawn_rate, emotion, lab, environment, secondary_task
        - STATE_CARLA model features: speed_ratio_max, speed_ratio_limit,
          brake, steer, precipitation, is_night, is_junction
        - Logging extras (not used by model, kept for analysis):
          speed_kmh, speed_limit_kmh, throttle, gear, hand_brake, reverse,
          acceleration, fog_density, traffic_light_state, headlight,
          fog_light, left_indicator, right_indicator, eye_ar, mar

  * ``fcd_out.csv``        — per-segment aggregated FCD features + ``Level_1..5``
      (feed to ``ProVoice.train_fcd_loa`` / ``data/processed_data/fcd_out.csv``)

A "segment" is one label: ``segment_id = <session_id>|win<idx>`` for a window with
a single prompt, ``<session_id>|win<idx>p<n>`` once a window holds several.
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


def _parse_loa_set(raw: str) -> List[int]:
    """Parse ``user_selected_loa`` into a sorted list of distinct LoAs.

    The drive UI lets a driver mark EVERY acceptable level, writing them as
    ``"2;3"``. A single mark stays a bare ``"2"``, so labels recorded before
    multi-select still parse unchanged.
    """
    out = set()
    for part in str(raw).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(float(part))
        except ValueError:
            # Not a number at all. Dropping this silently used to be
            # indistinguishable from a deliberately blank label.
            print(f"[labels][warn] un-parseable LoA {part!r} in {raw!r} -- mark IGNORED")
            continue
        if not 0 <= value <= 4:
            # Still clamped (behaviour unchanged), but said out loud: an
            # out-of-range level means the drive UI wrote something the scale
            # does not define, and absorbing it into a valid level turns a UI
            # bug into plausible-looking ground truth.
            print(f"[labels][warn] LoA {value} outside 0..4 in {raw!r} -- "
                  f"CLAMPED to {max(0, min(4, value))}")
        out.add(max(0, min(4, value)))
    return sorted(out)


def _loa_to_levels(loas) -> Dict[str, int]:
    """Multi-hot over Level_1..5. One marked LoA gives the old one-hot."""
    if isinstance(loas, int):
        loas = [loas]
    levels = {k: 0 for k in LEVELS}
    for loa in loas:
        levels[f"Level_{max(0, min(4, int(loa))) + 1}"] = 1
    return levels


def _prompt_order(prompt: Any) -> int:
    try:
        return int(float(prompt))
    except (TypeError, ValueError):
        return 0


def _segment_id(sid: str, window_idx: Any, prompt: Any) -> str:
    """``<session>|win<NNN>``, plus ``p<n>`` once a window holds several prompts.

    The suffix is what keeps the prompts of one window apart: they share a
    window_idx and its 20 s span and differ only in the function asked about.
    Labels written before prompts existed carry no ``prompt_in_window``, so they
    keep the original id and datasets rebuilt from older CSVs are unchanged.
    """
    idx = str(window_idx)
    base = (f"{sid}|win{int(float(idx)):03d}" if idx.replace('.', '', 1).isdigit()
            else f"{sid}|win{idx}")
    order = _prompt_order(prompt)
    return f"{base}p{order}" if order else base


def load_label_windows(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """session_id -> list of labels (start/end seconds-of-day, loa, context).

    One entry per LABEL row, not per window: a window answered about two
    functions contributes two entries with identical start/end.
    """
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    kept = skipped = 0
    windows_seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("session_id") or "").strip()
            start_s = _secs_of_day(row.get("window_start_timestamp"))
            end_s = _secs_of_day(row.get("window_end_timestamp"))
            loa_raw = (row.get("user_selected_loa") or "").strip()
            if not sid or start_s is None or end_s is None or loa_raw == "":
                skipped += 1
                continue
            loa = _parse_loa_set(loa_raw)
            if not loa:
                skipped += 1
                continue
            window_idx = (row.get("window_idx") or "").strip() or str(len(by_session[sid]) + 1)
            prompt = (row.get("prompt_in_window") or "").strip()
            by_session[sid].append({
                "window_idx": window_idx,
                "prompt_in_window": prompt,
                # Empty for a run that fixed one function with --functionname; the
                # frame's own name is then used, and the two agree anyway.
                "functionname": (row.get("functionname") or "").strip(),
                "segment_id": _segment_id(sid, window_idx, prompt),
                "start_s": start_s,
                "end_s": end_s,
                "loa": loa,
            })
            windows_seen.add((sid, window_idx))
            kept += 1
    for sid in by_session:
        by_session[sid].sort(key=lambda w: (w["start_s"], _prompt_order(w["prompt_in_window"])))
    print(f"[labels] {kept} usable labels over {len(windows_seen)} windows "
          f"across {len(by_session)} sessions ({skipped} skipped)")
    return by_session


def _match_windows(windows: List[Dict[str, Any]], t: float) -> List[Dict[str, Any]]:
    """EVERY label whose window contains ``t`` — a window may carry more than one."""
    out = []
    for w in windows:
        s, e = w["start_s"], w["end_s"]
        inside = (s <= t <= e) if s <= e else (t >= s or t <= e)  # tolerate midnight wrap
        if inside:
            out.append(w)
    return out


# How many individual bad lines get their own message before the rest are only
# counted. One truncated tail line is the expected case; a flood means something
# else is wrong and the per-line detail stops helping.
_MAX_BAD_LINE_REPORTS = 5


def iter_raw_frames(path: Path, stats: Optional[Dict[str, int]] = None):
    """Yield parsed frames, COUNTING and REPORTING lines that fail to parse.

    This used to swallow decode errors with a bare ``except: continue``. That is
    a silent-data-loss path, not a robustness measure: ``raw_data.jsonl`` is
    written in append mode and the collector is killed at session end, so a
    half-written final line is an ordinary outcome. Worse, ``n_total`` counts
    only lines that parsed, so nothing in the summary could ever reveal the loss
    -- frames would simply be missing from a training segment.

    ``stats['bad_lines']`` is set once the generator is exhausted.
    """
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                bad += 1
                if bad <= _MAX_BAD_LINE_REPORTS:
                    ell = "..." if len(line) > 80 else ""
                    print(f"[raw][warn] {path.name} line {lineno}: unparseable JSON "
                          f"({e}) -- frame DROPPED: {line[:80]}{ell}")
                elif bad == _MAX_BAD_LINE_REPORTS + 1:
                    print("[raw][warn] ... further unparseable lines are counted "
                          "but not printed individually")
                continue
            yield obj
    if stats is not None:
        stats["bad_lines"] = bad


def _fcd_vector(frame: Dict[str, Any], functionname: str = "") -> List[float]:
    """FCD features for one frame.

    ``functionname`` is the function the LABEL asked about, and it wins over the
    frame's own ``FCD``: ProVoice derived that from the single function it was
    started with, so it describes the wrong function for every prompt that drew a
    different one.
    """
    if functionname:
        fcd = get_fcd_for_function(functionname)
    else:
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

    # Only a COUNT per segment, never the frames themselves. The per-segment FCD
    # vector is a pure function of the segment's functionname (FCD is static per
    # function), so the old per-frame average was the mean of N identical
    # vectors -- and retaining every in-window frame to compute it was the
    # builder's only unbounded memory cost, growing linearly with the study.
    seg_n_frames: Dict[str, int] = defaultdict(int)
    seg_meta: Dict[str, Dict[str, Any]] = {}
    n_total = n_labeled = n_no_window = 0
    # Three distinct failures that used to share one counter, so a timestamp that
    # stopped parsing would be reported as a session-matching problem.
    n_no_sid = n_unknown_session = n_bad_timestamp = 0
    raw_stats: Dict[str, int] = {"bad_lines": 0}
    n_frames_used = n_multi = 0
    loa_hist: Dict[int, int] = defaultdict(int)

    with out_jsonl.open("w", encoding="utf-8") as fo:
        for frame in iter_raw_frames(raw_path, raw_stats):
            n_total += 1
            sid = (frame.get("session_id") or "").strip()
            t = _secs_of_day(frame.get("timestamp"))
            if not sid:
                n_no_sid += 1
                continue
            if sid not in windows:
                n_unknown_session += 1
                continue
            if t is None:
                n_bad_timestamp += 1
                continue
            matches = _match_windows(windows[sid], t)
            if not matches:
                n_no_window += 1
                continue
            n_frames_used += 1
            if len(matches) > 1:
                n_multi += 1
            # One row per label: the same 20 s of driving answered about two
            # functions is two training examples, identical features apart from
            # the function and the LoA.
            for w in matches:
                row = dict(frame)
                row["segment_id"] = w["segment_id"]
                row["user_loa"] = ";".join(str(v) for v in w["loa"])
                if w["functionname"] and w["functionname"] != str(frame.get("functionname", "")):
                    # What the trainer encodes from: encode_frame() rebuilds the
                    # FCD vector from this name, so it has to be the function the
                    # driver was actually asked about, not ProVoice's run-level
                    # one. Its FCD goes with it — a frame carrying one function's
                    # name next to another's FCD is a trap for anything reading
                    # the field directly. Untouched when the two already agree,
                    # so datasets rebuilt from older labels stay byte-identical.
                    row["functionname"] = w["functionname"]
                    row["FCD"] = get_fcd_for_function(w["functionname"])
                row.update(_loa_to_levels(w["loa"]))
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_labeled += 1
                for _loa in w["loa"]:   # a multi-mark counts toward every level it names
                    loa_hist[_loa] += 1
                seg_n_frames[w["segment_id"]] += 1
                if w["segment_id"] not in seg_meta:
                    seg_meta[w["segment_id"]] = {
                        "loa": w["loa"],
                        "functionname": w["functionname"] or str(frame.get("functionname", "")),
                        # Only consulted when NO functionname is known (labels
                        # written before the column existed); _fcd_vector then
                        # falls back to the frame's own logged FCD. Kept from the
                        # FIRST frame of the segment so that path survives
                        # without retaining the other ~380.
                        "fcd_fallback": frame.get("FCD") or frame.get("fcd"),
                    }

    # Per-segment aggregated FCD table for the FCD trainer.
    n_seg = 0
    with out_fcd.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(LEVELS + FEATS + ["function_group", "segment_id", "n_frames"])
        for seg_id, meta in seg_meta.items():
            # One vector per SEGMENT, not per frame: identical by construction,
            # so this is exactly what the average produced. A NaN dimension (one
            # the FCD config could not supply) still falls back to the neutral 3,
            # and the 1..5 clamp is retained.
            probe = {"FCD": meta.get("fcd_fallback"), "functionname": meta["functionname"]}
            vec = _fcd_vector(probe, meta["functionname"])
            agg = [max(1, min(5, int(round(v)))) if v == v else 3 for v in vec]
            levels = _loa_to_levels(meta["loa"])
            w.writerow([levels[k] for k in LEVELS] + agg
                       + [meta["functionname"], seg_id, seg_n_frames[seg_id]])
            n_seg += 1

    print(f"[frames] {n_total} read | {n_frames_used} inside a window | "
          f"{n_no_window} outside-any-window "
          f"(the scene is frozen while a popup is open, and no window covers that time)")
    print(f"[frames][skipped] unparseable-json-line={raw_stats['bad_lines']} | "
          f"blank-session-id={n_no_sid} | session-not-in-labels={n_unknown_session} | "
          f"unparseable-timestamp={n_bad_timestamp}")
    if raw_stats["bad_lines"] or n_no_sid or n_bad_timestamp:
        print("[frames][warn] the counters above should be ZERO on a healthy run "
              "-- a non-zero value means frames were DROPPED, not merely unlabelled")
    print(f"[rows] {n_labeled} labelled rows written "
          f"({n_multi} frames carried more than one label)")
    print(f"[segments] {n_seg} -> {out_fcd}")
    print(f"[labelled frames] -> {out_jsonl}")
    print("[LoA distribution] " + ", ".join(f"LoA{k}={loa_hist[k]}" for k in sorted(loa_hist)))
    by_function: Dict[str, int] = defaultdict(int)
    for meta in seg_meta.values():
        by_function[meta["functionname"] or "(unnamed)"] += 1
    print("[segments by function] " + ", ".join(
        f"{name}={n}" for name, n in sorted(by_function.items(), key=lambda kv: -kv[1])))
    if n_labeled == 0:
        raise SystemExit("nothing aligned — check that drive & ProVoice shared the same session_id "
                         "(PV_SESSION_ID / --session-id) and ran on the same machine clock.")


if __name__ == "__main__":
    main()
