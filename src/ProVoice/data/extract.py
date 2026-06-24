# Aggregate in windows of n records: categories use mode; bpm/LoA/FCD use mean rounded to int (LoA∈[0,4]; FCD∈[1,5]).
# Supports two Level encodings:
#   - onehot (default): only the Level_k corresponding to the selected LoA is set to 1 (k=1..5), rest are 0
#   - neighbor_up/neighbor_down: same as onehot but also sets the adjacent upper/lower level to 1
# Input supports JSONL or JSON arrays. Output is JSONL (one aggregated record per line).
from __future__ import annotations
import json, pathlib, os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

FCD_KEYS_CANON = [
    'Safety Risk','Increased Safety','Relevance','Magicality','Privacy','Trust',
    'Time Consumption','Repetitiveness','Situational Context','Social Risk','Urgency','Complexity'
]

# ============= Configurable =============
INPUT_PATH = pathlib.Path('data/raw_data.json')          # Input file path (JSONL or JSON array)
OUTPUT_PATH = pathlib.Path('data/saved_extracted_data/extracted_1.jsonl')     # Output JSONL
CHUNK_N = 20                                        # Number of samples per group n
DROP_INCOMPLETE = False                              # Drop groups missing LoA or FCD
LEVEL_ENCODING = 'onehot'                            # 'onehot' | 'neighbor_up' | 'neighbor_down'
KEEP_LOA_NUMERIC = True                              # Whether to also keep the numeric LoA field
# ========================================

# ---------------- I/O ----------------

def load_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding='utf-8')
    text_strip = text.strip()
    recs: List[Dict[str, Any]] = []
    if text_strip.startswith('['):
        data = json.loads(text_strip)
        if isinstance(data, list):
            recs = [x for x in data if isinstance(x, dict)]
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    recs.append(obj)
            except Exception as e:
                print(f"[Error] Failed to process line: {e}")
                continue
    return recs


def save_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = '\r\n' if os.name == 'nt' else '\n'
    with path.open('w', encoding='utf-8', newline='') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write(sep)

# ---------------- util ----------------

def to_float(x: Any) -> float | None:
    try:
        if x is None or x == '':
            return None
        v = float(x)
        if v != v:  # NaN check (NaN is the only float not equal to itself)
            return None
        return v
    except Exception as e:
        print(f"[Error] Failed to convert to float: {e}")
        return None


def to_bool(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ('true','1','t','yes','y'): return True
    if s in ('false','0','f','no','n'): return False
    return None


def mode(values: Iterable[Any]) -> Any:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

# ---------------- field getters ----------------

def get_bpm(r: Dict[str, Any]) -> float | None:
    v = to_float(r.get('bpm'))
    if v is None:
        v = to_float(r.get('heart_rate'))
    return v


def get_loa(r: Dict[str, Any]) -> float | None:
    v = to_float(r.get('LoA'))
    if v is not None:
        return v
    last = r.get('last_action')
    if isinstance(last, dict):
        v = to_float(last.get('LoA'))
        if v is not None:
            return v
    return None


def get_fcd(r: Dict[str, Any]) -> Dict[str, float] | None:
    src = None
    if isinstance(r.get('FCD'), dict):
        src = r['FCD']
    elif isinstance(r.get('last_action'), dict) and isinstance(r['last_action'].get('fcd'), dict):
        src = r['last_action']['fcd']
    if not isinstance(src, dict):
        return None
    out: Dict[str, float] = {}
    for k in FCD_KEYS_CANON:
        v = to_float(src.get(k))
        if v is not None:
            out[k] = v
    return out if out else None

# ---------------- LoA → Levels ----------------

def loa_to_levels(loa_int: int, encoding: str = 'onehot') -> dict:
    # Level_1..Level_5 correspond to LoA 0..4
    levels = {f'Level_{i}': 0 for i in range(1, 6)}
    idx = clamp(int(loa_int), 0, 4)
    levels[f'Level_{idx+1}'] = 1
    if encoding == 'neighbor_up' and idx < 4:
        levels[f'Level_{idx+2}'] = 1
    elif encoding == 'neighbor_down' and idx > 0:
        levels[f'Level_{idx}'] = 1
    return levels

# ---------------- aggregate one chunk ----------------

def aggregate_chunk(chunk: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not chunk:
        return None

    # Categorical/boolean mode
    emotion = mode([c.get('emotion') for c in chunk]) or 'neutral'
    lab = mode([c.get('lab') for c in chunk])
    drowsy = mode([to_bool(c.get('drowsiness_alert')) for c in chunk])
    gaze = mode([to_bool(c.get('gaze_distracted')) for c in chunk])

    # BPM mean (rounded to int)
    bpms = [get_bpm(c) for c in chunk]
    bpms = [v for v in bpms if v is not None]
    bpm_avg = int(round(sum(bpms)/len(bpms))) if bpms else None

    # LoA mean → round and clamp to [0,4]
    loas = [get_loa(c) for c in chunk]
    loas = [v for v in loas if v is not None]
    loa_avg = clamp(int(round(sum(loas)/len(loas))) if loas else 0, 0, 4)

    # FCD per-dimension mean → round and clamp to [1,5]
    fcd_acc: Dict[str, List[float]] = defaultdict(list)
    for c in chunk:
        fcd = get_fcd(c)
        if not fcd:
            continue
        for k, v in fcd.items():
            fcd_acc[k].append(v)
    fcd_out: Dict[str, int] = {}
    for k in FCD_KEYS_CANON:
        vals = fcd_acc.get(k, [])
        if vals:
            fcd_out[k] = clamp((int(round(sum(vals)/len(vals)))), 1, 5)

    # Assemble output: Level_* replaces the single LoA field (optionally keep numeric LoA)
    rec: Dict[str, Any] = {
        'emotion': emotion,
        'lab': lab,
        'drowsiness_alert': drowsy,
        'gaze_distracted': gaze,
        'bpm': bpm_avg,
        'FCD': fcd_out,
    }
    rec.update(loa_to_levels(loa_avg, LEVEL_ENCODING))
    if KEEP_LOA_NUMERIC:
        rec['LoA'] = loa_avg
    return rec

# ---------------- main ----------------

def main():
    records = load_records(INPUT_PATH)
    if not records:
        raise SystemExit(f'No records loaded from {INPUT_PATH}')

    n = max(1, int(CHUNK_N))
    agg_rows: List[Dict[str, Any]] = []

    for i in range(0, len(records), n):
        chunk = records[i:i+n]
        agg = aggregate_chunk(chunk)
        if agg is None:
            continue
        if DROP_INCOMPLETE:
            # Drop if numeric LoA is missing (when required) or FCD is absent
            if (KEEP_LOA_NUMERIC and agg.get('LoA') is None) or not agg.get('FCD'):
                continue
        agg_rows.append(agg)

    save_jsonl(OUTPUT_PATH, agg_rows)
    print(f'[OK] wrote {len(agg_rows)} aggregated rows → {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
