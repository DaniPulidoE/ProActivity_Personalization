"""
Merge JSONL files into two CSVs (FCD-view & State-view), using Level_1..Level_5
instead of a single LoA column. Defaults use filename as function_group.

Input row example (per line JSON):
{
  "emotion": "neutral",
  "lab": "face",
  "drowsiness_alert": false,
  "gaze_distracted": true,
  "bpm": 86,
  "FCD": { ... 12 dims ... },
  "Level_1": 0, "Level_2": 0, "Level_3": 1, "Level_4": 0, "Level_5": 0,
  "LoA": 2
}

Usage: tweak constants at top or run as-is.
"""
import csv, json, pathlib
from typing import Dict, Any, Iterable

# ================= user settings =================
INPUT_DIR = pathlib.Path('data/saved_extracted_data')  # folder to scan
PATTERN = '*.jsonl'                               # file pattern
FCD_OUT = pathlib.Path('data/processed_data/fcd_out.csv')
STATE_OUT = pathlib.Path('data/processed_data/state_out.csv')
RECURSIVE = False                                 # set True to recurse subdirs
# =================================================

FCD_NAMES = [
    'Safety Risk','Increased Safety','Relevance','Magicality','Privacy','Trust',
    'Time Consumption','Repetitiveness','Situational Context','Social Risk','Urgency','Complexity'
]
LEVEL_KEYS = [f'Level_{i}' for i in range(1, 6)]


def iter_files(input_dir: pathlib.Path, pattern: str, recursive: bool):
    if recursive:
        yield from (p for p in input_dir.rglob(pattern) if p.is_file())
    else:
        yield from (p for p in input_dir.glob(pattern) if p.is_file())


def load_jsonl(path: pathlib.Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[Error] Failed to parse JSON line: {e}")
                continue


def coerce_hr(row: Dict[str, Any]):
    v = row.get('heart_rate', row.get('bpm'))
    try:
        return float(v)
    except Exception as e:
        print(f"[Error] Failed to convert to float: {e}")
        return None


def as_bool01(x):
    if isinstance(x, bool):
        return int(x)
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ('true','1','t','yes','y'):
        return 1
    if s in ('false','0','f','no','n'):
        return 0
    return None


def get_levels(row: Dict[str, Any]):
    # Prefer explicit Level_1..5 if present; otherwise build one-hot from LoA
    levels = [row.get(k) for k in LEVEL_KEYS]
    if any(l is not None for l in levels):
        out = []
        for v in levels:
            try:
                out.append(int(v))
            except Exception as e:
                print(f"[Error] Failed to convert Level value to int: {e}")
                out.append(0)
        # sanity clamp to {0,1}
        out = [1 if x else 0 for x in out]
        return out
    # fallback from LoA
    loa = row.get('LoA')
    try:
        idx = int(loa)
    except Exception as e:
        print(f"[Error] Failed to convert LoA to int: {e}")
        return None
    if loa is None:
        return None
    idx = max(0, min(4, idx))
    out = [0,0,0,0,0]
    out[idx] = 1
    return out


def main():
    input_dir = INPUT_DIR
    if not input_dir.exists():
        raise SystemExit(f'Input dir not found: {input_dir}')

    files = list(iter_files(input_dir, PATTERN, RECURSIVE))
    if not files:
        raise SystemExit('No files matched.')

    FCD_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATE_OUT.parent.mkdir(parents=True, exist_ok=True)

    with FCD_OUT.open('w', newline='', encoding='utf-8') as f_fcd, \
         STATE_OUT.open('w', newline='', encoding='utf-8') as f_state:
        wf = csv.writer(f_fcd)
        ws = csv.writer(f_state)
        # headers: replace LoA with Level_1..Level_5
        wf.writerow(LEVEL_KEYS + [f'Feature_{i}' for i in range(1,13)] + ['function_group'])
        ws.writerow(LEVEL_KEYS + ['emotion','lab','drowsiness_alert','gaze_distracted','heart_rate','function_group'])

        for p in files:
            func_group = p.stem  # default group name from filename
            for row in load_jsonl(p):
                levels = get_levels(row)
                if levels is None:
                    # skip rows without Level_* and LoA
                    continue
                # ----- FCD view -----
                fcd_dict = row.get('FCD') or {}
                feats = []
                ok = True
                for name in FCD_NAMES:
                    v = fcd_dict.get(name)
                    if v is None:
                        ok = False
                        break
                    try:
                        feats.append(float(v))
                    except Exception as e:
                        print(f"[Error] Failed to convert FCD value to float: {e}")
                        ok = False
                        break
                if ok:
                    wf.writerow(levels + feats + [func_group])

                # ----- State view -----
                ws.writerow(levels + [
                    row.get('emotion'),
                    row.get('lab'),
                    as_bool01(row.get('drowsiness_alert')),
                    as_bool01(row.get('gaze_distracted')),
                    coerce_hr(row),
                    func_group
                ])

    print('[OK] merged:', FCD_OUT, 'and', STATE_OUT)

if __name__ == '__main__':
    main()
