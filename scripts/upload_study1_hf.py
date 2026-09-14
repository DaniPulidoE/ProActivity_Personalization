"""Package the study-1 population dataset and upload it to the Hugging Face Hub.

Implements ``docs/study1_dataset_schema.md`` -- that document is the contract:
which files are released, under which config name, with which columns, in
which type, and which columns are deliberately excluded. The goal of the
release is **full reproducibility**: someone who downloads it must be able to
run the project's documented commands, unchanged, and regenerate every derived
file and every model. That fixes three design choices:

* **Formats are the pipeline's formats.** JSONL for frames, CSV for labels --
  exactly what ``heart_rate_preprocessing.py``, ``build_loa_dataset.py`` and
  ``train_XLSTM.iter_jsonl`` read. Column types are declared in the dataset
  card (``dataset_info.features``) so ``load_dataset`` does not have to infer
  them -- inference fails on these files (``hr_repair_method`` is null for the
  first 12k rows; ``is_night``/``is_junction`` are bool/int mixed).
* **Every input of the chain ships.** ``frames_raw`` alone cannot regenerate
  ``frames_preprocessed``: the HR repair needs each driver's calibration
  baseline and per-tick calibration log, so ``calibration/`` is part of the
  release.
* **Verification is re-running the chain.** ``--verify-chain`` regenerates
  ``frames_preprocessed`` and ``labeled`` from the STAGED inputs with the
  project's own scripts and diffs them against the staged outputs, and
  ``normalize_row`` is applied in lockstep to the original and the filtered
  ``labeled`` file to prove the trainers see identical rows. The card records
  the code commit and a SHA-256 for every file.

Token: ``HF_TOKEN=hf_...`` in ``.env`` at the project root (git-ignored), or the
``HF_TOKEN`` environment variable.

Usage::

    python scripts/upload_study1_hf.py --repo-id <user>/provoice-study1 --dry-run --verify-chain
    python scripts/upload_study1_hf.py --repo-id <user>/provoice-study1 --skip-convert --private --license cc-by-nc-4.0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ProVoice.fcd_config import FCD_NAMES  # noqa: E402

SCHEMA_DOC = REPO_ROOT / "docs" / "study1_dataset_schema.md"
HR_SCRIPT = REPO_ROOT / "data_preprocessing" / "heart_rate_preprocessing.py"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_loa_dataset.py"
STUDY_PIDS = [f"{i:03d}" for i in range(1, 13)]

# --------------------------------------------------------------------------- #
# Column contract -- mirrors docs/study1_dataset_schema.md section by section.
# Order here is the order of keys in the published files.
# --------------------------------------------------------------------------- #
S, F, I, B = pa.string(), pa.float64(), pa.int64(), pa.bool_()
LIST_S = pa.list_(pa.string())
FCD_STRUCT = pa.struct([(name, I) for name in FCD_NAMES])

# A.1 identity / session context
_FRAME_A1 = [("timestamp", S), ("session_id", S), ("participantid", S),
             ("traffic_seed", I), ("environment", S), ("secondary_task", S),
             ("modeltype", S), ("state_model", S), ("w_fcd", F)]
# A.2 face / driver state
_FRAME_A2 = [("face_present", B), ("eye_ar", F), ("mar", F),
             ("gaze_score", F), ("gaze_score_raw", F), ("gaze_distracted", B),
             ("blink_rate", F), ("blink_rate_raw", F),
             ("yawn_rate", F), ("yawn_rate_raw", F),
             ("perclos", F), ("perclos_raw", F), ("drowsiness_alert", B),
             ("emotion", S), ("emotion_prob", F), ("lab", LIST_S),
             ("facebox_misses", I), ("facebox_consec_misses", I)]
# A.3 rPPG (shared part)
_FRAME_A3 = [("heart_rate", F), ("heart_rate_raw", F), ("hr_delta", F),
             ("hr_rejected", B), ("respiratory_rate", F), ("rr_delta", F),
             ("rppg_gaps", I), ("rppg_dropped", I), ("rppg_suppressed", I),
             ("rppg_harmonic_rejects", I)]
# A.3 rPPG, preprocessed-only provenance ("P" rows)
_FRAME_A3_P = [("hr_repaired", B), ("hr_repair_method", S)]
# A.4 vehicle / world
_FRAME_A4 = [("speed_kmh", F), ("speed_limit_kmh", F), ("speed_ratio_max", F),
             ("speed_ratio_limit", F), ("throttle", F), ("brake", F),
             ("steer", F), ("acceleration", F), ("gear", I), ("reverse", B),
             ("hand_brake", B), ("is_junction", B), ("is_night", B),
             ("traffic_light_state", S), ("lead_distance_m", F),
             ("headway_s", F), ("precipitation", F), ("fog_density", F),
             ("headlight", B), ("fog_light", B),
             ("left_indicator", B), ("right_indicator", B)]
# A.5 pipeline timing
_FRAME_A5 = [("frame_dt_ms", F), ("collect_ms", F), ("fps_inst", F), ("fps_avg", F)]

FRAMES_RAW_COLS = _FRAME_A1 + _FRAME_A2 + _FRAME_A3 + _FRAME_A4 + _FRAME_A5
FRAMES_PRE_COLS = _FRAME_A1 + _FRAME_A2 + _FRAME_A3 + _FRAME_A3_P + _FRAME_A4 + _FRAME_A5

# C.3 labeled = all preprocessed columns + 2 from the label row + 7 from the builder
LABELED_COLS = FRAMES_PRE_COLS + [("functionname", S), ("FCD", FCD_STRUCT)] + [
    ("segment_id", S), ("user_loa", S),
    ("Level_1", I), ("Level_2", I), ("Level_3", I), ("Level_4", I), ("Level_5", I)]

# B labels (CSV, values kept verbatim; types below are for the card only)
LABELS_COLS = [("session_id", S), ("participantid", S), ("window_idx", I),
               ("prompt_in_window", I), ("window_start_ms", I), ("window_end_ms", I),
               ("window_start_timestamp", S), ("window_end_timestamp", S),
               ("selection_timestamp", S), ("selection_frame", I),
               ("selection_sim_time", F), ("selection_speed_kmh", F),
               ("functionname", S), ("user_selected_loa", S),
               ("ambient_gain", F), ("ambient_seed", I), ("ambient_source", S),
               ("environment", S), ("secondary_task", S),
               ("modeltype", S), ("state_model", S), ("w_fcd", F)]

# D. columns present on disk but deliberately NOT published
EXCLUDED = {
    "frames_raw": {"functionname", "LoA", "FCD"},
    "frames_preprocessed": {"functionname", "LoA", "FCD", "hr_repair_reason"},
    "labeled": {"LoA", "hr_repair_reason"},
    "labels": {"emotion", "system_action", "system_level", "system_loa",
               "system_message", "system_probs", "system_profile",
               "system_fallback", "system_fallback_reason", "system_fcd"},
}

# The Hub applies ONE builder to every config of a repo (verified: with three
# JSONL configs and one CSV, `datasets` parsed the CSV as JSON and failed), so
# the `labels` config is JSONL. `build_loa_dataset.py` reads CSV, so the same
# rows are also written verbatim to this companion file, which is not a config.
LABELS_CSV = "data/labels.csv"

# Expected shape, straight from the release table in the schema doc.
RELEASE = {
    #  config                source file               published file                 rows     columns
    "frames_raw":          ("raw_data.jsonl",          "data/frames_raw.jsonl",          380_990, FRAMES_RAW_COLS),
    "frames_preprocessed": ("preprocessed_data.jsonl", "data/frames_preprocessed.jsonl", 380_990, FRAMES_PRE_COLS),
    "labels":              ("user_loa_labels.csv",     "data/labels.jsonl",                1_446, LABELS_COLS),
    "labeled":             ("labeled_data.jsonl",      "data/labeled.jsonl",             508_282, LABELED_COLS),
}

CARD_INTRO = """\
# ProVoice study 1 — driver state, vehicle context and preferred Level of Autonomy

Driving-simulator data from the population data collection of the ProVoice /
ProActivity project (CARLA 0.10): **12 drivers × 2 sessions**, ~20 Hz
multimodal driver-state and vehicle frames, and **1,446 driver-assigned
Level-of-Autonomy (LoA) labels** stating how autonomously an in-vehicle
assistant should act on a given task. Drivers were prompted every 20 s about
two randomly drawn in-vehicle tasks and marked, for each, the LoA(s) they
would accept (0 = do nothing … 4 = act autonomously).

| Config | What it is | Use it for |
| --- | --- | --- |
| `labels` | one row per prompt: the driver's answer plus the window it refers to | the ground truth; join key for everything else |
| `frames_preprocessed` | every logged frame with the heart-rate channel repaired offline | driver-state modelling (this is what the released models were trained on) |
| `frames_raw` | the same frames exactly as logged live | provenance; comparing the live vs. offline HR filter |
| `labeled` | `frames_preprocessed` joined to `labels`, **frames duplicated once per label** | one-file training input; read §C before counting rows |

`data/labels.csv` is the same table as the `labels` config in the CSV form
`scripts/build_loa_dataset.py` reads (the Hub loads one file format per repo).
`calibration/` (not a config) holds each driver's 60 s calibration baseline and
per-tick calibration log — inputs of the heart-rate repair, shipped so that
`frames_preprocessed` can be regenerated from `frames_raw`.

```python
from datasets import load_dataset
labels = load_dataset("{repo_id}", "labels", split="train")
frames = load_dataset("{repo_id}", "frames_preprocessed", split="train")
```

## Reproducing the pipeline

Files are in the exact formats the code reads (JSONL frames, CSV labels), so
the project's documented commands run on them unchanged{code_ref}. Code
version used to build and verify this release: `{commit}`.

```bash
hf download {repo_id} --repo-type dataset --local-dir provoice_study1

# 1. frames_raw + calibration  ->  frames_preprocessed   (heart-rate repair)
python data_preprocessing/heart_rate_preprocessing.py \\
    --in-data  provoice_study1/data/frames_raw.jsonl \\
    --calib-dir provoice_study1/calibration \\
    --out-data provoice_study1/data/frames_preprocessed.jsonl --no-write-calibration

# 2. frames_preprocessed + labels  ->  labeled            (window alignment)
python scripts/build_loa_dataset.py \\
    --raw    provoice_study1/data/frames_preprocessed.jsonl \\
    --labels provoice_study1/data/labels.csv \\
    --out-jsonl provoice_study1/data/labeled.jsonl --out-fcd fcd_out.csv

# 3. labeled  ->  population model
python -m ProVoice.models.train_XLSTM --in provoice_study1/data/labeled.jsonl \\
    --out trained_models/state_xlstm.pt --loss corn
```

{chain_note}The columns listed under "excluded" in the schema
below are absent from the released files; none of them is read by any step.

{hashes}

---

"""

CARD_OUTRO = """\

---

## Verification (performed at build time by `scripts/upload_study1_hf.py`)

{verification}
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_env_token(env_path: Path) -> Optional[str]:
    """Return HF_TOKEN from the environment or from a KEY=VALUE .env file."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            return val.strip().strip("'\"")
    return None


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce(value: Any, typ: pa.DataType) -> Any:
    """Coerce one JSON value to the declared type.

    Resolves the bool/int and int/float mixing at the source: the collector's
    ``__init__`` placeholders are ``int``, so the first ~20 frames of every
    session carry ``0`` where later frames carry ``false``. Nothing downstream
    distinguishes them (``encode_frame`` maps both to 0.0), verified by the
    ``normalize_row`` lockstep check.
    """
    if value is None or value == "":
        return None
    if typ == B:
        return bool(value)
    if typ == F:
        return float(value)
    if typ == I:
        return int(value)
    if typ == S:
        return str(value)
    if typ == LIST_S:
        return [str(v) for v in value]
    if typ == FCD_STRUCT:
        return {name: int(value[name]) for name in FCD_NAMES}
    raise TypeError(f"no coercion for {typ}")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _check_keys(config: str, seen: set, names: List[str]) -> None:
    """The keys on disk must be exactly the published columns plus the
    documented exclusions -- a renamed or new collector field fails here
    instead of silently vanishing from the release."""
    unexpected = seen - set(names) - EXCLUDED[config]
    if unexpected:
        raise KeyError(f"{config}: columns on disk not covered by the schema doc: {sorted(unexpected)}")
    missing = set(names) - seen
    if missing:
        raise KeyError(f"{config}: schema doc lists columns never seen on disk: {sorted(missing)}")


def _iter_csv(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def convert_jsonl(config: str, src: Path, dst: Path) -> int:
    _, _, _, cols = RELEASE[config]
    names = [n for n, _ in cols]
    types = dict(cols)
    seen: set = set()
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = _iter_csv(src) if src.suffix == ".csv" else _iter_jsonl(src)
    with open(dst, "w", encoding="utf-8", newline="\n") as out:
        for rec in rows:
            seen.update(rec.keys())
            row = {k: _coerce(rec.get(k), types[k]) for k in names}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    _check_keys(config, seen, names)
    return n


def convert_csv(config: str, src: Path, dst: Path) -> int:
    """Drop the excluded columns; every kept value is copied verbatim."""
    _, _, _, cols = RELEASE[config]
    names = [n for n, _ in cols]
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8", newline="") as fin, \
            open(dst, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        _check_keys(config, set(reader.fieldnames or []), names)
        writer = csv.DictWriter(fout, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for rec in reader:
            writer.writerow({k: rec[k] for k in names})
            n += 1
    return n


def stage_calibration(src_dir: Path, out_dir: Path) -> List[str]:
    """calibration_<pid>.json + calibration_logs/log_calibration_<pid>.csv for
    the 12 study participants. The ``*_preprocessed.json`` files are OUTPUTS of
    the HR repair and are not staged; the pipeline regenerates them."""
    staged: List[str] = []
    dst_dir = out_dir / "calibration"
    (dst_dir / "calibration_logs").mkdir(parents=True, exist_ok=True)
    for pid in STUDY_PIDS:
        for rel in (f"calibration_{pid}.json", f"calibration_logs/log_calibration_{pid}.csv"):
            src = src_dir / rel
            if not src.exists():
                raise FileNotFoundError(f"calibration input missing for participant {pid}: {src}")
            shutil.copyfile(src, dst_dir / rel)
            staged.append(f"calibration/{rel}")
    return staged


def count_rows(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        n = sum(1 for line in fh if line.strip())
    return n - 1 if path.suffix == ".csv" else n


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify_counts(out_dir: Path) -> List[str]:
    notes = []
    for config, (_, pub, exp_rows, cols) in RELEASE.items():
        path = out_dir / pub
        n = count_rows(path)
        first = next(_iter_jsonl(path)) if path.suffix == ".jsonl" else \
            next(csv.DictReader(open(path, encoding="utf-8", newline="")))
        if n != exp_rows or list(first.keys()) != [c for c, _ in cols]:
            raise AssertionError(f"{config}: {n:,} rows x {len(first)} cols, schema doc says "
                                 f"{exp_rows:,} x {len(cols)} (or column order differs)")
        notes.append(f"`{config}`: {n:,} rows × {len(cols)} columns, as documented")
    # labels.csv must be the same rows as labels.jsonl, value for value
    types = dict(LABELS_COLS)
    for i, (a, b) in enumerate(zip(_iter_csv(out_dir / LABELS_CSV), _iter_jsonl(out_dir / RELEASE["labels"][1]))):
        if {k: _coerce(v, types[k]) for k, v in a.items()} != b:
            raise AssertionError(f"labels.csv row {i} differs from labels.jsonl")
    notes.append(f"`{LABELS_CSV}` (pipeline input) and `data/labels.jsonl` (Hub config) hold identical rows")
    return notes


def verify_hub_load(out_dir: Path) -> List[str]:
    """Load the staged folder with ``datasets`` exactly as the Hub will: through
    the card's ``configs`` and ``dataset_info.features``."""
    from datasets import load_dataset
    notes = []
    for config, (_, _, exp_rows, cols) in RELEASE.items():
        ds = load_dataset(str(out_dir), name=config, split="train", download_mode="force_redownload")
        if len(ds) != exp_rows:
            raise AssertionError(f"{config}: datasets loaded {len(ds):,} rows, expected {exp_rows:,}")
        pid = ds[0]["participantid"]
        if not (isinstance(pid, str) and pid == "001"):
            raise AssertionError(f"{config}: participantid loaded as {pid!r} -- zero-padding lost")
        for name, typ in cols:
            if typ in (S, F, I, B) and str(ds.features[name]) != f"Value('{ {S: 'string', F: 'float64', I: 'int64', B: 'bool'}[typ] }')":
                raise AssertionError(f"{config}.{name}: loaded as {ds.features[name]}, declared {typ}")
        notes.append(f"`load_dataset(..., \"{config}\")` yields {len(ds):,} rows with the declared types; "
                     f"`participantid` keeps its zero padding")
        print(f"  [hub-load] {config}: OK")
    return notes


def verify_training_equivalence(src: Path, staged: Path) -> str:
    """The trainers never see a file -- they see ``normalize_row(row)``. Apply it
    in lockstep to the original ``labeled_data.jsonl`` and the released file and
    require identical output for every row: that is the statement that a model
    trained on the release is the model trained on the original."""
    from ProVoice.models.train_XLSTM import normalize_row
    n = 0
    for a, b in zip(_iter_jsonl(src), _iter_jsonl(staged)):
        na, nb = normalize_row(a), normalize_row(b)
        if na != nb:
            # the only tolerated difference: int 0 vs False / 0.0 on the coerced columns
            diff = {k: (na[k], nb[k]) for k in na if na[k] != nb[k] or type(na[k]) is not type(nb[k])}
            for k, (x, y) in diff.items():
                if not (x in (0, 0.0, False) and y in (0, 0.0, False)):
                    raise AssertionError(f"labeled row {n}: normalize_row differs on {k}: {x!r} vs {y!r}")
        n += 1
    if n != count_rows(staged) or n != count_rows(src):
        raise AssertionError("labeled: row counts differ between source and release")
    print(f"  [train-equiv] normalize_row identical on all {n:,} labeled rows")
    return (f"`train_XLSTM.normalize_row` applied to every row of the original `labeled_data.jsonl` "
            f"and of the released `labeled.jsonl` gives identical output on all {n:,} rows -- the "
            f"dropped columns and the bool/int normalisation are invisible to the trainers")


def _rows_equal(config: str, regenerated: Path, staged: Path) -> int:
    """Lockstep compare on the published columns, after the same coercion."""
    _, _, _, cols = RELEASE[config]
    names = [n for n, _ in cols]
    types = dict(cols)
    n = 0
    for a, b in zip(_iter_jsonl(regenerated), _iter_jsonl(staged)):
        ra = {k: _coerce(a.get(k), types[k]) for k in names}
        rb = {k: _coerce(b.get(k), types[k]) for k in names}
        if ra != rb:
            bad = [k for k in names if ra[k] != rb[k]]
            raise AssertionError(f"{config}: regenerated row {n} differs on {bad}: "
                                 f"{[(k, ra[k], rb[k]) for k in bad[:3]]}")
        n += 1
    if n != count_rows(regenerated) or n != count_rows(staged):
        raise AssertionError(f"{config}: regenerated {count_rows(regenerated):,} rows vs staged {count_rows(staged):,}")
    return n


def verify_chain(out_dir: Path) -> List[str]:
    """Re-run the project's own pipeline on the STAGED inputs and diff against
    the staged outputs. This is the reproducibility claim, executed."""
    notes = []
    with tempfile.TemporaryDirectory(prefix="hf_chain_") as tmp_s:
        tmp = Path(tmp_s)
        # the HR script may write *_preprocessed.json next to the baselines; work on a copy
        calib = tmp / "calibration"
        shutil.copytree(out_dir / "calibration", calib)
        pre = tmp / "frames_preprocessed.jsonl"
        t0 = time.time()
        subprocess.run([sys.executable, str(HR_SCRIPT),
                        "--in-data", str(out_dir / RELEASE["frames_raw"][1]),
                        "--calib-dir", str(calib), "--out-data", str(pre),
                        "--no-write-calibration"],
                       cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
        n = _rows_equal("frames_preprocessed", pre, out_dir / RELEASE["frames_preprocessed"][1])
        print(f"  [chain] heart_rate_preprocessing.py on frames_raw reproduces frames_preprocessed "
              f"({n:,} rows, {time.time() - t0:.0f}s)")
        notes.append(f"`heart_rate_preprocessing.py` run on the released `frames_raw` + `calibration/` "
                     f"reproduces the released `frames_preprocessed` on all {n:,} rows and every published column")

        lab = tmp / "labeled.jsonl"
        t0 = time.time()
        subprocess.run([sys.executable, str(BUILD_SCRIPT),
                        "--raw", str(pre), "--labels", str(out_dir / LABELS_CSV),
                        "--out-jsonl", str(lab), "--out-fcd", str(tmp / "fcd_out.csv")],
                       cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
        n = _rows_equal("labeled", lab, out_dir / RELEASE["labeled"][1])
        print(f"  [chain] build_loa_dataset.py on the regenerated frames + labels reproduces labeled "
              f"({n:,} rows, {time.time() - t0:.0f}s)")
        notes.append(f"`build_loa_dataset.py` run on that regenerated file + the released `labels` "
                     f"reproduces the released `labeled` on all {n:,} rows and every published column")
    return notes


# --------------------------------------------------------------------------- #
# card
# --------------------------------------------------------------------------- #
def features_yaml(cols) -> list:
    from datasets import Features
    return Features.from_arrow_schema(pa.schema(cols))._to_yaml_list()


def write_card(out_dir: Path, repo_id: str, license_id: Optional[str], code_url: Optional[str],
               hashes: Dict[str, str], verification: List[str]) -> Path:
    import yaml
    body = SCHEMA_DOC.read_text(encoding="utf-8")
    if body.startswith("# "):  # the card has its own title
        body = body.split("\n", 1)[1].lstrip("\n")

    meta: Dict[str, Any] = {}
    if license_id:
        meta["license"] = license_id
    meta.update({
        "pretty_name": "ProVoice Study 1 - Driver State and Preferred Level of Autonomy",
        "size_categories": ["100K<n<1M"],
        "tags": ["driving-simulator", "carla", "driver-state", "human-machine-interaction",
                 "in-vehicle-assistant", "level-of-autonomy", "ordinal-regression", "personalization"],
        "configs": [{"config_name": c, "data_files": pub} for c, (_, pub, _, _) in RELEASE.items()],
        "dataset_info": [{"config_name": c, "features": features_yaml(cols)}
                         for c, (_, _, _, cols) in RELEASE.items()],
    })
    front = "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000) + "---\n\n"

    hash_lines = ["| File | SHA-256 |", "| --- | --- |"] + \
                 [f"| `{p}` | `{h}` |" for p, h in sorted(hashes.items())]
    chain_ran = any("heart_rate_preprocessing.py` run on" in v for v in verification)
    chain_note = ("Steps 1 and 2 were re-run on the released files at build time and reproduce "
                  "the released `frames_preprocessed` and `labeled` **row for row** (see "
                  "\"Verification\" at the end). " if chain_ran else
                  "Steps 1 and 2 were NOT re-run for this build (`--verify-chain` not passed). ")
    intro = CARD_INTRO.format(
        repo_id=repo_id,
        code_ref=f" (code: [{code_url}]({code_url}))" if code_url else "",
        commit=git_commit(),
        chain_note=chain_note,
        hashes="\n".join(hash_lines))
    outro = CARD_OUTRO.format(verification="\n".join(f"- {v}" for v in verification))
    path = out_dir / "README.md"
    path.write_text(front + intro + body + outro, encoding="utf-8")
    return path


def upload(out_dir: Path, repo_id: str, token: str, private: bool,
           gated: Optional[str], message: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    url = api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"  repo: {url}")
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
                      commit_message=message)
    if gated and gated != "none":
        api.update_repo_settings(repo_id=repo_id, repo_type="dataset", gated=gated)
        print(f"  gating: {gated}")
    return str(url)


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="<user-or-org>/<dataset-name>")
    ap.add_argument("--src-dir", type=Path, default=REPO_ROOT / "data" / "study1_data")
    ap.add_argument("--calib-dir", type=Path, default=REPO_ROOT / "data" / "calibration_data_study",
                    help="study-1 calibration baselines + calibration_logs/")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "hf_release" / "study1",
                    help="staging folder that is uploaded verbatim")
    ap.add_argument("--env", type=Path, default=REPO_ROOT / ".env", help="file holding HF_TOKEN=...")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--gated", choices=["none", "auto", "manual"], default="none",
                    help="require access requests (auto-approved or reviewed by you)")
    ap.add_argument("--license", dest="license_id", default=None,
                    help="Hub license id for the card, e.g. cc-by-4.0, cc-by-nc-4.0")
    ap.add_argument("--code-url", default=None, help="URL of the code repository, linked from the card")
    ap.add_argument("--commit-message", default="Add study-1 population dataset")
    ap.add_argument("--skip-convert", action="store_true", help="reuse the files already in --out-dir")
    ap.add_argument("--verify-chain", action="store_true",
                    help="re-run heart_rate_preprocessing.py and build_loa_dataset.py on the staged "
                         "files and diff against the staged outputs (several minutes)")
    ap.add_argument("--no-verify", action="store_true", help="skip the datasets load and normalize_row checks")
    ap.add_argument("--dry-run", action="store_true",
                    help="convert, verify and write the card, but do not touch the Hub")
    args = ap.parse_args(argv)

    if not SCHEMA_DOC.exists():
        print(f"[error] schema doc not found: {SCHEMA_DOC}", file=sys.stderr)
        return 2
    if not args.license_id:
        print("[warn] no --license given: the card will carry no license field.")

    print(f"[1/4] stage -> {args.out_dir}")
    for config, (src_name, pub, _, _) in RELEASE.items():
        dst = args.out_dir / pub
        if args.skip_convert and dst.exists():
            print(f"  {config}: reusing {pub}")
            continue
        src = args.src_dir / src_name
        if not src.exists():
            print(f"[error] missing source file {src}", file=sys.stderr)
            return 2
        t0 = time.time()
        n = convert_jsonl(config, src, dst)
        print(f"  {config}: {src_name} -> {pub}  {n:,} rows, {dst.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s")
        if src.suffix == ".csv":
            m = convert_csv(config, src, args.out_dir / LABELS_CSV)
            assert m == n, (m, n)
            print(f"  {config}: {src_name} -> {LABELS_CSV}  (verbatim companion for build_loa_dataset.py)")
    calib_files = stage_calibration(args.calib_dir, args.out_dir)
    print(f"  calibration: {len(calib_files)} files for {len(STUDY_PIDS)} participants")

    print("[2/4] verify")
    verification = verify_counts(args.out_dir)
    for v in verification:
        print(f"  [counts] {v}")
    if args.verify_chain:
        verification += verify_chain(args.out_dir)
    if not args.no_verify:
        verification.append(verify_training_equivalence(
            args.src_dir / RELEASE["labeled"][0], args.out_dir / RELEASE["labeled"][1]))

    print("[3/4] dataset card")
    # the card is written twice: the hub-load check needs the features block to exist first
    payload = sorted(p for p in args.out_dir.rglob("*") if p.is_file() and p.name != "README.md")
    hashes = {p.relative_to(args.out_dir).as_posix(): sha256(p) for p in payload}
    write_card(args.out_dir, args.repo_id, args.license_id, args.code_url, hashes, verification)
    if not args.no_verify:
        verification += verify_hub_load(args.out_dir)
    card = write_card(args.out_dir, args.repo_id, args.license_id, args.code_url, hashes, verification)
    print(f"  {card}  ({card.stat().st_size / 1e3:.0f} kB)")
    allowed_prefixes = ("data/", "calibration/")
    extra = [p for p in hashes if not p.startswith(allowed_prefixes)]
    if extra:
        print(f"[error] unexpected files in {args.out_dir}: {extra} -- the folder is uploaded verbatim", file=sys.stderr)
        return 2
    print("  staged:", ", ".join(sorted(hashes)))

    if args.dry_run:
        print("[4/4] upload skipped (--dry-run)")
        return 0

    token = load_env_token(args.env)
    if not token:
        print(f"[error] no HF token: set HF_TOKEN in {args.env} or in the environment", file=sys.stderr)
        return 2
    print(f"[4/4] upload -> https://huggingface.co/datasets/{args.repo_id}"
          f"  ({'private' if args.private else 'public'}, gated={args.gated})")
    upload(args.out_dir, args.repo_id, token, args.private, args.gated, args.commit_message)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
