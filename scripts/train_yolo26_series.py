"""Train the full YOLO26 classification series on the State Farm dataset.

Trains n / s / m / l / x back-to-back (classification task, whole-image
labels) using ``datasets/distraction_sf`` built from the State Farm
Distracted Driver dataset. The recipe matches the user's "full match"
preference but tuned for classification:

    50 epochs (State Farm is small, overfits quickly beyond this)
    imgsz 224  (standard YOLO cls pretrained size)
    patience 15
    cosine LR

Batch sizes are sized so that GPU memory stays inside ~14 GB (leaving
headroom for other processes on a 16 GB card):
    n: 128   s: 64   m: 48   l: 32   x: 16

Each variant is packaged into::

    exports/provoice-distraction-yolo26/
      n/  model.pt + results + model card
      s/  ...  m/  ...  l/  ...  x/  ...
      README.md   <- comparison card (all variants)

Run::

    PYTHONPATH=src uv run --no-sync python scripts/train_yolo26_series.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS   = REPO_ROOT / "exports" / "provoice-distraction-yolo26"
DATA      = REPO_ROOT / "datasets" / "distraction_sf"
DATASET_DIR = DATA

# (variant, cls weights, batch)
VARIANTS = [
    ("n", "yolo26n-cls.pt", 128),
    ("s", "yolo26s-cls.pt",  64),
    ("m", "yolo26m-cls.pt",  48),
    ("l", "yolo26l-cls.pt",  32),
    ("x", "yolo26x-cls.pt",  16),
]

EPOCHS   = 50
IMGSZ    = 224
PATIENCE = 15

PROJECT_CLASSES  = ["safe", "phone", "drink", "distracted"]
PARAMS_M = {"n": "~2.8M", "s": "~10M", "m": "~22M", "l": "~26M", "x": "~59M"}


def log(msg: str) -> None:
    line = f"[series {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # Windows consoles default to cp1252; never let a stray glyph crash a run.
        import sys
        enc = sys.stdout.encoding or "ascii"
        sys.stdout.write(line.encode(enc, "replace").decode(enc) + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# GPU wait
# ---------------------------------------------------------------------------

def gpu_used_mb() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=30)
        return max(int(x.strip()) for x in out.strip().splitlines() if x.strip())
    except Exception:
        return None


def wait_for_gpu(threshold_mb: int, idle_checks: int, interval_s: int, max_wait_s: int) -> None:
    log(f"waiting for GPU idle (used < {threshold_mb} MB, "
        f"{idle_checks} consecutive @ {interval_s}s)...")
    start = time.time()
    consecutive = 0
    while True:
        used = gpu_used_mb()
        if used is None or used < threshold_mb:
            consecutive += 1
            log(f"  idle check {consecutive}/{idle_checks} (used={used} MB)")
            if consecutive >= idle_checks:
                log("GPU is idle — starting.")
                return
        else:
            if consecutive:
                log(f"  GPU busy (used={used} MB); resetting count")
            consecutive = 0
        if max_wait_s and (time.time() - start) > max_wait_s:
            log(f"max wait {max_wait_s}s exceeded; starting anyway.")
            return
        time.sleep(interval_s)


# ---------------------------------------------------------------------------
# Find run output dir
# ---------------------------------------------------------------------------

def find_run_dir(name: str) -> Optional[Path]:
    # Ultralytics can nest under classify/ or detect/
    patterns = [
        str(REPO_ROOT / "runs" / "**" / name / "weights" / "best.pt"),
        str(REPO_ROOT / "runs" / name / "weights" / "best.pt"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime)).parent.parent


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def best_cls_metrics(run_dir: Path) -> dict:
    """Read best top1_acc row from results.csv (classification runs)."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    if not rows:
        return {}

    # Classification CSV uses metrics/accuracy_top1 or top1/top5
    acc_key = next((k for k in rows[0] if "top1" in k.lower() and "acc" in k.lower()), None)
    if acc_key is None:
        acc_key = next((k for k in rows[0] if "top1" in k.lower()), None)

    def _f(r):
        try:
            return float(r.get(acc_key, 0) or 0)
        except ValueError:
            return 0.0

    best = max(rows, key=_f) if acc_key else rows[-1]
    out = {}
    for k in best:
        v = (best[k] or "").strip()
        if not v:
            continue
        label = k.strip().lstrip("metrics/")
        try:
            out[label] = f"{float(v):.4f}"
        except ValueError:
            out[label] = v
    return out


# ---------------------------------------------------------------------------
# Per-variant model card
# ---------------------------------------------------------------------------

def per_variant_card(variant: str, weights: str, metrics: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items()) or "| (no metrics) | |"
    cls_yaml = "\n".join(f"  {i}: {c}" for i, c in enumerate(PROJECT_CLASSES))
    from datetime import date
    return f"""---
license: agpl-3.0
library_name: ultralytics
tags:
- image-classification
- yolo
- yolo26
- driver-monitoring
- distraction-detection
pipeline_tag: image-classification
---

# ProVoice Distraction Classifier — YOLO26{variant}

Fine-tuned [Ultralytics YOLO26](https://docs.ultralytics.com) **classification**
model for in-cabin driver-distraction detection. Part of the
ProActivity / ProVoice driver-monitoring stack.

## Classes

```yaml
names:
{cls_yaml}
```

## Metrics (best validation epoch)

| metric | value |
|---|---|
{rows}

## Training data

**State Farm Distracted Driver Detection** dataset (22,424 labelled in-cabin
images, 26 subjects, 10 SF classes merged to 4 ProVoice labels).
Subject-aware 80/20 train/val split (no driver appears in both splits).

## Usage

```python
from ultralytics import YOLO
model = YOLO("model.pt")
result = model.predict("driver_frame.jpg", imgsz=224)
label = model.names[result[0].probs.top1]   # e.g. "phone"
conf  = float(result[0].probs.top1conf)
print(label, conf)
```

In ProVoice, set `PROVOICE_YOLO_WEIGHTS` to this `model.pt` or place it at
`src/ProVoice/trained_models/distraction/distraction_best.pt`.

## Reproduce

```bash
uv run --no-sync python -m ProVoice.train_distraction \\
    --task classify \\
    --data datasets/distraction_sf \\
    --weights {weights} --epochs {EPOCHS} --imgsz {IMGSZ} --cos-lr
```

_Exported {date.today().isoformat()}._
"""


EXAMPLE_PY = '''"""Minimal inference example for the ProVoice distraction classifier."""
import sys
from ultralytics import YOLO

model = YOLO("model.pt")
src = sys.argv[1] if len(sys.argv) > 1 else "driver_frame.jpg"
r = model.predict(src, imgsz=224)[0]
print(model.names[r.probs.top1], round(float(r.probs.top1conf), 3))
'''


# ---------------------------------------------------------------------------
# Comparison README
# ---------------------------------------------------------------------------

def write_comparison_readme(variants_done: List[str]) -> None:
    order = ["n", "s", "m", "l", "x"]
    lines = [
        "---",
        "license: agpl-3.0",
        "library_name: ultralytics",
        "tags:",
        "- image-classification",
        "- yolo",
        "- yolo26",
        "- driver-monitoring",
        "pipeline_tag: image-classification",
        "---",
        "",
        "# ProVoice Distraction Classifiers — YOLO26 series",
        "",
        "In-cabin driver-distraction classifiers fine-tuned on the "
        "**State Farm Distracted Driver** dataset (22 424 in-cabin images, "
        "4 ProVoice output classes). Pick the size that fits your latency budget.",
        "",
        "## Classes",
        "",
        "| id | label | State Farm source classes |",
        "|---|---|---|",
        "| 0 | safe | c0 (normal driving) |",
        "| 1 | phone | c1–c4 (texting or calling) |",
        "| 2 | drink | c6 (drinking) |",
        "| 3 | distracted | c5, c7–c9 (other distractions) |",
        "",
        "## Model comparison",
        "",
        "| variant | params | top-1 acc | top-5 acc | subfolder |",
        "|---|---|---|---|---|",
    ]

    for v in order:
        rd = find_run_dir(f"distraction_sf_{v}")
        m = best_cls_metrics(rd) if rd else {}
        acc1 = m.get("accuracy_top1", m.get("top1", "-"))
        acc5 = m.get("accuracy_top5", m.get("top5", "-"))
        params = PARAMS_M.get(v, "?")
        lines.append(f"| yolo26{v}-cls | {params} | {acc1} | {acc5} | `{v}/` |")

    lines += [
        "",
        "## Usage",
        "",
        "```python",
        "from huggingface_hub import hf_hub_download",
        "from ultralytics import YOLO",
        "",
        "# pick a size: n / s / m / l / x",
        "path = hf_hub_download('<user>/provoice-distraction-yolo26', 's/model.pt')",
        "model = YOLO(path)",
        "r = model.predict('driver_frame.jpg', imgsz=224)[0]",
        "print(model.names[r.probs.top1], float(r.probs.top1conf))",
        "```",
        "",
        "Each subfolder contains `model.pt`, training curves, and a per-variant model card.",
        "",
        f"_Series assembled {datetime.now().date().isoformat()}._",
        "",
    ]
    EXPORTS.mkdir(parents=True, exist_ok=True)
    (EXPORTS / "README.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote comparison card -> {EXPORTS / 'README.md'}")


# ---------------------------------------------------------------------------
# Train + package one variant
# ---------------------------------------------------------------------------

def train_variant(variant: str, weights: str, batch: int, device: str, env: dict) -> bool:
    name = f"distraction_sf_{variant}"
    log(f"=== training yolo26{variant}-cls (batch={batch}) ===")
    cmd = [
        sys.executable, "-u", "-m", "ProVoice.train_distraction",
        "--task",    "classify",
        "--data",    str(DATA),
        "--weights", weights,
        "--epochs",  str(EPOCHS),
        "--imgsz",   str(IMGSZ),
        "--batch",   str(batch),
        "--device",  device,
        "--project", "runs",
        "--name",    name,
        "--patience", str(PATIENCE),
        "--cos-lr",
    ]
    rc = subprocess.call(cmd, env=env, cwd=str(REPO_ROOT))
    if rc != 0:
        log(f"!! yolo26{variant} training exited {rc}; skipping package")
        return False

    run_dir = find_run_dir(name)
    if run_dir is None:
        log(f"!! could not locate run dir for {name}; skipping package")
        return False

    out_dir = EXPORTS / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_pt = run_dir / "weights" / "best.pt"
    if weights_pt.exists():
        shutil.copy2(weights_pt, out_dir / "model.pt")
    for fname in ("results.csv", "results.png"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    metrics = best_cls_metrics(run_dir)
    (out_dir / "README.md").write_text(per_variant_card(variant, weights, metrics), encoding="utf-8")
    (out_dir / "example.py").write_text(EXAMPLE_PY, encoding="utf-8")
    log(f"yolo26{variant} packaged -> {out_dir}")

    # Wire the s variant into the runtime as the default best model
    if variant == "s" and weights_pt.exists():
        bundled = REPO_ROOT / "src" / "ProVoice" / "trained_models" / "distraction" / "distraction_best.pt"
        bundled.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights_pt, bundled)
        log(f"yolo26s-cls wired into runtime: {bundled}")

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="n,s,m,l,x")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--gpu-threshold-mb", type=int, default=3000)
    ap.add_argument("--idle-checks", type=int, default=3)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--max-wait", type=int, default=0)
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(
            f"Dataset not found: {DATA}\n"
            "Run: uv run --no-sync python scripts/build_statefarm_dataset.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    if not args.no_wait:
        wait_for_gpu(args.gpu_threshold_mb, args.idle_checks, args.interval, args.max_wait)

    want = {v.strip() for v in args.variants.split(",") if v.strip()}
    todo = [(v, w, b) for (v, w, b) in VARIANTS if v in want]

    done: List[str] = []
    for variant, weights, batch in todo:
        try:
            if train_variant(variant, weights, batch, args.device, env):
                done.append(variant)
                write_comparison_readme(done)
        except Exception as exc:
            log(f"!! unexpected error on yolo26{variant}: {exc}")

    write_comparison_readme(done)
    log(f"ALL DONE. Trained: {done}. Export at {EXPORTS}")
    log("To publish: huggingface-cli upload <user>/provoice-distraction-yolo26 "
        f"{EXPORTS} . --repo-type model")


if __name__ == "__main__":
    main()
