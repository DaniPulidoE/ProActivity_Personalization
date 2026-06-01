"""Convert the State Farm Distracted Driver dataset to YOLO classification format.

State Farm has 26 subjects × 10 classes × ~86 images = 22,424 labelled images.
The unlabelled test set (79,726 images) is skipped (no ground-truth labels).

Class mapping (10 SF classes → 4 ProVoice labels):
    c0  safe driving            → safe
    c1  texting (right)         → phone
    c2  phone to right ear      → phone
    c3  texting (left)          → phone
    c4  phone to left ear       → phone
    c5  radio / infotainment    → distracted
    c6  drinking                → drink
    c7  reaching behind         → distracted
    c8  hair / makeup           → distracted
    c9  talking to passenger    → distracted

Split strategy: subject-aware (all images of a driver go to ONE split).
This prevents the model from shortcutting on face identity.
The first ~20 % of subjects (sorted) become val; the rest train.

Output layout (YOLO classification format)::

    datasets/distraction_sf/
      train/
        safe/  phone/  drink/  distracted/
      val/
        safe/  phone/  drink/  distracted/

Run::

    uv run --no-sync python scripts/build_statefarm_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

SF_ROOT = Path("datasets/state-farm-distracted-driver-detection")
OUT_DEFAULT = Path("datasets/distraction_sf")

SF_CLASS_MAP: dict[str, str] = {
    "c0": "safe",
    "c1": "phone",
    "c2": "phone",
    "c3": "phone",
    "c4": "phone",
    "c5": "distracted",
    "c6": "drink",
    "c7": "distracted",
    "c8": "distracted",
    "c9": "distracted",
}
PROJECT_CLASSES = ["safe", "phone", "drink", "distracted"]
VAL_FRACTION = 0.20


def build(out_dir: Path, reset: bool) -> None:
    if reset and out_dir.exists():
        shutil.rmtree(out_dir)

    csv_path = SF_ROOT / "driver_imgs_list.csv"
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    # Subject-aware split
    subjects = sorted(set(r["subject"] for r in rows))
    n_val = max(1, round(len(subjects) * VAL_FRACTION))
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])
    print(f"Subjects: {len(subjects)} total | "
          f"{len(train_subjects)} train | {len(val_subjects)} val")
    print(f"Val subjects: {sorted(val_subjects)}")

    # Build subject → list[(sf_class, img_name)]
    by_subject: dict[str, list] = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append((r["classname"], r["img"]))

    counts: dict[str, dict[str, int]] = {"train": defaultdict(int), "val": defaultdict(int)}

    for split_name, split_subjects in [("train", train_subjects), ("val", val_subjects)]:
        for subject in split_subjects:
            for sf_cls, img_name in by_subject[subject]:
                proj_cls = SF_CLASS_MAP.get(sf_cls)
                if proj_cls is None:
                    continue
                src = SF_ROOT / "imgs" / "train" / sf_cls / img_name
                if not src.exists():
                    continue
                dst_dir = out_dir / split_name / proj_cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / img_name)
                counts[split_name][proj_cls] += 1

    print("\nDataset written to:", out_dir)
    for split in ("train", "val"):
        total = sum(counts[split].values())
        breakdown = " | ".join(f"{c}: {counts[split][c]}" for c in PROJECT_CLASSES)
        print(f"  {split}: {total} images  ({breakdown})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    build(Path(args.out).resolve(), args.reset)


if __name__ == "__main__":
    main()
