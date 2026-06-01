"""Fine-tune Ultralytics YOLO26 on a custom driver-distraction dataset.

Two training modes are supported via ``--task``:

**classify** (default for State Farm / whole-image datasets)::

    datasets/distraction_sf/
      train/  safe/  phone/  drink/  distracted/
      val/    safe/  phone/  drink/  distracted/

    uv run --no-sync python -m ProVoice.train_distraction \\
        --task classify \\
        --data datasets/distraction_sf \\
        --weights yolo26s-cls.pt \\
        --epochs 50 --imgsz 224 --batch 64 --device 0

**detect** (for datasets with per-object bounding boxes)::

    dataset/
      images/  train/ *.jpg   val/ *.jpg
      labels/  train/ *.txt   val/ *.txt

    uv run --no-sync python -m ProVoice.train_distraction \\
        --task detect \\
        --data datasets/distraction_v2/dataset.yaml \\
        --weights yolo26s.pt \\
        --epochs 150 --imgsz 832 --batch 8 --device 0

The best checkpoint is saved as ``<project>/<name>/weights/best.pt``
(detect) or ``<project>/<name>/weights/best.pt`` (classify).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 for driver distraction detection.")
    parser.add_argument("--task", default="classify", choices=["classify", "detect"],
                        help="Training task: classify (whole-image) or detect (bounding boxes). "
                             "Default: classify (matches State Farm dataset).")
    parser.add_argument("--data", required=True,
                        help="Dataset dir (classify) or YAML path (detect).")
    parser.add_argument("--weights", default=None,
                        help="Starting weights. Defaults: yolo26n-cls.pt (classify) or "
                             "yolo26n.pt (detect).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Image size. Defaults: 224 (classify) or 640 (detect).")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size. Defaults: 64 (classify) or 16 (detect).")
    parser.add_argument("--device", default=None, help="cuda device, 'cpu', or None for auto.")
    parser.add_argument("--project", default="runs", help="Output project directory.")
    parser.add_argument("--name", default="distraction", help="Run name under --project.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience in epochs (0 disables).")
    parser.add_argument("--cos-lr", action="store_true",
                        help="Use a cosine learning-rate schedule.")
    parser.add_argument("--cache", default=None,
                        help="Cache images: 'ram', 'disk', or None.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last interrupted run.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("ultralytics not installed. Run `uv sync`.") from exc

    # Apply task-specific defaults
    is_cls = args.task == "classify"
    weights = args.weights or ("yolo26n-cls.pt" if is_cls else "yolo26n.pt")
    imgsz  = args.imgsz  or (224 if is_cls else 640)
    batch  = args.batch  or (64  if is_cls else 16)

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    model = YOLO(weights)
    print(f"[train] task={args.task}  weights={weights}  data={data_path}")
    print(f"[train] epochs={args.epochs}  imgsz={imgsz}  batch={batch}")

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cos_lr=args.cos_lr,
        resume=args.resume,
    )
    if args.cache:
        train_kwargs["cache"] = args.cache

    model.train(**train_kwargs)
    print(f"[train] done → {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
