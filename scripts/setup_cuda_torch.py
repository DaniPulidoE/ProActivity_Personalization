"""Install the CUDA 12.8 PyTorch build into the project venv.

Why this is a separate step (and not just a `uv sync`):

The `mmrphys` git dependency pins `torch`/`torchvision` to the CPU
wheel index in its own ``[tool.uv.sources]``. uv refuses to resolve a
single lockfile that mixes the CPU and CUDA indexes for the same
package, so ``pyproject.toml`` keeps the CPU pin (that is what makes
``uv lock``/``uv sync`` succeed). On an NVIDIA box you then run this
script once to overlay the CUDA build on top of the synced env.

It also repairs the OpenCV install: ``ultralytics`` depends on
``opencv-python`` while ``mediapipe`` depends on
``opencv-contrib-python``; installing both clobbers the shared ``cv2/``
directory and breaks ``import cv2``. We keep only the contrib build
(a superset that satisfies both).

Usage::

    uv sync                                  # CPU torch, resolvable lock
    uv run --no-sync python scripts/setup_cuda_torch.py   # overlay CUDA

After this, launch GPU work with ``uv run --no-sync ...`` (NOT plain
``uv run``, which would re-sync and revert torch to the CPU build).
"""

from __future__ import annotations

import subprocess
import sys

CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
OPENCV_PIN = "opencv-contrib-python==4.11.0.86"


def run(cmd: list[str]) -> None:
    print("[setup] $", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    # 1. Single, clean OpenCV (contrib superset) — removes the
    #    opencv-python / opencv-contrib-python collision.
    run(["uv", "pip", "uninstall", "opencv-python", "opencv-contrib-python"])
    run(["uv", "pip", "install", OPENCV_PIN])

    # 2. CUDA 12.8 torch/torchvision (RTX 5080 / Blackwell sm_120).
    run(["uv", "pip", "install", "--reinstall", "torch", "torchvision",
         "--index-url", CUDA_INDEX])

    # 3. Verify.
    code = (
        "import torch, cv2;"
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available());"
        "print('device', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None;"
        "print('cv2', cv2.__version__, 'imshow', hasattr(cv2,'imshow'));"
        "from ultralytics import YOLO; print('ultralytics OK')"
    )
    run(["uv", "run", "--no-sync", "python", "-c", code])
    print("[setup] done. Launch GPU work with `uv run --no-sync ...`.")


if __name__ == "__main__":
    sys.exit(main())
