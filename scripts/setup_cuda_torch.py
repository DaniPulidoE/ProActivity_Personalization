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

That repair CANNOT go through ``uv pip uninstall``. The same clobbering
that breaks ``import cv2`` also leaves one distribution's metadata
incomplete, and uv refuses to uninstall what it cannot inventory::

    error: Failed to uninstall package; `RECORD` file not found at:
        .venv/Lib/site-packages/opencv_python-4.x.x.dist-info/RECORD

Because that used to be step 1 under ``check_call``, the failure aborted
the script BEFORE the CUDA torch install — so the visible symptom was a
training run pinned at 100% CPU with an idle GPU, with the actual cause
several steps upstream and about OpenCV. ``purge_opencv()`` deletes the
directories directly instead, which needs no metadata to be intact.

Usage::

    uv sync                                  # CPU torch, resolvable lock
    uv run --no-sync python scripts/setup_cuda_torch.py   # overlay CUDA

After this, launch GPU work with ``uv run --no-sync ...`` (NOT plain
``uv run``, which would re-sync and revert torch to the CPU build).
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import sysconfig

CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
OPENCV_PIN = "opencv-contrib-python==4.11.0.86"


def run(cmd: list[str]) -> None:
    print("[setup] $", " ".join(cmd))
    subprocess.check_call(cmd)


def purge_opencv() -> None:
    """Delete both OpenCV distributions by removing their files directly.

    ``uv pip uninstall`` CANNOT be used here, and this is not a style preference.
    ``opencv-python`` (via ultralytics) and ``opencv-contrib-python`` (via
    mediapipe) install into the SAME ``cv2/`` directory, so whichever is installed
    second overwrites the first's files and leaves its metadata inconsistent —
    typically a ``.dist-info`` whose ``RECORD`` is gone. uv then refuses to
    uninstall a distribution it cannot inventory::

        error: Failed to uninstall package; `RECORD` file not found at:
            .venv/Lib/site-packages/opencv_python-4.x.x.dist-info/RECORD

    and ``check_call`` aborts the whole script BEFORE the CUDA torch install ever
    runs — which is why the symptom is "no GPU" rather than "no OpenCV".

    Removing the dist-info directories and the shared ``cv2/`` tree is exactly
    the operation the uninstall was standing in for, and it works whichever of
    the two clobbered the other. Everything removed is reinstalled from the index
    on the next line, so this is recoverable, not destructive.
    """
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    targets = sorted(site.glob("opencv_python-*.dist-info")) \
        + sorted(site.glob("opencv_contrib_python-*.dist-info")) \
        + sorted(site.glob("opencv_python_headless-*.dist-info"))
    cv2_dir = site / "cv2"
    if cv2_dir.is_dir():
        targets.append(cv2_dir)
    if not targets:
        print("[setup] no existing OpenCV install found — nothing to purge")
        return
    for t in targets:
        print(f"[setup] removing {t.relative_to(site)}")
        shutil.rmtree(t, ignore_errors=True)


def main() -> None:
    # 1. Single, clean OpenCV (contrib superset) — removes the
    #    opencv-python / opencv-contrib-python collision.
    purge_opencv()
    run(["uv", "pip", "install", "--reinstall", OPENCV_PIN])

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
