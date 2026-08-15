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

DO NOT RUN THIS (or the project) FROM A ONEDRIVE-SYNCED FOLDER
--------------------------------------------------------------
A venv is tens of thousands of small files. OneDrive keeps handles open
while it uploads them and, with Files On-Demand, replaces unused ones
with cloud placeholders that must be rehydrated on read. Both break
package installs with ``PermissionError: [WinError 5] Access is
denied``, at a different file each time. Training makes it worse: the
population pipeline writes 420 checkpoints and metric CSVs, and the
dataset is ~1 GB.

Keep the repo on a local path (``C:\\dev\\...``). Note that a venv
cannot simply be MOVED there -- ``.venv/pyvenv.cfg`` and the console
scripts hold absolute paths -- so delete ``.venv`` and re-run ``uv
sync`` after relocating.

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
    failed: list[tuple[pathlib.Path, str]] = []
    for t in targets:
        print(f"[setup] removing {t.relative_to(site)}")
        # NOT ignore_errors=True: a half-deleted cv2/ tree is worse than no
        # deletion at all, because the reinstall then lands on top of it and the
        # next failure surfaces from inside uv, several steps from the cause.
        # onexc, not onerror: onerror is deprecated in 3.12, which this project pins.
        shutil.rmtree(t, onexc=lambda f, p, e: failed.append((pathlib.Path(p), str(e))))
    if failed:
        print("\n[setup] FAILED to remove:")
        for p, err in failed:
            print(f"    {p}\n        {err}")
        raise SystemExit(
            "\n[setup] Something is holding these files open. On Windows the usual\n"
            "causes, in order of likelihood:\n"
            "  1. ONEDRIVE is syncing this folder. A venv is tens of thousands of\n"
            "     small files; OneDrive keeps handles open and turns unused ones into\n"
            "     cloud placeholders, so installs hit 'Access is denied' at random.\n"
            "     Pause syncing from the tray icon and re-run. This project should\n"
            "     NOT live inside a OneDrive folder at all -- see the module docstring.\n"
            "  2. An editor, terminal or python.exe still has the venv loaded.\n"
            "  3. Antivirus is scanning a freshly written DLL.\n"
            "If it persists, delete the listed paths by hand and re-run.")


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
    print(
        "\n[setup] done.\n"
        "[setup] THIS INSTALL IS UNDONE BY ANY PLAIN `uv run`. That command re-syncs\n"
        "        against the lockfile, which pins torch to the CPU index, so it\n"
        "        silently reinstalls +cpu over what was just installed. The failure is\n"
        "        invisible -- training simply runs on the CPU.\n"
        "\n"
        "        Make it structural instead of remembering a flag (uv reads --no-sync\n"
        "        from the environment):\n"
        "\n"
        "            PowerShell, once, persists for this user:\n"
        "            [Environment]::SetEnvironmentVariable('UV_NO_SYNC','1','User')\n"
        "\n"
        "            bash, this session only:\n"
        "            export UV_NO_SYNC=1\n"
        "\n"
        "        Trade-off: `uv run` then NEVER syncs, so after changing dependencies\n"
        "        you must run `uv sync` yourself -- and that re-reverts torch, so\n"
        "        re-run this script afterwards. On a training box that is the safer\n"
        "        default.\n"
        "\n"
        "[setup] Verify at any time with:  uv run --no-sync python scripts/bench_gpu.py")


if __name__ == "__main__":
    sys.exit(main())
