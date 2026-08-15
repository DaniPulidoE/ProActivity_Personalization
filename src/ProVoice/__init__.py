"""ProVoice package.

``main`` is exposed LAZILY (PEP 562). It used to be imported eagerly here::

    from .main import main

which meant that importing *any* ProVoice submodule ran the whole live-experiment
import chain — ``main`` -> ``data_collector`` -> ``perception`` -> ``ultralytics``
-> ... Measured cost of ``import ProVoice.models.train_XLSTM`` under that line:

    5,359 modules, 13.9 s, and these native stacks mapped into the process:
    ultralytics, mediapipe, cv2, onnxruntime, torchvision, matplotlib,
    fastapi, socketio, dash, carla, pyarrow

None of which a trainer, a sweep or an offline analysis touches. Three concrete
costs, in increasing order of seriousness:

1. **Time.** ~14 s of imports per process, and the population sweep launches 180
   of them — about 40 minutes of pure overhead.
2. **A network call at import time.** ``ultralytics.utils`` runs ``is_online()``
   (a DNS lookup) while being imported, so every offline training run reached
   for the network before doing anything.
3. **Native-DLL surface.** Loading CUDA, MKL/OpenMP, OpenCV, ONNX Runtime,
   MediaPipe and the CARLA client into one process is the classic Windows
   setup for DLL and OpenMP-runtime conflicts. This project already has a
   documented history of exactly that failure mode — see the "Vehicle-State
   Bridge" note in CLAUDE.md, where CARLA-in-process corrupted the heap and the
   crashes surfaced in unrelated threads (YOLO convolutions, encode_frame, a
   CPython dict lookup). Observed here as 0xc0000005 access violations
   attributed to ``arrow.dll``, a library nothing in the training path calls.

``python -m ProVoice`` is unaffected: ``__main__.py`` imports ``ProVoice.main``
directly and never relied on this re-export.
"""
import sys as _sys
from typing import Any

__all__ = ["main"]


def _force_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 on Windows. Explicit, because it used to be luck.

    Modules all over this package print non-ASCII — arrows in the data-contract
    lines, lambda/tau in the adaptation logs, en-dashes in warnings. Windows
    gives a piped (non-console) stream the ANSI code page, cp1252 here, so the
    first such print raises UnicodeEncodeError and kills the process. That is
    exactly what happens when the sweep runs a trainer under
    ``subprocess.run(capture_output=True)``: 36 runs, 36 instant failures.

    It never surfaced before because ``ultralytics.utils`` reconfigures stdout to
    UTF-8 on Windows as a side effect of being imported, and the old eager
    ``from .main import main`` here dragged ultralytics into every process. So a
    computer-vision library was, by accident, the only thing keeping the trainer
    printable. Making the import lazy removed 12 s of startup and most of the
    native-DLL surface — and took that accident with it.

    ``errors="replace"`` so this can never itself become the crash; a mangled
    character in a log line is strictly better than a dead training run.
    """
    for stream in ("stdout", "stderr"):
        s = getattr(_sys, stream, None)
        if s is None or not hasattr(s, "reconfigure"):
            continue
        try:
            if (getattr(s, "encoding", "") or "").lower().replace("-", "") != "utf8":
                s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass      # detached/replaced stream — nothing to do, and not fatal


_force_utf8_streams()


def __getattr__(name: str) -> Any:
    """Resolve ``ProVoice.main`` on first access instead of at import time."""
    if name == "main":
        from .main import main as _main
        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
