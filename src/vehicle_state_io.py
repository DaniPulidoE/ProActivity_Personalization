"""Locked single-file channel for vehicle state, shared by the writer
(scripts/vehicle_state_file_bridge.py) and the reader (ProVoice's
DataCollector._poll_vehicle_state).

WHY NOT tmp-file + os.replace, the usual atomic-write recipe:

    On Windows os.replace() fails with a sharing violation (WinError 5) when
    another process holds the destination open, because Python's open() does
    not pass FILE_SHARE_DELETE. With a writer at 20 Hz and a reader at 20 Hz
    those windows overlap constantly, so the "atomic" publish would fail
    several times a second and the reader would keep seeing stale data.

    So: ONE file, written in place, with an advisory lock held across the whole
    write and the whole read. The reader can then never observe a torn record.

WHY NOT a lock-free single write() + checksum:

    A single small write() is *usually* atomic but is not guaranteed to be, and
    "usually" is not a property to build a participant's vehicle data on.

WHY THIS LIVES IN src/ RATHER THAN src/ProVoice/:

    The bridge process must stay lightweight and must NOT import the ProVoice
    package -- ProVoice/__init__.py pulls in main.py, and with it torch, cv2,
    mediapipe and the whole perception stack. A standalone module on the same
    sys.path root is importable from both sides with no such cost.

The lock is a single byte at offset 0 used purely as a mutex; the file's
contents are never inside the locked range, which keeps the semantics identical
on Windows (msvcrt, byte-range) and POSIX (flock, whole-file).
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

# One byte at offset 0, used only as a mutex. Windows allows locking a range
# that extends beyond EOF, so this works even on a freshly created file.
_LOCK_OFFSET = 0
_LOCK_LEN = 1
_RETRY_S = 0.002        # ~2 ms: a write holds the lock for tens of microseconds
_DEFAULT_TIMEOUT = 0.5  # bail out rather than stall a 20 Hz loop

if os.name == "nt":
    import msvcrt

    def _try_lock(fh) -> bool:
        fh.seek(_LOCK_OFFSET)
        try:
            # LK_NBLCK, not LK_LOCK: LK_LOCK retries internally for ~10 s,
            # which would freeze the collection loop on contention.
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, _LOCK_LEN)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        fh.seek(_LOCK_OFFSET)
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, _LOCK_LEN)
        except OSError:
            pass
else:
    import fcntl

    def _try_lock(fh) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


@contextmanager
def file_lock(fh, timeout: float = _DEFAULT_TIMEOUT):
    """Hold the advisory lock, or raise TimeoutError rather than block forever."""
    deadline = time.monotonic() + timeout
    while not _try_lock(fh):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"vehicle-state file lock not acquired within {timeout:.2f}s")
        time.sleep(_RETRY_S)
    try:
        yield
    finally:
        _unlock(fh)


class VehicleStateChannel:
    """Persistent handle onto the shared file.

    The handle is opened ONCE and reused: at 20 Hz, reopening on every access
    would add 40 file opens a second for no benefit. Both roles open "r+b" --
    the reader needs write access because a byte-range lock cannot be taken on
    a read-only handle on Windows.
    """

    __slots__ = ("path", "_fh")

    def __init__(self, path: str, create: bool = False):
        self.path = os.path.abspath(path)
        self._fh = None
        if create:
            # Create empty if absent, without truncating an existing file that
            # a reader may already be holding.
            if not os.path.exists(self.path):
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.path, "a+b"):
                    pass
            self._open()

    def _open(self) -> None:
        if self._fh is None or self._fh.closed:
            self._fh = open(self.path, "r+b")

    def publish(self, state: dict) -> None:
        """Write one record. Raises on I/O or lock failure; caller decides."""
        self._open()
        data = json.dumps(state).encode("utf-8")
        with file_lock(self._fh):
            self._fh.seek(0)
            self._fh.write(data)
            self._fh.truncate()
            # flush() pushes it out of Python's buffer into the OS page cache,
            # which is all another process needs. No fsync: forcing a disk
            # write 20 times a second would be real I/O for no gain, since we
            # never need this file to survive a power cut.
            self._fh.flush()

    def read(self) -> dict:
        """Return the latest record. Raises if absent, locked out, or invalid."""
        self._open()
        with file_lock(self._fh):
            self._fh.seek(0)
            raw = self._fh.read()
        if not raw:
            raise ValueError("vehicle-state file is empty (bridge not publishing yet)")
        return json.loads(raw.decode("utf-8"))

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_exc):
        self.close()
        return False
