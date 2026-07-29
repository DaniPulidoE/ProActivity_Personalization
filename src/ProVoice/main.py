from __future__ import annotations

import signal
import sys
import time
import os
import uuid
import argparse as ap
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

# ── CPU thread budget ────────────────────────────────────────────────────────
# Must run BEFORE numpy/torch/cv2 are imported: the OpenMP/MKL runtimes size
# their thread pools once, at load time, and ignore these variables afterwards.
#
# Why cap them at all: this process shares the machine with the CARLA server and
# the Drive client. CARLA runs ASYNCHRONOUSLY — drive_improved.py only calls
# sim_world.tick() under `if args.sync:`, `--sync` defaults to False, and
# start_experiment.py never passes it, so Drive takes the wait_for_tick() branch
# and the simulator advances on its own clock. That means an all-core burst here
# does not stall a tick Drive owes the server; it starves CARLA's render and
# physics threads and Drive's render loop directly, which the participant sees
# as dropped frames and laggy steering. Either way the cap is what stops this
# process from taking the whole machine.
#
# (An earlier version of this comment asserted synchronous mode with
# fixed_delta_seconds=0.05. That was wrong — check the `--sync` flag before
# relying on any claim about tick ownership.)
#
# Left at its default, torch sizes its intra-op pool to the core count (20 on
# the lab machine) and every xLSTM forward becomes an all-core stampede.
# Measured on a full 320-frame window:
#
#   threads   forward wall   cores busy
#      20         5.6 ms        19.2      <- default
#       4         3.6 ms         4.6
#       2         5.1 ms         2.0      <- default here
#       1         8.1 ms         1.0
#
# 20 threads buys NOTHING in wall-clock over 2 (the ops are tiny; fork/join
# across 20 cores costs more than it saves) while consuming ~10x the CPU. At
# decision_hz=4 that is four all-core bursts per second, which is exactly the
# lag that appears once calibration ends and the decision thread starts running
# full windows. KMP_BLOCKTIME/OMP_WAIT_POLICY stop the idle OpenMP workers
# spin-waiting between bursts, which otherwise keeps the cores hot in between.
PV_NUM_THREADS = os.getenv("PV_NUM_THREADS", "2")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, PV_NUM_THREADS)
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Crash diagnostics ────────────────────────────────────────────────────────
# This process has died with STATUS_ACCESS_VIOLATION (0xC0000005 -> exit code
# 3221225477) inside native code. A hard fault like that kills the interpreter
# outright: no traceback, no excepthook, nothing on stdout past the last normal
# print. faulthandler installs an OS-level handler that dumps the Python stack
# of EVERY thread at the moment of the fault, which turns "PROVOICE exited with
# code 3221225477" into a named file and line.
#
# Enabled here — before torch/cv2/mediapipe are imported — so a fault during
# library init is covered too. The log is opened line-buffered and kept alive
# for the whole process: faulthandler writes to the raw fd, so the object must
# not be garbage-collected (hence the module-level name).
_FAULT_LOG = None
try:
    import faulthandler
    _log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs")
    os.makedirs(_log_dir, exist_ok=True)
    _fault_path = os.path.join(_log_dir, "provoice_faults.log")
    _FAULT_LOG = open(_fault_path, "a", buffering=1, encoding="utf-8", errors="replace")
    _FAULT_LOG.write(f"\n===== ProVoice start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"pid={os.getpid()} =====\n")
    faulthandler.enable(file=_FAULT_LOG, all_threads=True)
    print(f"[main] faulthandler active -> {_fault_path}")
except Exception as _e:
    print(f"[main] could not enable faulthandler: {_e}")

import uvicorn

try:
    import carla
    HAS_CARLA = True
except Exception:
    carla = None  # type: ignore
    HAS_CARLA = False

from ProVoice.data_collector import DataCollector
from ProVoice.logger import Logger
from ProVoice.decision_engine import (
    CombinedFusionStrategy,
    XGBoostLoAStrategy,
    StateLevelsLoAStrategy,
    StateXLSTMLoAStrategy,
)
from ProVoice.provoice_actuator import ProVoiceActuator
import ProVoice.webui.app as dashboard

# The env vars above configure the OpenMP/MKL runtimes; these are torch's own
# knobs and are authoritative for intra-op (per-operator) and inter-op parallelism.
# Set here, after the imports that pull torch in, because torch is not imported
# directly by this module. set_num_interop_threads() throws once any inter-op
# work has started, so a failure is non-fatal and simply leaves the default.
try:
    import torch as _torch
    _torch.set_num_threads(int(PV_NUM_THREADS))
    try:
        _torch.set_num_interop_threads(int(PV_NUM_THREADS))
    except Exception:
        pass
    print(f"[main] torch CPU threads capped at {_torch.get_num_threads()} "
          f"(PV_NUM_THREADS={PV_NUM_THREADS}) to leave cores for CARLA.")
except Exception as _e:
    print(f"[main] could not cap torch threads: {_e}")

try:
    import cv2 as _cv2
    # OpenCV runs its own thread pool, independent of OpenMP's. At 640x480 the
    # colour conversions and resizes on the capture path are single-thread work
    # anyway, so this costs nothing and removes another all-core consumer.
    _cv2.setNumThreads(int(PV_NUM_THREADS))
except Exception as _e:
    print(f"[main] could not cap OpenCV threads: {_e}")

# import ProVoice.logo as logo

# logo.print_mech()
# fallback: LoA0
class LoAZeroFallback:
    def __init__(self, reason: str = "fallback LoA0"):
        self.reason = reason

    def decide(self, data: dict) -> dict:
        return {
            "action": "manual_control",
            "level": "low",
            "LoA": 0,
            "message": self.reason,
            "probs": [1.0, 0.0, 0.0, 0.0, 0.0],
            "fallback": True,
        }


import re as _re

_ARG_KEY_RE = _re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _normalize_argv(tokens):
    """Rewrite bare ``key=value`` tokens into ``--key=value`` so argparse can
    parse BOTH styles the project uses: ``key=value`` (README examples and
    ``start_experiment.py``) and ``--flag value`` (CLAUDE.md). Underscores in
    the key become dashes to match the dash-form option strings. Tokens whose
    part before ``=`` is not a bare identifier (e.g. a ``http://…?a=b`` value)
    are left untouched.
    """
    out = []
    for tok in tokens:
        t = tok.strip().strip(",")
        if not t:
            continue
        if not t.startswith("-") and "=" in t and _ARG_KEY_RE.match(t.split("=", 1)[0]):
            k, v = t.split("=", 1)
            out.append(f"--{k.replace('_', '-')}={v}")
        else:
            out.append(t)
    return out


# ── Camera selection ─────────────────────────────────────────────────────────
# There is no camera *identity* anywhere in this pipeline: DataCollector passes
# whatever it is given straight to cv2.VideoCapture, and an index means only
# "the n-th device this OpenCV backend enumerated". On Windows that order
# normally puts a laptop's built-in camera at 0 and a USB webcam after it, so
# --webcam takes the lowest working index above 0. That is a CONVENTION, not a
# guarantee (see the docstrings below) -- the probe prints the full map so the
# operator can see what was picked and pin it with camera_source=N instead.
_CAMERA_PROBE_RANGE = 5


def _probe_cameras(max_index: int = _CAMERA_PROBE_RANGE) -> list[int]:
    """Open every index in ``range(max_index)`` and report which ones answer.

    Probing goes through the SAME (default) backend DataCollector opens with.
    This matters: an index is only meaningful relative to a backend, and on
    Windows MSMF and DSHOW enumerate devices separately, so probing with one
    and opening with the other can land on a different physical camera.

    Each probe is released immediately, and all of them before this returns --
    the device has to be free again when DataCollector opens it moments later.
    A failing index is slow on Windows (the backend waits for a device that
    isn't there), so this costs a few seconds at startup.
    """
    try:
        import cv2
    except Exception as e:  # noqa: BLE001 — no OpenCV means no camera at all
        print(f"[camera] cannot probe: OpenCV unavailable ({e})")
        return []

    found: list[int] = []
    for i in range(max_index):
        cap = None
        ok = False
        try:
            cap = cv2.VideoCapture(i)
            ok = bool(cap.isOpened())
        except Exception as e:  # noqa: BLE001 — a bad index must not be fatal
            print(f"[camera] index {i}: probe raised {e!r}")
        finally:
            if cap is not None:
                cap.release()
        print(f"[camera] index {i}: {'OPEN' if ok else 'unavailable'}")
        if ok:
            found.append(i)
    return found


def _select_webcam(max_index: int = _CAMERA_PROBE_RANGE) -> int:
    """Lowest working camera index ABOVE 0 — the external webcam on a laptop.

    Falls back to 0 with a loud warning when nothing else answers, because the
    alternative (an unopenable index) fails silently: cv2.VideoCapture does not
    raise on a missing device, so DataCollector would keep visual_enabled=True
    and run the whole session blind.
    """
    found = _probe_cameras(max_index)
    # The probe just closed the device DataCollector is about to open, and
    # Windows does not always free a capture device the instant release()
    # returns. The gap between here and cv2.VideoCapture() downstream is a few
    # milliseconds, so give the driver a moment rather than racing it.
    time.sleep(1.0)
    external = [i for i in found if i > 0]
    if external:
        chosen = external[0]
        if len(external) > 1:
            print(f"[camera] --webcam: {len(external)} cameras above index 0 "
                  f"({external}); taking the lowest. Pin one with camera_source=N "
                  f"if this is the wrong device.")
        print(f"[camera] --webcam: selected index {chosen}")
        return chosen
    print(f"[camera] --webcam: NO camera opened above index 0 "
          f"(working indices: {found or 'none'}). Falling back to index 0, the "
          f"built-in camera — CHECK THIS IS THE CAMERA YOU MEANT before driving.")
    return 0


def _build_parser() -> ap.ArgumentParser:
    p = ap.ArgumentParser(
        prog="ProVoice.main",
        description="ProVoice decision engine + dashboard.",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--participantid", default="")
    p.add_argument("--environment", default="")
    p.add_argument("--secondary-task", dest="secondary_task", default="")
    p.add_argument("--functionname", default="Adjust seat positioning")
    p.add_argument("--emotion", "--affect", dest="emotion", default="")
    p.add_argument("--modeltype", default="combined",
                   help="fcd | state | combined | collection")
    p.add_argument("--state-model", "--statemodel", dest="state_model", default="xlstm",
                   help="classic | xlstm")
    p.add_argument("--w-fcd", dest="w_fcd", type=float, default=0.7)
    p.add_argument("--session-id", dest="session_id", default=None)
    p.add_argument("--window", type=int, default=400,
                   help="Frame-count cap on the model input window (safety bound).")
    p.add_argument("--window-seconds", dest="window_seconds", type=float, default=None,
                   help="Time span (s) of the xLSTM window; unset inherits the checkpoint's.")
    p.add_argument("--decision-hz", dest="decision_hz", type=float, default=4.0,
                   help="Rate of the decision thread, decoupled from data collection. "
                        "Sets the decisions.csv row rate; capped by the achieved "
                        "collection rate (one decision per distinct frame).")
    p.add_argument("--camera-source", dest="camera_source", default="front")
    p.add_argument("--camera-url", dest="camera_url", default="udp://127.0.0.1:8554")
    p.add_argument("--webcam", action="store_true",
                   help="Probe camera indices 0..%d and use the lowest that opens "
                        "above 0 — the external webcam on a laptop, where 0 is the "
                        "built-in one. Only fills in for the DEFAULT camera source; "
                        "an explicit camera_source= still wins. Adds a few seconds "
                        "to startup and is a convention, not a guarantee: check the "
                        "printed map." % (_CAMERA_PROBE_RANGE - 1))
    p.add_argument("--vehicle-id", dest="vehicle_id", default=None,
                   help="Skip vehicle_id.txt discovery when set.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--carla-timeout", dest="carla_timeout", type=float, default=10.0)
    p.add_argument("--vehicle-state-url", dest="vehicle_state_url", default=None)
    p.add_argument("--vehicle-state-file", dest="vehicle_state_file", default=None,
                   help="Path published by scripts/vehicle_state_file_bridge.py. "
                        "Like --vehicle-state-url it means THIS process makes no "
                        "CARLA calls at all, but it reads a file instead of a "
                        "socket, keeping the network stack out of the loop.")
    p.add_argument("--status-url", dest="status_url", default=None,
                   help="Reverse bridge on the CARLA machine "
                        "(scripts/provoice_status_server.py): this process posts "
                        "'collection_started' at its first logged frame and "
                        "'provoice_ended' when it exits, so Drive over there "
                        "knows when to open its LoA windows and when to stop the "
                        "car. Normally left unset and adopted from the vehicle "
                        "bridge's /session.")
    p.add_argument("--bridge-session-timeout", dest="bridge_session_timeout",
                   type=float, default=10.0,
                   help="How long to wait for the remote bridge to publish a "
                        "session id AND a participant id at /session. Only used "
                        "with vehicle_state_url. A bridge that answers with "
                        "neither is still starting up, so that counts as not "
                        "answered. 0 skips the fetch entirely, leaving both "
                        "exactly as given here.")
    p.add_argument("--state-poll-hz", dest="state_poll_hz", type=float, default=2.0,
                   help="How often to poll vehicle_state_url. The 2 Hz default "
                        "is the historical one; start_experiment.py --vehicle-bridge "
                        "raises it to match the collection loop, because at 2 Hz "
                        "steer/brake are sampled ten times slower than a direct "
                        "CARLA read and are visibly degraded.")
    p.add_argument("--log-path", dest="log_path", default=None,
                   help="JSONL log of features fed to the xLSTM; unset disables.")
    p.add_argument("--calibration-only", dest="calibration_only", action="store_true",
                   help="Run the 180 s calibration, store the baseline for this "
                        "participant, then exit. Always measures fresh, ignoring any "
                        "stored calibration.")
    p.add_argument("--data-collection", dest="data_collection", action="store_true",
                   help="Data-collection run: record raw data only. No decision engine "
                        "is loaded or run, and the live calibration is skipped (this "
                        "participant's stored baseline is reused, or neutral defaults "
                        "if there is none). Mutually exclusive with --calibration-only.")
    return p


def _parse_args(argv):
    parser = _build_parser()
    args, unknown = parser.parse_known_args(_normalize_argv(argv))
    if unknown:
        # Surface stray tokens instead of silently dropping them (the old
        # key=value parser ignored anything without an '=').
        print(f"[main] ignoring unrecognized argument(s): {unknown}")
    if args.calibration_only and args.data_collection:
        parser.error("--calibration-only and --data-collection are mutually "
                     "exclusive: one runs the calibration and nothing else, the "
                     "other skips it and only records data.")
    return args

def read_vehicle_id(path: str | None = None, wait_seconds: float = 10.0) -> int | None:
    """
    Read the vehicle ID from vehicle_id.txt in the project root (written by the Drive UI).
    """
    # Project root = two levels up from src/ProVoice/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # vehicle id file path
    default_path = os.path.join(project_root, "vehicle_id.txt")

    # Use default_path if path is not specified
    path = path or default_path

    print(f"[INFO] Waiting for vehicle id file at: {path}")

    deadline = time.time() + float(wait_seconds)
    while time.time() < deadline:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    raw = f.read().strip()
                if raw:
                    try:
                        vid = int(raw)
                        print(f"[INFO] Read vehicle id {vid} from {path}")
                        return vid
                    except ValueError:
                        print(f"[WARN] Invalid vehicle id content: {raw!r}")
                else:
                    print(f"[WARN] vehicle_id file {path} empty, waiting...")
        except Exception as e:
            print(f"[WARN] Error reading vehicle id file {path}: {e}")

        time.sleep(0.1)

    print(f"[WARN] vehicle_id file not found at {path} after {wait_seconds}s")
    return None


def fetch_bridge_session(url: str, timeout: float = 10.0,
                         request_timeout: float = 3.0) -> dict | None:
    """Wait up to ``timeout`` for {session_id, participantid} from /session.

    In a --remote run the launcher on the CARLA machine mints the session id and
    holds the participant id, and THIS process has to agree with both or the
    session's LoA labels and raw frames end up filed under different names. The
    bridge publishes them; this reads them back, so neither id depends on a
    human retyping it into a second terminal.

    It waits for the IDS, not merely for the endpoint. Those are different
    events: the bridge starts serving as soon as its socket is open, and a
    /session that answers 200 with two nulls is a bridge that is up but does not
    yet know who is driving. Returning on the first answer would take those
    nulls as the truth and mint a local session id a second later — the exact
    split-identity failure this function exists to prevent. So a nulls-only
    answer is treated like no answer at all and polled again.

    The wait also covers the ordinary case of the two machines being started by
    hand, in either order: a ProVoice launched first would otherwise get one
    refused connection and fall back for the rest of the session. This runs
    ONCE, at startup, so a plain urlopen is right here — the keep-alive
    machinery in DataCollector exists for the 2 Hz state poll, not for this.

    Returns None when the bridge never answered at all, or answered with
    something unusable (a bridge predating this route 404s). None means "no
    information", never "no session": the caller keeps whatever it already had.
    A partial answer at the deadline is returned as-is rather than discarded —
    one id is better than none, and the caller says out loud what is missing.
    """
    endpoint = url.rstrip("/") + "/session"
    deadline = time.monotonic() + max(0.0, timeout)
    attempt = 0
    last_err: str | None = None
    last_info: dict | None = None
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(endpoint, timeout=request_timeout) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP {resp.status}")
                info = json.loads(resp.read())
            if not isinstance(info, dict):
                raise ValueError(f"expected an object, got {type(info).__name__}")
            last_info = {
                # Normalize "" to None here so the caller has one "unknown".
                "session_id": (info.get("session_id") or None),
                "participantid": (info.get("participantid") or None),
                "status_url": (info.get("status_url") or None),
            }
            if last_info["session_id"] and last_info["participantid"]:
                return last_info
            last_err = ("bridge is up but has published no %s yet"
                        % (" or ".join(
                            k for k in ("session_id", "participantid")
                            if not last_info[k])))
        except urllib.error.HTTPError as e:
            # A routed answer, just not this route: the bridge is older than
            # this feature. Retrying cannot change that, so stop immediately.
            print(f"[bridge] {endpoint} -> HTTP {e.code}; this bridge does not "
                  f"publish session identity. Using the local command line.")
            return None
        except Exception as e:  # noqa: BLE001 — startup convenience, never fatal
            last_err = f"{type(e).__name__}: {e}"
        if time.monotonic() >= deadline:
            break
        if attempt == 1:
            print(f"[bridge] waiting up to {timeout:.0f}s for the session ids at "
                  f"{endpoint} ({last_err}) ...")
        time.sleep(1.0)

    if last_info is not None:
        # Answered, but never with both ids. Hand back what there was: status_url
        # alone is still worth having, and resolve_session_identity() reports
        # each missing id where the operator will see it.
        print(f"[bridge] {endpoint} did not publish both ids within "
              f"{timeout:.0f}s ({last_err}).")
        return last_info
    print(f"[bridge] {endpoint} did not answer within {timeout:.0f}s "
          f"({last_err}). Using the local command line.")
    return None


def _status_channel_alive(status_url: str, timeout: float = 2.0) -> bool:
    """Can this process actually reach the reverse bridge? One cheap GET."""
    try:
        with urllib.request.urlopen(status_url.rstrip("/") + "/health",
                                    timeout=timeout) as resp:
            resp.read()
            return resp.status == 200
    except Exception:  # noqa: BLE001 — unreachable is the answer, not an error
        return False


def resolve_status_url(status_url: str, vehicle_state_url: str | None,
                       timeout: float = 2.0) -> str | None:
    """Confirm the reverse bridge is reachable, repairing the host if it is not.

    The address published at /session is built on the CARLA machine from ITS
    idea of its own IP. That is a guess, and it is wrong in two setups that
    both look fine until the moment it matters:

      * a tunnel (ngrok) in front of the vehicle bridge — the ProVoice machine
        reaches the tunnel, never the LAN address behind it;
      * a CARLA machine with several NICs (Hyper-V, VPN adapters), where the
        interface that routes to 8.8.8.8 is not the one this machine talks to.

    Either way the POST is dropped rather than refused, so it fails by TIMEOUT,
    and the failure that matters is provoice_ended — sent while the process is
    exiting, with nothing left to retry and a participant still driving.

    So: probe it now. If the published address does not answer, try the host
    this process demonstrably reaches the vehicle bridge on, with the published
    port. That covers the multi-NIC case exactly and the LAN-address-behind-a-
    tunnel case whenever the status port is exposed the same way.

    Returns a URL that answered, or None. None is not fatal anywhere — it means
    the operator gets told NOW, at startup, instead of at shutdown.
    """
    if not status_url:
        return None
    if _status_channel_alive(status_url, timeout):
        print(f"[status] reverse bridge reachable at {status_url}")
        return status_url

    print(f"[status] {status_url} did not answer /health within {timeout:.0f}s.")
    candidate = None
    if vehicle_state_url:
        published = urllib.parse.urlsplit(status_url)
        reached = urllib.parse.urlsplit(vehicle_state_url)
        if published.port and reached.hostname and \
                reached.hostname != published.hostname:
            candidate = f"{reached.scheme}://{reached.hostname}:{published.port}"
            print(f"[status] trying {candidate} instead — that is the host this "
                  f"process actually reaches the vehicle bridge on.")
            if _status_channel_alive(candidate, timeout):
                print(f"[status] reverse bridge reachable at {candidate}; using it.")
                return candidate

    print("[status] NO reverse channel. This run will still record everything, "
          "but the CARLA machine will not learn when it starts or ends: Drive "
          "there falls back to its popup-wait timeout and must be stopped by "
          "hand. Fix by allowing the status port inbound on the CARLA machine, "
          "or — behind a tunnel — by exposing it too and passing "
          "status_url=<that url> here.")
    return None


def post_status_event(status_url: str, event: str, session_id: str,
                      participantid: str, reason: str = "",
                      timeout: float = 3.0, attempts: int = 2) -> bool:
    """Tell the CARLA machine that collection started, or that we are done.

    The receiver is scripts/provoice_status_server.py; Drive on that machine
    polls the file it publishes. Both signals are ONE-SHOT and nothing replays
    them, so each is retried briefly -- but never for long, because both are
    sent from paths that must not stall: the first from the collection loop,
    the second from process shutdown.

    Returns whether the server acknowledged. Failure is logged and swallowed:
    ProVoice's job is to record this participant's data, and it must not lose a
    session because the other machine's status port was unreachable. The visible
    cost of a lost signal is on the other end -- Drive falls back to its timeout
    for the start, and to the operator for the end.
    """
    if not status_url:
        return False
    endpoint = status_url.rstrip("/") + "/event"
    body = json.dumps({
        "event": event,
        "session_id": session_id or "",
        "participantid": participantid or "",
        "reason": reason,
        "ts": time.time(),
    }).encode("utf-8")
    last = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            req = urllib.request.Request(
                endpoint, data=body, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    resp.read()
                    print(f"[status] {event} acknowledged by {endpoint}")
                    return True
                last = f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            # 409 is the server telling us we are the wrong session. Retrying
            # cannot fix that, and it is worth saying loudly: it means two
            # sessions are live and one of them is talking to the wrong machine.
            detail = e.read().decode("utf-8", "replace")[:200]
            print(f"[status] {event} REJECTED by {endpoint}: HTTP {e.code} {detail}")
            return False
        except Exception as e:  # noqa: BLE001 — never fatal, see the docstring
            last = f"{type(e).__name__}: {e}"
        if attempt < attempts:
            time.sleep(0.5)
    print(f"[status] could not deliver {event} to {endpoint} ({last}). "
          + ("Drive will fall back to its popup-wait timeout."
             if event == "collection_started"
             else "The other machine will NOT show the end-of-session screen; "
                  "stop the drive there by hand."))
    return False


def resolve_session_identity(session_id_local: str | None,
                             participantid_local: str | None,
                             bridge: dict | None) -> tuple[str | None, str | None]:
    """Reconcile this machine's ids with the ones the bridge publishes.

    Rules, and why:

    * bridge value, nothing local  -> adopt it. This is the point of the whole
      exercise: `vehicle_state_url=...` alone is enough to join the session.
    * local value, no bridge value -> keep it. An unreachable or older bridge
      must not take a run down; that is the same isolation contract the state
      poller works under (see DataCollector._poll_vehicle_state).
    * both, and they AGREE         -> fine, and say so, since "the two machines
      agree" is exactly what the operator wants confirmed.
    * both, and they DISAGREE      -> raise. Not a bridge failure but an
      operator error, with two authoritative-looking answers and no way to pick
      one. Continuing writes this session's data under a name that does not
      match the LoA labels on the other machine, and nothing downstream would
      notice. Stopping costs seconds; the alternative costs the session.
    """
    resolved = []
    for field, local in (("session_id", session_id_local),
                         ("participantid", participantid_local)):
        remote = (bridge or {}).get(field)
        local = local or None
        if remote and local and remote != local:
            raise SystemExit(
                f"[FATAL] {field} mismatch: this command line says {local!r}, the "
                f"vehicle-state bridge says {remote!r}. Those are two different "
                f"sessions and only one of them matches the LoA labels being "
                f"recorded on the CARLA machine. Fix the command line, or drop "
                f"{field} from it and let the bridge supply it "
                f"(bridge_session_timeout=0 skips the check entirely).")
        if remote and not local:
            print(f"[bridge] adopting {field}={remote!r} from the bridge.")
            resolved.append(remote)
        else:
            if remote and local:
                print(f"[bridge] {field}={local!r} confirmed by the bridge.")
            resolved.append(local)
    return resolved[0], resolved[1]


def get_carla_vehicle_by_id(actor_id: int, host: str = "127.0.0.1", port: int = 2000, timeout: float = 2.0, retries: int = 5):
    """
    Connect to CARLA and return the actor (or None).
    Note: read-only; do not call apply_control on this actor.
    Retries handle intermittent UnicodeDecodeError / RuntimeError that occur
    when the CARLA binary RPC protocol is used over a network tunnel (e.g. ngrok).
    """
    if not HAS_CARLA:
        print("[WARN] CARLA python API not available in this process.")
        return None, None, None
    for attempt in range(1, retries + 1):
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            world = client.get_world()
            actor = world.get_actor(actor_id)
            if actor is None:
                print(f"[WARN] No actor with id {actor_id} in CARLA world.")
                return None, None, None
            if not actor.type_id.startswith("vehicle"):
                print(f"[WARN] Actor {actor_id} is not a vehicle (type: {actor.type_id})")
            else:
                print(f"[INFO] Connected to CARLA vehicle actor id={actor_id} type={actor.type_id}")
            # The client and world are returned, NOT dropped on the floor.
            # Previously both were locals and only the actor escaped, so the
            # objects owning the RPC connection lost their last Python
            # reference the moment this function returned and were left to the
            # garbage collector, while the actor kept being used for the whole
            # session. CARLA's documented usage is to keep the client alive for
            # as long as anything obtained from it is in use; the caller now
            # holds all three for the process lifetime.
            return actor, client, world
        except (NotImplementedError, RuntimeError, UnicodeDecodeError) as e:
            print(f"[WARN] CARLA connect attempt {attempt}/{retries} failed ({type(e).__name__}): {e}")
            if attempt < retries:
                time.sleep(1.0)
    print(f"[WARN] Could not connect to CARLA after {retries} attempts. Running without vehicle actor.")
    return None, None, None



def main():
    args = _parse_args(sys.argv[1:])
    participantid = args.participantid
    environment = args.environment
    secondary_task = args.secondary_task
    functionname = args.functionname
    emotion = args.emotion
    modeltype = args.modeltype.lower()  # fcd | state | combined | collection
    state_model = args.state_model.lower()
    w_fcd = args.w_fcd
    # Everything the operator supplied HERE, before any fallback. PV_SESSION_ID
    # counts as supplied: on this machine only start_experiment.py sets it, and
    # it is just as authoritative (and just as wrong, if it disagrees with the
    # other machine) as the flag. The generated id, by contrast, must not be
    # invented until the bridge has had its say, or the adoption below could
    # never happen.
    session_id = args.session_id or os.getenv("PV_SESSION_ID") or None
    window_sz = args.window  # frame-count cap on the model input (safety bound)
    # Time span (seconds) of the window fed to the xLSTM. Unset = inherit the
    # window the checkpoint was trained with (falling back to 20 s for legacy
    # checkpoints). Explicit value overrides; 0 disables the time cap, leaving
    # the rate-dependent frame-count cap only (400 frames ≈ 100 s at ~4 Hz).
    window_seconds = args.window_seconds
    camera_source = args.camera_source
    camera_url = args.camera_url
    webcam = args.webcam
    vehicle_id_arg = args.vehicle_id  # optional: skip file-based discovery when set
    host = args.host
    port = args.port
    carla_timeout = args.carla_timeout
    vehicle_state_url = args.vehicle_state_url  # e.g. http://0.tcp.ngrok.io:PORT
    vehicle_state_file = args.vehicle_state_file  # local file bridge (preferred)

    # --- Session identity, from the remote bridge when there is one ----------
    # Only the HTTP bridge: it is the one that spans two machines, and so the
    # only one where the ids can drift apart. The file bridge runs beside a
    # launcher that already sets both.
    status_url = args.status_url
    if vehicle_state_url and args.bridge_session_timeout > 0:
        bridge = fetch_bridge_session(vehicle_state_url,
                                      timeout=args.bridge_session_timeout)
        session_id, participantid = resolve_session_identity(
            session_id, participantid, bridge)
        # The reverse channel is an ADDRESS, not an identity: two spellings of
        # the same endpoint are not a session mismatch, so this adopts rather
        # than reconciling. An explicit --status-url still wins, which is what
        # makes a tunnel in front of the status server possible.
        if not status_url and (bridge or {}).get("status_url"):
            status_url = bridge["status_url"]
            print(f"[status] adopting status_url={status_url!r} from the bridge.")
        # Probed here rather than trusted: an address that only fails at
        # shutdown fails at the one moment nothing can be done about it.
        if args.status_url:
            # Hand-typed, so it is checked but never replaced — an operator
            # naming a tunnel outranks anything derivable from the bridge host.
            if not _status_channel_alive(status_url):
                print("[status] the status_url given on the command line did "
                      "not answer /health. Keeping it, since it was given "
                      "explicitly, but the signals will not arrive unless it "
                      "becomes reachable.")
        elif status_url:
            status_url = resolve_status_url(status_url, vehicle_state_url)

    # Last resort, after the bridge has had its chance: a local run, or a bridge
    # that published nothing. A generated id keeps this session's rows
    # self-consistent even though nothing else shares it.
    if not session_id:
        session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        if vehicle_state_url:
            print(f"[WARN] No session id from the command line or the bridge; "
                  f"generated {session_id}. This machine's raw_data.jsonl will "
                  f"NOT share a session id with the LoA labels recorded on the "
                  f"CARLA machine — they can still be joined on timestamps, but "
                  f"check this is what you meant before driving.")
    participantid = participantid or ""
    if args.calibration_only and not participantid:
        # The baseline is filed as calibration_<participantid>.json, so with no
        # id DataCollector.save_calibration() finds no path and drops it with a
        # warning — after the full 180 s measurement. Refuse up front instead:
        # the whole run would otherwise produce nothing, and the operator would
        # only learn that at the end of it.
        raise SystemExit(
            "[FATAL] --calibration-only with no participant id: the baseline is "
            "stored per participant, so this run would measure for 180 s and "
            "then have nowhere to save. Pass participantid=..., or point "
            "vehicle_state_url at a bridge that publishes one at /session.")
    if vehicle_state_url and not participantid:
        print("[WARN] No participant id from the command line or the bridge: "
              "this session's rows will be recorded with an empty participantid.")

    logger = Logger(raw_data_file="data/raw_data.jsonl", processed_data_file="data/decisions.csv")
    xlstm_log = args.log_path  # e.g. "state_data.log"; empty/unset = disabled

    strategy = None
    fcd_engine = None
    state_engine = None

    # --- Data collection: no inference, so no model is loaded at all ---
    # DataCollector only starts its decision worker when it has an engine, so
    # leaving `strategy` None is what actually disables inference; skipping the
    # load also keeps the XGBoost/xLSTM weights and their threads out of a run
    # that would never use them.
    if args.data_collection:
        print("[main] --data-collection: no decision engine, recording data only.")

    # --- FCD only ---
    elif modeltype == "fcd":
        try:
            fcd_engine = XGBoostLoAStrategy(
                model_path="trained_models/fcd_levels.pkl",
                default_function=functionname,
                conservative=True,
            )
            strategy = fcd_engine
            print("[main] FCD model loaded successfully from trained_models/fcd_levels.pkl")
        except Exception as e:
            print("[main] FCD load error:", e)
            strategy = LoAZeroFallback("FCD model load error → LoA0")

    elif modeltype == "collection":
        try:
            fcd_engine = XGBoostLoAStrategy(
                model_path="trained_models/fcd_levels.pkl",
                default_function=functionname,
                conservative=True,
            )
            strategy = fcd_engine
            print("[main] FCD model loaded successfully for collection")
        except Exception as e:
            print("[main] FCD load error:", e)
            strategy = LoAZeroFallback("FCD model load error → LoA0")
    # STATE only
    elif modeltype == "state":
        try:
            if state_model == "xlstm":
                state_engine = StateXLSTMLoAStrategy(
                    model_path="trained_models/state_xlstm.pt",
                    default_function=functionname,
                    window=window_sz,
                    fcd_fallback=None,
                    log_path=xlstm_log or None,
                    window_seconds=window_seconds,
                )
                print(f"[main] xLSTM model loaded successfully from trained_models/state_xlstm.pt (window={state_engine.window_seconds}s)")
            else:
                state_engine = StateLevelsLoAStrategy(
                    model_path="trained_models/state_levels.pkl",
                    default_function=functionname,
                    conservative=True,
                    prob_threshold=0.0,
                    fcd_fallback=None,
                )
                print("[main] STATE (classic) model loaded successfully from trained_models/state_levels.pkl")
            strategy = state_engine
        except Exception as e:
            print("[main] STATE load error:", e)
            strategy = LoAZeroFallback("STATE model load error → LoA0")

    # COMBINED (fusion of FCD + State)
    else:
        # FCD
        try:
            fcd_engine = XGBoostLoAStrategy(
                model_path="trained_models/fcd_levels.pkl",
                default_function=functionname,
                conservative=True,
            )
            print("[main] Combined-FCD part loaded successfully.")
        except Exception as e:
            print("[main] FCD load error:", e)
            fcd_engine = LoAZeroFallback("FCD model load error → LoA0")
        # STATE
        try:
            if state_model == "xlstm":
                state_engine = StateXLSTMLoAStrategy(
                    model_path="trained_models/state_xlstm.pt",
                    default_function=functionname,
                    window=window_sz,
                    fcd_fallback=None,
                    log_path=xlstm_log or None,
                    window_seconds=window_seconds,
                )
                print(f"[main] Combined-STATE (xLSTM) part loaded successfully (window={state_engine.window_seconds}s).")
            else:
                state_engine = StateLevelsLoAStrategy(
                    model_path="trained_models/state_levels.pkl",
                    default_function=functionname,
                    conservative=True,
                    prob_threshold=0.0,
                    fcd_fallback=None,
                )
                print("[main] Combined-STATE (classic) part loaded successfully.")
        except Exception as e:
            print("[main] STATE load error:", e)
            state_engine = LoAZeroFallback("STATE model load error → LoA0")

        try:
            strategy = CombinedFusionStrategy(
                fcd_strategy=fcd_engine,
                state_strategy=state_engine,
                w_fcd=w_fcd,
                conservative=True,
            )
            print("[main] CombinedFusionStrategy initialized successfully.")
        except Exception as e:
            print("[main] Combined init error:", e)
            strategy = fcd_engine if fcd_engine is not None else LoAZeroFallback("Combined init error → LoA0")

    actuator = ProVoiceActuator()
    static_context = {
        "session_id": session_id,
        "participantid": participantid,
        "environment": environment,
        "secondary_task": secondary_task,
        "functionname": functionname,
        "emotion": emotion,
        "modeltype": modeltype,
        "state_model": state_model,
        "w_fcd": w_fcd,
    }

    print(f"[main] session_id={session_id}")
    print(f"[main] Static context: {static_context}")

    # ---------------------------------------------------------------------
    # Add: Read vehicle_id and attempt to connect to CARLA to get the vehicle actor (optional)
    # ---------------------------------------------------------------------
    vehicle_actor = None
    # Held for the process lifetime, deliberately: these own the RPC connection
    # the actor is read through, and letting them be collected would leave the
    # actor pointing at a torn-down client. Not otherwise used, hence the names.
    _carla_client = None
    _carla_world = None
    if vehicle_state_file:
        # File bridge: a separate process owns the CARLA client and publishes
        # state to a file. Skipping the connection here is the entire point —
        # this process must never construct a carla.Client, because polling
        # CARLA from inside ProVoice corrupts its heap.
        print(f"[INFO] vehicle_state_file set ({vehicle_state_file}) — "
              f"skipping direct CARLA connection.")
    elif vehicle_state_url:
        # Bridge URL provided — no direct CARLA connection needed; the bridge
        # reads from CARLA locally on the remote and serves speed/location over HTTP.
        print(f"[INFO] vehicle_state_url set — skipping direct CARLA connection.")
    else:
        if vehicle_id_arg is not None:
            try:
                vehicle_id = int(vehicle_id_arg)
                print(f"[INFO] Using vehicle_id={vehicle_id} from command-line argument.")
            except ValueError:
                print(f"[WARN] Invalid vehicle_id argument {vehicle_id_arg!r}; ignoring.")
                vehicle_id = None
        else:
            vehicle_id = read_vehicle_id(wait_seconds=10.0)

        if vehicle_id is not None and HAS_CARLA:
            vehicle_actor, _carla_client, _carla_world = get_carla_vehicle_by_id(
                vehicle_id, host=host, port=port, timeout=carla_timeout)
            if vehicle_actor is None:
                print("[WARN] Could not obtain vehicle actor from CARLA. DataCollector will run without carla_vehicle.")
            else:
                print(f"[INFO] Connected to CARLA vehicle actor id={vehicle_id} type={vehicle_actor.type_id}")
        else:
            if vehicle_id is None:
                print("[WARN] No vehicle_id available; DataCollector will run without carla_vehicle.")
            elif not HAS_CARLA:
                print("[WARN] CARLA API not available in this process; DataCollector will run without carla_vehicle.")

    # Determine cam_index for DataCollector
    if webcam and camera_source != "front":
        # Both given. Honour the explicit source -- it is the more specific
        # instruction -- but say so out loud: silently ignoring a flag is how a
        # session ends up recording the wrong camera without anyone noticing.
        print(f"[camera] --webcam ignored: camera_source={camera_source} was given "
              f"explicitly and takes precedence.")
        webcam = False

    if webcam:
        cam_index = _select_webcam()
    elif camera_source == "udp":
        cam_index = camera_url
    elif camera_source.isdigit():
        cam_index = int(camera_source)
    elif camera_source == "local":
        cam_index = 0
    else:
        # Default case, e.g. "front"
        cam_index = 0

    # Create the data collector, passing in carla_vehicle (if available)
    data_collector = DataCollector(
        visual=True,
        physiological=True,
        context=True,
        sample_rate=20.0,
        logger=logger,
        decision_engine=strategy,
        actuator=actuator,
        function_name=functionname,
        cam_index=cam_index,
        static_context=static_context,
        carla_vehicle=vehicle_actor,  # might be None
        vehicle_state_url=vehicle_state_url,
        vehicle_state_path=vehicle_state_file,
        state_poll_hz=args.state_poll_hz,
        window_size=window_sz,
        decision_hz=args.decision_hz,
        calibration_only=args.calibration_only,
        data_collection=args.data_collection,
    )

    dashboard.data_collector = data_collector
    dashboard.actuator = actuator

    config = uvicorn.Config(dashboard.app, host="127.0.0.1", port=8001, reload=False)
    server = uvicorn.Server(config)

    if args.calibration_only:
        # The collector runs on its own thread and cannot stop the server; give
        # it the hook. Installed BEFORE start() so a calibration that finishes
        # unusually early (e.g. no camera → default baselines immediately) still
        # finds the callback in place.
        def _calibration_finished():
            print("[main] calibration stored — exiting (--calibration-only).")
            data_collector.stop()
            server.should_exit = True
        data_collector.on_calibration_complete = _calibration_finished

    # --- Reverse channel to the CARLA machine --------------------------------
    # Locally, Drive watches raw_data.jsonl for this session's first line and
    # starts its LoA windows on it. Split across two machines that file is on
    # THIS side, so the same fact has to be sent rather than observed.
    if status_url:
        print(f"[status] reverse bridge: {status_url} "
              f"(collection_started at the first logged frame, provoice_ended "
              f"on exit)")

        def _first_frame():
            # Off the collection loop: this posts over the network, and that
            # loop is the one holding the perception cadence. A blocked tick
            # here would show up as a gap in the very data being announced.
            threading.Thread(
                target=post_status_event,
                args=(status_url, "collection_started", session_id, participantid),
                name="status-started", daemon=True).start()
        data_collector.on_first_frame_logged = _first_frame
    elif vehicle_state_url:
        # Either none was published, or the one published could not be reached
        # (resolve_status_url has already said which, and how to fix it).
        print("[status] running WITHOUT a reverse bridge: Drive on the other "
              "machine will fall back to its popup-wait timeout for the start, "
              "and will not be told when this run ends — stop the drive there "
              "by hand.")

    data_collector.start()

    def handle_exit(_, __):
        print("[main] shutdown signal received — stopping cleanly.")
        if data_collector:
            data_collector.stop()
        server.should_exit = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    # SIGBREAK is what Windows delivers for CTRL_BREAK_EVENT, and that is how
    # start_experiment.py now asks this process to stop. Without this handler
    # the default action terminates the process outright, which is exactly the
    # abrupt death we are trying to avoid: it leaves the CARLA RPC connection
    # severed rather than closed, and CARLA cannot then distinguish a normal
    # end-of-run from a ProVoice crash.
    #
    # Registering it matters more than it looks. Until now the launcher stopped
    # this process with TerminateProcess (Popen.terminate() is an alias for
    # kill() on Windows), so NEITHER handler above had ever run in a real
    # session -- every run ended abruptly, whether or not anything went wrong.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_exit)

    try:
        server.run()
    finally:
        data_collector.stop()
        # Sent from the finally, so it covers EVERY way this process ends that
        # leaves Python running: a clean exit, Ctrl-C, the launcher's
        # CTRL_BREAK, an exception out of server.run(), and the calibration
        # callback's own shutdown. It cannot cover a hard native fault
        # (0xC0000005 and friends) -- nothing in this process can -- so the
        # other machine also keeps its manual stop.
        if status_url:
            post_status_event(
                status_url, "provoice_ended", session_id, participantid,
                reason=("calibration complete" if args.calibration_only
                        else "data collection ended"))
        logger.close()
        for _s in (state_engine, fcd_engine, strategy):
            if hasattr(_s, "close"):
                try:
                    _s.close()
                except Exception:
                    pass
        print("App exiting cleanly")

if __name__ == "__main__":
    main()
