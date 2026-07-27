from __future__ import annotations

import signal
import sys
import time
import os
import uuid
import argparse as ap

# ── CPU thread budget ────────────────────────────────────────────────────────
# Must run BEFORE numpy/torch/cv2 are imported: the OpenMP/MKL runtimes size
# their thread pools once, at load time, and ignore these variables afterwards.
#
# Why cap them at all: this process shares the machine with the CARLA server and
# the Drive client, and CARLA runs in SYNCHRONOUS mode (drive_improved.py sets
# fixed_delta_seconds=0.05 and drives the world with sim_world.tick()), so the
# simulation only advances as fast as Drive's loop does. Anything that stalls
# Drive is a visible sim hitch.
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
    p.add_argument("--vehicle-id", dest="vehicle_id", default=None,
                   help="Skip vehicle_id.txt discovery when set.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--carla-timeout", dest="carla_timeout", type=float, default=10.0)
    p.add_argument("--vehicle-state-url", dest="vehicle_state_url", default=None)
    p.add_argument("--log-path", dest="log_path", default=None,
                   help="JSONL log of features fed to the xLSTM; unset disables.")
    return p


def _parse_args(argv):
    args, unknown = _build_parser().parse_known_args(_normalize_argv(argv))
    if unknown:
        # Surface stray tokens instead of silently dropping them (the old
        # key=value parser ignored anything without an '=').
        print(f"[main] ignoring unrecognized argument(s): {unknown}")
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


def get_carla_vehicle_by_id(actor_id: int, host: str = "127.0.0.1", port: int = 2000, timeout: float = 2.0, retries: int = 5):
    """
    Connect to CARLA and return the actor (or None).
    Note: read-only; do not call apply_control on this actor.
    Retries handle intermittent UnicodeDecodeError / RuntimeError that occur
    when the CARLA binary RPC protocol is used over a network tunnel (e.g. ngrok).
    """
    if not HAS_CARLA:
        print("[WARN] CARLA python API not available in this process.")
        return None
    for attempt in range(1, retries + 1):
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            world = client.get_world()
            actor = world.get_actor(actor_id)
            if actor is None:
                print(f"[WARN] No actor with id {actor_id} in CARLA world.")
                return None
            if not actor.type_id.startswith("vehicle"):
                print(f"[WARN] Actor {actor_id} is not a vehicle (type: {actor.type_id})")
            else:
                print(f"[INFO] Connected to CARLA vehicle actor id={actor_id} type={actor.type_id}")
            return actor
        except (NotImplementedError, RuntimeError, UnicodeDecodeError) as e:
            print(f"[WARN] CARLA connect attempt {attempt}/{retries} failed ({type(e).__name__}): {e}")
            if attempt < retries:
                time.sleep(1.0)
    print(f"[WARN] Could not connect to CARLA after {retries} attempts. Running without vehicle actor.")
    return None



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
    session_id = args.session_id or os.getenv("PV_SESSION_ID") or f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    window_sz = args.window  # frame-count cap on the model input (safety bound)
    # Time span (seconds) of the window fed to the xLSTM. Unset = inherit the
    # window the checkpoint was trained with (falling back to 20 s for legacy
    # checkpoints). Explicit value overrides; 0 disables the time cap, leaving
    # the rate-dependent frame-count cap only (400 frames ≈ 100 s at ~4 Hz).
    window_seconds = args.window_seconds
    camera_source = args.camera_source
    camera_url = args.camera_url
    vehicle_id_arg = args.vehicle_id  # optional: skip file-based discovery when set
    host = args.host
    port = args.port
    carla_timeout = args.carla_timeout
    vehicle_state_url = args.vehicle_state_url  # e.g. http://0.tcp.ngrok.io:PORT

    logger = Logger(raw_data_file="data/raw_data.jsonl", processed_data_file="data/decisions.csv")
    xlstm_log = args.log_path  # e.g. "state_data.log"; empty/unset = disabled

    strategy = None
    fcd_engine = None
    state_engine = None

    # --- FCD only ---
    if modeltype == "fcd":
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
    if vehicle_state_url:
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
            vehicle_actor = get_carla_vehicle_by_id(vehicle_id, host=host, port=port, timeout=carla_timeout)
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
    if camera_source == "udp":
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
        window_size=window_sz,
        decision_hz=args.decision_hz,
    )

    dashboard.data_collector = data_collector
    dashboard.actuator = actuator

    data_collector.start()

    config = uvicorn.Config(dashboard.app, host="127.0.0.1", port=8001, reload=False)
    server = uvicorn.Server(config)

    def handle_exit(_, __):
        print("KeyboardInterrupt received")
        if data_collector:
            data_collector.stop()
        server.should_exit = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        server.run()
    finally:
        data_collector.stop()
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
