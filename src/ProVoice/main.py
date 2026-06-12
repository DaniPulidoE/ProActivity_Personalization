from __future__ import annotations

import signal
import sys
import time
import os
import uuid

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


def _parse_kv_argv(argv):
    out = {}
    for tok in argv[1:]:
        tok = tok.strip().strip(",")
        if not tok or "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out

def read_vehicle_id(path: str | None = None, wait_seconds: float = 10.0) -> int | None:
    """
    Read the vehicle ID from src/drive/vehicle_id.txt relative to the parent directory of this file.
    """
    # Directory of the current file, e.g., .../src/ProVoice/
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Parent directory, i.e., src/
    src_dir = os.path.dirname(current_file_dir)

    # Default file location: src/drive/vehicle_id.txt
    default_path = os.path.join(src_dir, "drive", "vehicle_id.txt")

    src_dir = os.path.dirname(src_dir)
    default_path = os.path.join(src_dir, "vehicle_id.txt")

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
        except NotImplementedError as e:
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
    args = _parse_kv_argv(sys.argv)
    participantid = args.get("participantid", "")
    environment = args.get("environment", "")
    secondary_task = args.get("secondary_task", "")
    functionname = args.get("functionname", "Adjust seat positioning")
    emotion = args.get("emotion", args.get("affect", ""))
    modeltype = args.get("modeltype", "combined").lower()  # fcd | state | combined
    state_model = args.get("state_model", args.get("statemodel", "classic")).lower()
    w_fcd = float(args.get("w_fcd", "0.5"))
    session_id = args.get("session_id") or os.getenv("PV_SESSION_ID") or f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    window_sz = int(args.get("window", "256"))
    camera_source = args.get("camera_source", "front")
    camera_url = args.get("camera_url", "udp://127.0.0.1:8554")
    vehicle_id_arg = args.get("vehicle_id")  # optional: skip file-based discovery when set
    host = args.get("host", "localhost")
    port = int(args.get("port", "2000"))
    carla_timeout = float(args.get("carla_timeout", "10.0"))
    vehicle_state_url = args.get("vehicle_state_url")  # e.g. http://0.tcp.ngrok.io:PORT

    logger = Logger(raw_data_file="data/raw_data.jsonl", processed_data_file="data/decisions.csv")
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
        except NotImplementedError as e:
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
        except NotImplementedError as e:
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
                )
                print("[main] xLSTM model loaded successfully from trained_models/state_xlstm.pt")
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
        except NotImplementedError as e:
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
        except NotImplementedError as e:
            print("[main] FCD load error:", e)
            fcd_engine = LoAZeroFallback("FCD model load error → LoA0")
        # STATE
        try:
            if state_model == "xlstm":
                # TODO
                state_engine = StateXLSTMLoAStrategy(
                    model_path="trained_models/state_xlstm.pt",
                    default_function=functionname,
                    window=window_sz,
                    fcd_fallback=None,
                )
                print("[main] Combined-STATE (xLSTM) part loaded successfully.")
            else:
                state_engine = StateLevelsLoAStrategy(
                    model_path="trained_models/state_levels.pkl",
                    default_function=functionname,
                    conservative=True,
                    prob_threshold=0.0,
                    fcd_fallback=None,
                )
                print("[main] Combined-STATE (classic) part loaded successfully.")
        except NotImplementedError as e:
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
        except NotImplementedError as e:
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
    )

    dashboard.data_collector = data_collector
    dashboard.actuator = actuator

    data_collector.start()

    config = uvicorn.Config(dashboard.app, host="0.0.0.0", port=8001, reload=False)
    server = uvicorn.Server(config)

    def handle_exit(_, __):
        print("KeyboardInterrupt received")
        if data_collector:
            data_collector.stop()
        server.should_exit = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    server.run()
    print("App exiting cleanly")

if __name__ == "__main__":
    main()
