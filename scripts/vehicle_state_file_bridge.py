#!/usr/bin/env python3
"""
Local vehicle-state bridge that publishes through a FILE, not a socket.

    python scripts/vehicle_state_file_bridge.py --vehicle-id 42 --out vehicle_state.json

Why a file and not the HTTP server in scripts/vehicle_state_server.py:

  * ProVoice must not hold a CARLA client of its own. Polling CARLA directly
    corrupts its heap -- crashes land in unrelated threads (YOLO convolutions,
    xLSTM encode_frame, a CPython dict lookup) -- while runs with the CARLA
    calls removed stay clean. So the client moves into this separate process,
    where a fault costs a restart instead of the session.

  * But the HTTP version pays for that isolation with ~20 TCP connections per
    SECOND (urllib sends Connection: close, BaseHTTPRequestHandler is HTTP/1.0),
    i.e. ~36,000 connect/teardown cycles in a 30-minute session, each churning
    nonpaged pool through afd.sys -> tcpip.sys -> the NDIS filter stack. On the
    lab machine that stack also carries an NDIS lightweight filter and Hyper-V
    virtual networking with VBS/HVCI enabled. Two machine-level bugchecks
    (0x1E, then 0xD1 with the EXECUTE flag -- a corrupted function pointer)
    occurred during the only two sessions that used the HTTP bridge.

    A user-mode process cannot corrupt kernel memory itself, so the bugchecks
    are a driver defect. But that connection churn is the one kernel path the
    HTTP bridge introduced and nothing else in the experiment touches, which
    makes it the prime suspect for provoking it.

  * This version writes a small JSON file instead. No sockets, no NDIS, no
    virtual switch, no TIME_WAIT. It is not a guaranteed fix for a bugcheck we
    have not yet attributed to a specific driver -- it removes the suspected
    path rather than proving it was to blame.

Concurrency: writer and reader share one file guarded by an advisory lock
(src/vehicle_state_io.py), held across the whole write and the whole read, so
no one can read a half-written record and two writers cannot interleave. A
temp-file + os.replace publish was rejected because on Windows the replace
fails with a sharing violation whenever the reader has the file open, which at
20 Hz on both sides would be most of the time.

Every record carries a wall-clock "ts" so the reader can tell fresh state from
a bridge that has stopped updating.
"""

import argparse
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# src/ on the path, NOT the ProVoice package: importing ProVoice would pull in
# torch, cv2 and mediapipe, which this process must never load.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vehicle_state_io import VehicleStateChannel  # noqa: E402

try:
    import carla
except ImportError:
    print("[filebridge] ERROR: CARLA Python API not found. Run from the project venv.")
    sys.exit(1)


def read_vehicle_id(path: str, wait: float = 60.0) -> "int | None":
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            raw = open(path).read().strip()
            if raw:
                return int(raw)
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.5)
    return None


def install_graceful_stop():
    """Exit cleanly on the launcher's CTRL_BREAK so the CARLA client closes.

    Popen.terminate() is an alias for kill() on Windows and runs no cleanup, so
    start_experiment.py signals instead. A cleanly closed client matters here:
    an abruptly severed CARLA connection is what the server struggles to
    survive when the next client connects.
    """
    def _stop(_sig, _frame):
        raise KeyboardInterrupt

    for name in ("SIGBREAK", "SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):
            pass


def collect(actor, world, carla_map) -> dict:
    """One snapshot, with the same field names the HTTP bridge serves so
    DataCollector._poll_vehicle_state consumes either without changes."""
    vel = actor.get_velocity()
    speed_kmh = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5 * 3.6
    acc = actor.get_acceleration()
    acceleration = (acc.x ** 2 + acc.y ** 2 + acc.z ** 2) ** 0.5
    control = actor.get_control()
    speed_limit = actor.get_speed_limit()
    weather = world.get_weather()
    waypoint = carla_map.get_waypoint(actor.get_location())

    try:
        tl_state = str(actor.get_traffic_light_state()).split('.')[-1]
    except Exception:
        tl_state = None
    try:
        ls = int(actor.get_light_state())
        headlight = bool(ls & (2 | 4))
        fog_light = bool(ls & 128)
        left_indicator = bool(ls & 32)
        right_indicator = bool(ls & 16)
    except Exception:
        headlight = fog_light = left_indicator = right_indicator = None

    return {
        "ts":                  round(time.time(), 3),
        "speed_kmh":           round(speed_kmh, 2),
        "brake":               round(float(control.brake), 3),
        "steer":               round(float(control.steer), 3),
        "throttle":            round(float(control.throttle), 3),
        "gear":                int(control.gear),
        "hand_brake":          bool(control.hand_brake),
        "reverse":             bool(control.reverse),
        "acceleration":        round(acceleration, 3),
        "speed_limit_kmh":     round(float(speed_limit), 1),
        "precipitation":       round(weather.precipitation / 100.0, 3),
        "fog_density":         round(weather.fog_density / 100.0, 3),
        "is_night":            bool(weather.sun_altitude_angle < 0),
        "is_junction":         bool(waypoint.is_junction),
        "traffic_light_state": tl_state,
        "headlight":           headlight,
        "fog_light":           fog_light,
        "left_indicator":      left_indicator,
        "right_indicator":     right_indicator,
    }


def zeroed(sample: dict) -> dict:
    """The same record shape with every vehicle field inert.

    Values mirror DataCollector's initial _cached_* defaults exactly, so a
    --zeros run is indistinguishable from --provoice-no-carla in what ProVoice
    ends up holding -- while the bridge, the file, the lock and the successful
    20 Hz polls all still happen. "ts" stays real so staleness detection keeps
    working; a frozen timestamp would make ProVoice flag the feed as dead and
    change a second variable.
    """
    out = dict(sample)
    # Mirrors DataCollector.__init__ field by field, INCLUDING the types --
    # several of these are int rather than float or bool there:
    #     _cached_speed: int = 0        _cached_throttle: float = 0.0
    #     _cached_steer: int = 0        _cached_gear: int = 0
    #     _cached_brake: int = 0        _cached_acceleration: float = 0.0
    #     _cached_precipitation: int=0  _cached_fog_density: float = 0.0
    #     _cached_speed_limit: int = 0  _cached_traffic_light_state = None
    #     _cached_night: int = 0        _cached_hand_brake/reverse = False
    #     _cached_junction: int = 0     _cached_*light/indicator = False
    out["speed_kmh"] = 0            # int
    out["steer"] = 0                # int
    out["brake"] = 0                # int
    out["precipitation"] = 0        # int
    out["speed_limit_kmh"] = 0      # int
    out["is_night"] = 0             # int, not False
    out["is_junction"] = 0          # int, not False
    out["gear"] = 0                 # int
    out["throttle"] = 0.0           # float
    out["acceleration"] = 0.0       # float
    out["fog_density"] = 0.0        # float
    out["hand_brake"] = False
    out["reverse"] = False
    out["headlight"] = False
    out["fog_light"] = False
    out["left_indicator"] = False
    out["right_indicator"] = False
    out["traffic_light_state"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="vehicle_state.json",
                        help="File to publish state into. ProVoice reads this path.")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Publish rate. Should match the collection loop; "
                             "every field is held constant between writes, so "
                             "this is the true sampling rate of steer/brake.")
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout", type=float, default=10.0)
    parser.add_argument("--vehicle-id-path", default="vehicle_id.txt")
    parser.add_argument("--vehicle-id", type=int, default=None,
                        help="Skip vehicle_id.txt discovery. The launcher passes "
                             "this, having already read and validated the id, "
                             "which also stops a restart racing a rewrite.")
    parser.add_argument("--zeros", action="store_true",
                        help="DIAGNOSTIC: still read CARLA at the normal rate, "
                             "but publish an all-zero record. This isolates ONE "
                             "variable -- the numbers ProVoice receives -- while "
                             "keeping everything else identical (this process, "
                             "its CARLA load, the file, the lock, the 20 Hz "
                             "publish, ProVoice's successful polls). "
                             "Zeros match DataCollector's own defaults, so "
                             "ProVoice sees exactly what --provoice-no-carla "
                             "gives it, but by a completely different route.")
    parser.add_argument("--max-consecutive-errors", type=int, default=50,
                        help="Exit (for the supervisor to restart us with a "
                             "fresh CARLA client) after this many failed reads "
                             "in a row. A single bad read is normal; a long run "
                             "of them means the client or the actor is gone.")
    args = parser.parse_args()

    install_graceful_stop()

    if args.vehicle_id is not None:
        vehicle_id = args.vehicle_id
        print(f"[filebridge] Using vehicle id {vehicle_id} from the command line.")
    else:
        print(f"[filebridge] Waiting for {args.vehicle_id_path} ...")
        vehicle_id = read_vehicle_id(args.vehicle_id_path)
        if vehicle_id is None:
            print("[filebridge] vehicle_id.txt not found after 60 s. Exiting.")
            sys.exit(1)

    print(f"[filebridge] Connecting to CARLA at {args.carla_host}:{args.carla_port} ...")
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(args.carla_timeout)
    world = client.get_world()
    actor = world.get_actor(vehicle_id)
    if actor is None:
        print(f"[filebridge] Actor id={vehicle_id} not found in CARLA world.")
        sys.exit(1)
    # Cached: get_map() re-parses the whole OpenDRIVE description and must not
    # be called per frame.
    carla_map = world.get_map()

    channel = VehicleStateChannel(args.out, create=True)
    interval = 1.0 / max(1e-3, args.hz)
    print(f"[filebridge] Tracking actor id={vehicle_id} type={actor.type_id}")
    print(f"[filebridge] Publishing to {channel.path} at {args.hz:.1f} Hz"
          + ("  *** --zeros: CARLA IS READ BUT ZEROS ARE PUBLISHED, "
             "THIS RUN IS NOT USABLE DATA ***" if args.zeros else ""), flush=True)

    errors = 0
    written = 0
    next_t = time.monotonic()
    try:
        while True:
            try:
                # CARLA is read either way, so --zeros changes only what
                # ProVoice receives, not what this process does.
                sample = collect(actor, world, carla_map)
                if args.zeros:
                    sample = zeroed(sample)
                # The lock is held inside publish() across write+truncate, so
                # the reader sees either the previous record or this one.
                channel.publish(sample)
                written += 1
                if errors:
                    print(f"[filebridge] recovered after {errors} failed read(s)",
                          flush=True)
                    errors = 0
            except Exception as e:  # noqa: BLE001
                errors += 1
                if errors == 1 or errors % 20 == 0:
                    print(f"[filebridge] read/write FAILED (#{errors}): "
                          f"{type(e).__name__}: {e}", flush=True)
                if errors >= args.max_consecutive_errors:
                    print(f"[filebridge] {errors} consecutive failures; exiting so "
                          f"the supervisor can restart with a fresh client.",
                          flush=True)
                    sys.exit(1)

            next_t += interval
            now = time.monotonic()
            if next_t < now:          # fell behind: skip missed ticks
                next_t = now
            time.sleep(max(0.0, next_t - now))
    except KeyboardInterrupt:
        print(f"[filebridge] stopping after {written} records.")
        channel.close()


if __name__ == "__main__":
    main()
