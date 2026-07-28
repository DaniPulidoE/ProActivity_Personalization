#!/usr/bin/env python3
"""
Lightweight HTTP bridge: owns the CARLA client in its OWN process and serves
vehicle state as JSON.

PRIMARY USE (local, same machine) -- process isolation, not remoting:

    start_experiment.py --vehicle-bridge

    ProVoice then runs with carla_vehicle=None and never loads a CARLA client
    of its own. Measured 2026-07-28: ProVoice reliably corrupts its own heap
    when it polls CARLA directly (crashes landing in unrelated threads -- YOLO
    convolutions, xLSTM encode_frame, a CPython dict lookup), and runs clean
    across repeated sessions with the CARLA calls removed. Moving those calls
    into this process contains the fault: if libcarla misbehaves here, only
    this bridge dies, the launcher restarts it, and the participant session
    plus all logging continue.

    Binds 127.0.0.1 by default, so nothing is exposed off the machine.

SECONDARY USE (the original one) -- CARLA on a remote host: run this next to
CARLA, expose it with `--bind 0.0.0.0` plus a tunnel, and pass the public URL
to ProVoice as vehicle_state_url=...

    python scripts/vehicle_state_server.py --port 8080 --bind 0.0.0.0
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import carla
except ImportError:
    print("[bridge] ERROR: CARLA Python API not found. Run from the project venv.")
    sys.exit(1)


def read_vehicle_id(path: str, wait: float = 60.0) -> int | None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="127.0.0.1",
                        help="Interface to listen on. Default is loopback only, "
                             "which is what the local same-machine setup wants. "
                             "Use 0.0.0.0 only when CARLA is remote and the "
                             "bridge must be reachable from another host.")
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--vehicle-id-path", default="vehicle_id.txt")
    parser.add_argument("--vehicle-id", type=int, default=None,
                        help="Skip vehicle_id.txt discovery. start_experiment.py "
                             "passes this because it has already read and "
                             "validated the id, which also avoids re-reading a "
                             "file that a restart could catch mid-write.")
    args = parser.parse_args()

    if args.vehicle_id is not None:
        vehicle_id = args.vehicle_id
        print(f"[bridge] Using vehicle id {vehicle_id} from the command line.")
    else:
        print(f"[bridge] Waiting for {args.vehicle_id_path} ...")
        vehicle_id = read_vehicle_id(args.vehicle_id_path)
        if vehicle_id is None:
            print("[bridge] vehicle_id.txt not found after 60 s. Exiting.")
            sys.exit(1)

    print(f"[bridge] Connecting to CARLA at {args.carla_host}:{args.carla_port} ...")
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(10.0)
    world = client.get_world()
    actor = world.get_actor(vehicle_id)
    if actor is None:
        print(f"[bridge] Actor id={vehicle_id} not found in CARLA world.")
        sys.exit(1)
    carla_map = world.get_map()  # cache — get_map() is expensive to call at 20 Hz
    print(f"[bridge] Tracking actor id={vehicle_id} type={actor.type_id}")
    print(f"[bridge] Serving on http://{args.bind}:{args.port}/", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
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
                    headlight      = bool(ls & (2 | 4))
                    fog_light      = bool(ls & 128)
                    left_indicator = bool(ls & 32)
                    right_indicator = bool(ls & 16)
                except Exception:
                    headlight = fog_light = left_indicator = right_indicator = None
                payload = json.dumps({
                    "speed_kmh":          round(speed_kmh, 2),
                    "brake":              round(float(control.brake), 3),
                    "steer":              round(float(control.steer), 3),
                    "throttle":           round(float(control.throttle), 3),
                    "gear":               int(control.gear),
                    "hand_brake":         bool(control.hand_brake),
                    "reverse":            bool(control.reverse),
                    "acceleration":       round(acceleration, 3),
                    "speed_limit_kmh":    round(float(speed_limit), 1),
                    "precipitation":      round(weather.precipitation / 100.0, 3),
                    "fog_density":        round(weather.fog_density / 100.0, 3),
                    "is_night":           bool(weather.sun_altitude_angle < 0),
                    "is_junction":        bool(waypoint.is_junction),
                    "traffic_light_state": tl_state,
                    "headlight":          headlight,
                    "fog_light":          fog_light,
                    "left_indicator":     left_indicator,
                    "right_indicator":    right_indicator,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())

        def log_message(self, *args):
            pass  # suppress per-request noise

    # allow_reuse_address so a supervised restart can rebind immediately instead
    # of failing on the previous socket's TIME_WAIT.
    HTTPServer.allow_reuse_address = True
    HTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
