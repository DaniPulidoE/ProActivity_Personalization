#!/usr/bin/env python3
"""
Lightweight HTTP bridge: reads vehicle state from a local CARLA instance and
serves it as JSON so ProVoice can fetch it through an ngrok tunnel.

Run this on the REMOTE machine (alongside CARLA and drive_improved.py):

    python scripts/vehicle_state_server.py --port 8080

Then open a second ngrok tunnel:

    ngrok http 8080

And pass the URL to ProVoice on your local machine:

    python src/ProVoice/main.py vehicle_state_url=https://<ngrok-id>.ngrok-free.app ...

NOTE: for a LOCAL, same-machine setup do NOT use this script -- use
scripts/vehicle_state_file_bridge.py instead. It publishes the same fields
through a file rather than a socket, which keeps ~20 TCP connections/second
out of the kernel's network stack.
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
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--vehicle-id-path", default="vehicle_id.txt")
    args = parser.parse_args()

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
    print(f"[bridge] Serving on http://0.0.0.0:{args.port}/")

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

    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
