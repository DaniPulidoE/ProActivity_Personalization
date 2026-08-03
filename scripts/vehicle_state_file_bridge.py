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


# =========================
# TIME HEADWAY TO THE VEHICLE AHEAD
# =========================
#
# NOT a CARLA variable. There is no get_headway() and no lead-vehicle field in
# the API -- the traffic manager keeps its own notion internally for
# distance_to_leading_vehicle and does not expose it -- so it is computed here
# from geometry. What CARLA does provide, and what makes this affordable at
# 20 Hz, is the two things below.
#
# COST, because this runs inside the sampler loop that the bridge exists to keep
# cheap (see the header: the old design's ~8 RPCs per REQUEST is what motivated
# the cached-sample architecture):
#
#   world.get_snapshot()   ONE RPC returning every actor's transform. The naive
#                          version -- actor.get_transform() per NPC -- is 11
#                          RPCs per sample, 220/s at 20 Hz, on the connection
#                          this file is careful not to load.
#   carla_map.get_waypoint / Waypoint.next
#                          LOCAL. The map is downloaded once and queried
#                          client-side, so walking the lane ahead costs no
#                          network traffic at all.
#   world.get_actors()     an RPC, so the fleet list and the bounding boxes
#                          (which never change) are cached and refreshed on a
#                          timer rather than per sample.
#
# WHY A LANE WALK rather than a straight corridor ahead of the bumper: 122 of
# Town10HD's 149 arcs are under 20 m radius, the tightest 7.4 m. On a bend the
# lead vehicle is not in front of the ego in a straight line -- it is around the
# corner -- so a straight-line test loses it exactly where following distance
# matters most, and can pick up a car in the adjacent lane instead. Walking the
# ego's own lane forward and measuring against that path gets both cases right.
#
# KNOWN LIMITATION, deliberately not solved: at a junction Waypoint.next()
# branches and this takes the first successor, so the "lane ahead" becomes one
# arbitrary exit. A lead vehicle on a different exit is then missed. Junctions
# are where the concept is ill-defined anyway, and is_junction is already
# logged per frame, so analysis can condition on it rather than have this guess.
_HEADWAY_MAX_RANGE_M = 100.0     # beyond this there is no meaningful leader
_HEADWAY_PATH_STEP_M = 4.0       # lane-walk resolution
_HEADWAY_LANE_HALF_W_M = 1.75    # Town10HD driving lanes are 3.50 m, all 168
_HEADWAY_MIN_SPEED_MPS = 0.5     # below this, headway is undefined, not huge
_HEADWAY_MAX_DZ_M = 4.0          # reject a vehicle on a bridge or underpass
_FLEET_REFRESH_S = 2.0

# Vehicle ids and half-lengths, refreshed on a timer. Module level because each
# bridge is a single process with one ego; both bridges import this function.
_fleet_cache: dict = {"t": 0.0, "vehicles": []}


def _fleet(world, ego_id, now):
    """Other vehicles as (id, half_length_m), refreshed every _FLEET_REFRESH_S.

    Cached because world.get_actors() is an RPC and the bounding box behind
    half_length never changes for a spawned actor. The NPC fleet is spawned once
    per session, so the list is nearly static; the refresh exists so a mid-run
    respawn is picked up rather than requiring a restart.
    """
    if now - _fleet_cache["t"] >= _FLEET_REFRESH_S:
        vehicles = []
        for other in world.get_actors().filter("vehicle.*"):
            if other.id == ego_id:
                continue
            try:
                vehicles.append((other.id, float(other.bounding_box.extent.x)))
            except Exception:
                continue
        _fleet_cache["vehicles"] = vehicles
        _fleet_cache["t"] = now
    return _fleet_cache["vehicles"]


def lead_and_headway(actor, world, carla_map, speed_mps):
    """(gap_to_lead_m, time_headway_s) for the vehicle ahead in the ego's lane.

    The gap is BUMPER TO BUMPER -- centre distance minus both half-lengths --
    because that is the quantity a driver and a following-distance controller
    both act on, and because centre-to-centre would make a truck look further
    away than a hatchback at the same real gap.

    Returns (None, None) when there is no leader within _HEADWAY_MAX_RANGE_M.
    Returns (gap, None) when the ego is below _HEADWAY_MIN_SPEED_MPS: at a
    standstill the time to close a gap is not large, it is undefined, and
    logging a huge number instead would put a spike in the data every time the
    participant stops at a light.
    """
    try:
        ego_tf = actor.get_transform()
        ego_loc = ego_tf.location
        ego_half = float(actor.bounding_box.extent.x)

        # The ego's own lane, walked forward. Local computation -- see above.
        path = []
        wp = carla_map.get_waypoint(ego_loc)
        if wp is None:
            return None, None
        travelled = 0.0
        while travelled < _HEADWAY_MAX_RANGE_M:
            nxt = wp.next(_HEADWAY_PATH_STEP_M)
            if not nxt:
                break
            wp = nxt[0]
            travelled += _HEADWAY_PATH_STEP_M
            path.append((travelled, wp.transform.location))
        if not path:
            return None, None

        snapshot = world.get_snapshot()          # ONE RPC for every transform
        now = time.time()

        best_gap = None
        for other_id, other_half in _fleet(world, actor.id, now):
            other_snap = snapshot.find(other_id)
            if other_snap is None:               # destroyed since the last refresh
                continue
            other_loc = other_snap.get_transform().location
            if abs(other_loc.z - ego_loc.z) > _HEADWAY_MAX_DZ_M:
                continue

            # Nearest point on the path, projected onto the SEGMENTS rather
            # than snapped to the sampled points. Snapping folds the
            # longitudinal residual into the lateral measurement: with a 4 m
            # step a vehicle halfway between two points reads as 2 m off-axis,
            # which is outside a 1.75 m half-lane, so a car dead ahead in the
            # ego's own lane was rejected unless it happened to sit on a step
            # boundary. Projection gives the true perpendicular offset and an
            # interpolated arc length, and makes the step size a resolution
            # choice rather than a correctness one.
            best_s = None
            best_off = None
            prev_s, prev_p = 0.0, ego_loc
            for s, point in path:
                vx, vy = point.x - prev_p.x, point.y - prev_p.y
                seg2 = vx * vx + vy * vy
                if seg2 > 0.0:
                    t = ((other_loc.x - prev_p.x) * vx
                         + (other_loc.y - prev_p.y) * vy) / seg2
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                    dx = other_loc.x - (prev_p.x + t * vx)
                    dy = other_loc.y - (prev_p.y + t * vy)
                    off = (dx * dx + dy * dy) ** 0.5
                    if best_off is None or off < best_off:
                        best_off = off
                        best_s = prev_s + t * (s - prev_s)
                prev_s, prev_p = s, point
            if best_off is None or best_off > _HEADWAY_LANE_HALF_W_M:
                continue                          # not in the ego's lane

            gap = best_s - ego_half - other_half
            if gap < 0.0:
                gap = 0.0                         # overlapping: touching, not negative
            if best_gap is None or gap < best_gap:
                best_gap = gap

        if best_gap is None:
            return None, None
        if speed_mps is None or speed_mps < _HEADWAY_MIN_SPEED_MPS:
            return round(best_gap, 2), None
        return round(best_gap, 2), round(best_gap / speed_mps, 3)

    except Exception:
        # A read failure here must not cost the whole sample. The caller's
        # error handling is built around losing a WHOLE record; losing one
        # optional field is strictly better, and None already means "unknown"
        # for both of these.
        return None, None


def collect(actor, world, carla_map) -> dict:
    """One snapshot, with the same field names the HTTP bridge serves so
    DataCollector._poll_vehicle_state consumes either without changes."""
    vel = actor.get_velocity()
    speed_mps = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
    speed_kmh = speed_mps * 3.6
    acc = actor.get_acceleration()
    acceleration = (acc.x ** 2 + acc.y ** 2 + acc.z ** 2) ** 0.5
    control = actor.get_control()
    speed_limit = actor.get_speed_limit()
    weather = world.get_weather()
    waypoint = carla_map.get_waypoint(actor.get_location())
    lead_distance_m, headway_s = lead_and_headway(actor, world, carla_map, speed_mps)

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
        # Both None when there is no vehicle ahead in the ego's lane within
        # _HEADWAY_MAX_RANGE_M; headway alone is None below walking pace, where
        # it is undefined rather than large. None means UNKNOWN in both cases
        # and must not be coerced to 0 downstream -- a 0 s headway is a
        # collision, which is the opposite of "the road ahead is clear".
        "lead_distance_m":     lead_distance_m,
        "headway_s":           headway_s,
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
    # None, not 0: these two mean "no leader / undefined", and DataCollector's
    # defaults are None for the same reason. Zeroing them would say the ego is
    # bumper to bumper with a stationary car, which is the strongest possible
    # signal rather than an inert one.
    out["lead_distance_m"] = None
    out["headway_s"] = None
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
