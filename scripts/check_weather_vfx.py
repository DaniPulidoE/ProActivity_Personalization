#!/usr/bin/env python3
"""Diagnostic: does THIS CARLA server render rain at all, on ANY map?

Run this with CARLA already started (CarlaUnreal.exe), nothing else needs to
be running. It does three things, each printed so you can correlate with what
you see in the CARLA window:

  1. Reports the current map and the full list of maps this server has, and
     drops the SPECTATOR to a low, close-up vantage point near a spawn point
     -- NOT the default top-down bird's-eye view. That default view is a bad
     test for rain: particles are spawned in a radius around whatever
     camera/pawn is active (for performance), and even the lighting/cloudiness
     change is subtle from directly overhead at altitude. A top-down view
     showing "no change" does not mean the weather system is broken.
  2. Sets HardRainNoon on the CURRENT map (whatever is loaded -- Mine_01 in
     the normal experiment setup) and reads the weather back to prove the RPC
     round-trips correctly.
  3. If a stock CARLA town is available, loads it and sets HardRainNoon there
     too (repositioning the spectator there as well). Watch the CARLA window
     when this happens.

If rain renders in step 3 but not step 2 -- despite both using the identical
set_weather(HardRainNoon) call -- the conclusion is that Mine_01 (or whatever
custom map is loaded) has no weather-reactive Niagara/rain actor placed in
its level, which is a map-authoring issue, not something fixable from the
Python client.

The most representative test of all is still the actual Drive window (the
in-car dash-cam view a participant sees), e.g.:
    python start_experiment.py --test-drive --no-popup --condition-sun-rain
which starts at 80% precipitation immediately -- if this diagnostic and a
real test-drive disagree, trust the test-drive.

Usage:
    python scripts/check_weather_vfx.py [--host 127.0.0.1] [--port 2000]

WARNING: step 3 calls client.load_world(), which is DESTRUCTIVE -- it tears
down the current level, including any spawned vehicles. Do not run this
against a session you care about; run it standalone with a fresh CARLA.
"""
import argparse
import sys
import time

import carla


def _drop_spectator_low(world, label):
    """Move the spectator to a ground-level vantage point near a spawn point.

    CARLA's default spectator sits high above the map looking straight down
    -- exactly the view that makes rain particles and lighting changes hard
    or impossible to notice. A handful of metres above the road, angled
    slightly down, is what a driver's-eye check actually needs.
    """
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print(f"[{label}] No spawn points on this map; spectator left where it was.")
        return
    sp = spawn_points[0]
    loc = sp.location + carla.Location(z=3.0)
    rot = carla.Rotation(pitch=-10.0, yaw=sp.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(loc, rot))
    print(f"[{label}] Spectator moved to ground level near spawn point 0 "
          f"({loc.x:.0f}, {loc.y:.0f}, {loc.z:.0f}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--hold-seconds", type=float, default=20.0,
                     help="How long to hold each rain test before moving on, "
                          "so there is time to actually look at the window.")
    ap.add_argument("--skip-town-test", action="store_true",
                     help="Only run steps 1-2 (non-destructive: never calls "
                          "load_world). Use this if a session is live and "
                          "you don't want to tear down the map.")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    current_map = world.get_map().name
    print("=" * 72)
    print(f"[STEP 1] Current map: {current_map}")
    try:
        available = client.get_available_maps()
        print(f"[STEP 1] Server has {len(available)} map(s) available:")
        for m in available:
            print(f"           {m}")
    except Exception as e:
        print(f"[STEP 1] Could not list available maps: {e}")
        available = []
    print("=" * 72)

    _drop_spectator_low(world, "STEP 2")
    print(f"[STEP 2] Setting HardRainNoon on the CURRENT map ({current_map})...")
    world.set_weather(carla.WeatherParameters.HardRainNoon)
    readback = world.get_weather()
    print(f"[STEP 2] Read back: precipitation={readback.precipitation:.0f} "
          f"cloudiness={readback.cloudiness:.0f} "
          f"precipitation_deposits={readback.precipitation_deposits:.0f}")
    print(f"[STEP 2] RPC round-trip confirms the VALUE is set correctly. "
          f"LOOK AT THE CARLA WINDOW NOW for {args.hold_seconds:.0f}s -- "
          f"do you see rain / dark sky / wet roads?")
    time.sleep(args.hold_seconds)

    if args.skip_town_test:
        print("[STEP 3] Skipped (--skip-town-test). Restoring ClearNoon on "
              f"{current_map}.")
        world.set_weather(carla.WeatherParameters.ClearNoon)
        return

    stock_towns = [m for m in available if "/Town" in m or m.split("/")[-1].startswith("Town")]
    if not stock_towns:
        print("[STEP 3] No stock Town map found on this server -- cannot "
              "A/B test. If Mine_01 (or whatever is loaded) is the ONLY map "
              "this server has, the comparison has to happen some other way "
              "(e.g. ask whoever built the map whether a weather/rain "
              "Blueprint was placed in the level).")
        return

    target = stock_towns[0]
    print(f"[STEP 3] Loading stock map {target} (this WILL tear down "
          f"anything currently in the world)...")
    world = client.load_world(target)
    time.sleep(3.0)  # let the map finish loading before touching weather
    _drop_spectator_low(world, "STEP 3")
    print(f"[STEP 3] Setting HardRainNoon on {target}...")
    world.set_weather(carla.WeatherParameters.HardRainNoon)
    readback = world.get_weather()
    print(f"[STEP 3] Read back: precipitation={readback.precipitation:.0f} "
          f"cloudiness={readback.cloudiness:.0f}")
    print(f"[STEP 3] LOOK AT THE CARLA WINDOW NOW for {args.hold_seconds:.0f}s. "
          f"If rain renders HERE but did not on {current_map} in step 2, "
          f"the custom map is missing the weather VFX actor -- confirmed.")
    time.sleep(args.hold_seconds)

    print("=" * 72)
    print("[DONE] Restoring ClearNoon before you switch back to the "
          "experiment map.")
    world.set_weather(carla.WeatherParameters.ClearNoon)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
