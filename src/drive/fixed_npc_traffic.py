import carla
import random
import signal
import time
import argparse


def _install_graceful_stop():
    """Turn a stop signal into KeyboardInterrupt so the cleanup below runs.

    start_experiment.py asks children to exit with CTRL_BREAK_EVENT, which
    arrives here as SIGBREAK; its default action kills the process outright.
    Re-raising as KeyboardInterrupt hands control to the `except
    KeyboardInterrupt` block at the end of main(), which destroys the spawned
    NPC vehicles.

    That block had never executed in a real session: the launcher previously
    stopped this process with TerminateProcess (Popen.terminate() is an alias
    for kill() on Windows), so every run left its vehicles behind in a CARLA
    world that outlives the run, and severed its CARLA connection abruptly
    rather than closing it.
    """
    def _graceful(_sig, _frame):
        raise KeyboardInterrupt

    for name in ("SIGBREAK", "SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _graceful)
        except (ValueError, OSError):
            pass  # not settable on this platform/thread; default stays


def main():
    _install_graceful_stop()
    # =========================
    # CONFIG
    # =========================

    #CARLA_HOST = "localhost"
    #CARLA_PORT = 2000
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    CARLA_HOST = args.host
    CARLA_PORT = args.port

    TM_PORT = 9000

    SEED = 42

    NUM_VEHICLES = 50

    SYNC_MODE = False

    FIXED_DELTA_SECONDS = 0.05

    # fixed npc vehicle config
    # (spawn_point_index, blueprint_id)

    VEHICLE_CONFIGS = [
        (0, "vehicle.sprinter.mercedes"),
        (5, "vehicle.ambulance.ford"),
        (10, "vehicle.firetruck.actors"),
        (15, "vehicle.lincoln.mkz"),
        (20, "vehicle.dodgecop.charger"),
        (25, "vehicle.mini.cooper"),
        (30, "vehicle.dodge.charger"),
        (35, "vehicle.fuso.mitsubishi"),
        (40, "vehicle.nissan.patrol"),
        (45, "vehicle.carlacola.actors"),
        (50, "vehicle.taxi.ford"),
    ]

    # =========================
    # CONNECT
    # =========================

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)

    world = client.get_world()

    # =========================
    # SYNCHRONOUS MODE
    # =========================

    settings = world.get_settings()

    if SYNC_MODE:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    else:
        settings.synchronous_mode = False

    world.apply_settings(settings)

    # =========================
    # TRAFFIC MANAGER
    # =========================

    tm = client.get_trafficmanager(TM_PORT)

    tm.set_synchronous_mode(SYNC_MODE)

    tm.set_random_device_seed(SEED)

    # Hybrid physics mode is OFF. It was True, and it is the leading suspect for
    # the CARLA server crashes (GameThread, EXCEPTION_ACCESS_VIOLATION reading
    # 0x1b8, see %LOCALAPPDATA%\CarlaUnreal\Saved\Crashes).
    #
    # What it does: vehicles further than the hybrid radius from the HERO actor
    # (Drive's ego -- drive_improved.py's --rolename defaults to "hero") get
    # their physics DISABLED and are moved kinematically; inside the radius
    # physics is switched back on. So while the participant drives, NPCs cross
    # that boundary over and over, and each crossing issues actor-level physics
    # toggles that the server executes on its GameThread -- exactly the thread
    # and exactly the kind of actor-state churn that is faulting.
    #
    # Why turning it off is nearly free: it exists to make fleets of HUNDREDS of
    # vehicles affordable. This script spawns 11 (VEHICLE_CONFIGS has 11 entries;
    # the NUM_VEHICLES = 50 above is dead -- the spawn loop never reads it). Full
    # physics on 11 cars costs almost nothing, so this trades a feature we do not
    # need for the removal of the most complex thing the traffic manager does.
    #
    # Evidence this is the right area: freezing the traffic manager entirely
    # stopped the crashes, and the crash uptimes are scattered from 10 min to
    # 24 h (event-driven, not a timed leak). Evidence it is specifically THIS
    # setting: none yet -- it is the cheapest single thing to eliminate first.
    # If crashes continue, re-enable this and try auto_lane_change(False) next.
    tm.set_hybrid_physics_mode(False)

    # =========================
    # RANDOM SEED
    # =========================

    random.seed(SEED)

    # =========================
    # CLEAN OLD VEHICLES
    # =========================

    print("Destroying old vehicles...")

    actors = world.get_actors()

    old_vehicles = actors.filter('vehicle.*')

    for vehicle in old_vehicles:
        try:
            vehicle.destroy()
        except:
            pass

    time.sleep(1)

    # =========================
    # SPAWN VEHICLES
    # =========================

    blueprint_library = world.get_blueprint_library()

    spawn_points = world.get_map().get_spawn_points()

    vehicles_list = []

    print("Spawning fixed NPC vehicles...")

    for spawn_index, blueprint_id in VEHICLE_CONFIGS:

        if spawn_index >= len(spawn_points):
            print(f"Spawn point {spawn_index} not available")
            continue

        try:

            blueprint = blueprint_library.find(blueprint_id)

            transform = spawn_points[spawn_index]

            vehicle = world.try_spawn_actor(
                blueprint,
                transform
            )

            if vehicle is None:
                print(f"Failed spawn at {spawn_index}")
                continue

            # autopilot
            vehicle.set_autopilot(True, TM_PORT)

            # lane change
            tm.auto_lane_change(vehicle, True)

            tm.distance_to_leading_vehicle(vehicle, 5.0)

            tm.vehicle_percentage_speed_difference(
                vehicle,
                random.uniform(-10, 10)
            )

            vehicles_list.append(vehicle)

            print(
                f"Spawned: {vehicle.type_id} "
                f"at spawn point {spawn_index}"
            )

        except Exception as e:
            print(e)

    # =========================
    # MAIN LOOP
    # =========================

    print("NPC traffic running...")

    # for bp in world.get_blueprint_library().filter('vehicle'):
    #     print(bp.id)

    try:

        while True:

            if SYNC_MODE:
                world.tick()
            else:
                world.wait_for_tick()

    except KeyboardInterrupt:

        print("Cleaning up vehicles...")

        for vehicle in vehicles_list:
            try:
                vehicle.destroy()
            except:
                pass

        settings = world.get_settings()

        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None

        world.apply_settings(settings)

        print("Done.")

if __name__ == "__main__":
    main()