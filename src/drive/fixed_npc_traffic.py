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

    # Vans, trucks and the big SUV. The traffic manager's lateral controller is
    # tuned for a sedan-sized vehicle: it picks a corner entry speed from the
    # road geometry, not from the mass and centre-of-gravity height of the thing
    # it is steering. Five of the eleven blueprints above are heavy and tall, and
    # they are the ones that understeer wide, catch a kerb and then drift. They
    # get a much larger speed penalty below rather than being removed, because
    # the mix of vehicle types is part of the scene the participant is driving in.
    HEAVY_BLUEPRINTS = {
        "vehicle.sprinter.mercedes",
        "vehicle.ambulance.ford",
        "vehicle.firetruck.actors",
        "vehicle.fuso.mitsubishi",
        "vehicle.carlacola.actors",
        "vehicle.nissan.patrol",
    }

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

    # Physics substep headroom -- this is why NPCs spin out and drift.
    #
    # In async mode the frame delta is whatever the server managed, and physics
    # is integrated in substeps capped by max_substeps * max_substep_delta_time.
    # CARLA's defaults (10 * 0.01) give a budget of 0.1 s of simulated time per
    # frame. This rig runs the CARLA server, Drive AND the ProVoice perception
    # stack (YOLO26 + MediaPipe + rPPG + EmotiEffLib) on one machine, so frames
    # routinely take longer than that -- and when they do, the substep delta is
    # stretched past 0.01 s. The tire/suspension model is stiff and integrating
    # it at a coarse delta is exactly what produces "took the corner too fast,
    # now it is drifting": the slip solve diverges, not the driving logic.
    #
    # 24 * 0.01 keeps substeps at or under 0.01 s down to ~4 FPS. Cost is CPU on
    # the server's physics thread, and with 11 vehicles that is negligible --
    # rendering is the bottleneck here, not vehicle dynamics.
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 24

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

            # Lane changes OFF. The traffic manager decides to change lane from
            # its own world snapshot, and in async mode that snapshot is up to a
            # full frame stale -- at the frame times this rig actually hits, the
            # gap it thought was there has moved. This is the source of the
            # "drives into the car next to it" collisions; a vehicle that stays
            # in lane has no such failure mode. Costs nothing the study needs:
            # the NPCs exist to populate the scene, not to overtake.
            tm.auto_lane_change(vehicle, False)

            # 5 m was under a 0.4 s headway at urban speeds -- less than one
            # decision frame of margin, so any leader braking became a rear-end
            # hit. 8 m gives the controller room to respond to a stale snapshot.
            tm.distance_to_leading_vehicle(vehicle, 8.0)

            # Speed, and the reason the sign matters: in this API a NEGATIVE
            # percentage means FASTER than the posted limit. random.uniform(-10,
            # 10) therefore ran the whole fleet at 90-110% of the limit, while
            # CARLA's own default for traffic-manager vehicles is +30 (i.e. 70%
            # of the limit). The NPCs were doing roughly a third more speed than
            # the controller is tuned for, everywhere, including into corners.
            #
            # Positive values here put them back under the limit; heavy vehicles
            # get more, because their stable cornering speed is genuinely lower.
            if blueprint_id in HEAVY_BLUEPRINTS:
                speed_penalty = random.uniform(35, 55)
            else:
                speed_penalty = random.uniform(15, 35)

            tm.vehicle_percentage_speed_difference(vehicle, speed_penalty)

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