"""Spawn the fixed NPC fleet, and -- with --sync -- own the simulation clock.

WHO TICKS THE WORLD is the whole design question in this file, so it is stated
here rather than left to be rediscovered.

CARLA's traffic manager is not a server-side service: it runs INSIDE the client
process that created it (TrafficManagerLocal), computes controls there and
applies them as batch commands. A traffic manager reached from another process
via the same port is only a remote proxy. That is why the CARLA docs say the TM
"must be set to synchronous mode too in the same client that does the tick" --
the TM's synchronous step is chained to that process's own world.tick() call and
cannot be driven by anyone else's.

The consequence for this project: the process that manages the NPCs and the
process that ticks have to be THE SAME PROCESS. This one manages the NPCs, so in
synchronous mode this one ticks, and Drive (src/drive/drive_improved.py --sync)
becomes a passive client that paces itself with world.wait_for_tick().

A previous attempt put --sync on Drive alone. That set the world synchronous and
synchronised Drive's own traffic manager on port 8000 -- which has no NPC
registered to it -- while the traffic manager actually driving the eleven cars,
on port 9000 here, was never ticked at all. Every NPC stopped dead. That episode
is recorded as a "nonexistent sync mismatch" in start_experiment.py; it was a
real mismatch, and it was this one.

Second reason the clock lives here: the tick loop below is paced against the
wall clock, so one second of simulated time takes one second of real time no
matter how fast Drive is rendering. Tying the tick to Drive's render loop
instead would make simulated time run at (Drive FPS x fixed_delta_seconds) --
fast-forward on a good frame, slow motion on a bad one. Everything this
experiment measures is wall-clock (the 20 s label window, ProVoice's frame
timestamps, the 60 s calibration, rPPG heart rate off a real camera), so
simulated time drifting against real time under machine load would put a
variable, load-dependent amount of driving inside each label window.
"""

import carla
import os
import random
import signal
import time
import argparse


# A pause request older than this is treated as abandoned and ignored. Drive
# removes the file when the popup closes and again in its finally block, so the
# only way one goes stale is Drive dying while a popup is open -- and a dead
# Drive must not be able to freeze the rig indefinitely. Comfortably longer than
# any real deliberation over a five-option prompt.
PAUSE_MAX_AGE_S = 300.0


def _pause_requested(path):
    """True if Drive is currently asking for the clock to be held."""
    if not path:
        return False
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return False  # gone, or unreadable: not a pause
    if age > PAUSE_MAX_AGE_S:
        return False
    return True


# Physics substep budget. CARLA's documented requirement is
#     fixed_delta_seconds <= max_substep_delta_time * max_substeps
# and violating it silently degrades the physics the budget exists to protect,
# so --delta is checked against it at parse time.
#
# THE COUNTER-INTUITIVE PART, and the reason --substep-delta is exposed at all:
# the server runs ceil(delta / substep_delta) physics substeps per tick, and
# 1/delta ticks per simulated second. Multiply them and the delta cancels --
#
#     substeps per simulated second = 1 / substep_delta
#
# -- so the physics cost of one second of simulated time is FIXED, no matter what
# tick rate is asked for. Lowering the tick rate does not buy real time: it makes
# each tick proportionally more expensive. That is why this rig achieved ~0.8x at
# a 20 Hz target and ~0.8x again at 10 Hz, a constant ratio rather than a ceiling.
# Only substep_delta (or cheaper physics) moves that number.
DEFAULT_SUBSTEP_DELTA_TIME = 0.01
DEFAULT_MAX_SUBSTEPS = 24

# How long to let the scene settle after destroying the previous fleet, and again
# after spawning the new one before handing it to the traffic manager.
SETTLE_S = 1.0


def _advance_world(world, seconds, sync_mode, delta):
    """Let SIMULATED time pass, under either clock.

    The distinction this exists to enforce: time.sleep() advances the WALL clock.
    In asynchronous mode the server free-runs, so sleeping also advances the
    simulation and the two are interchangeable. Under --sync they are not --
    simulated time only moves when this process ticks, so a bare sleep advances
    the simulation by exactly nothing.

    That difference caused a real bug. The 1 s sleep after destroying the old
    fleet was quietly load-bearing: it gave the server time to commit the
    destruction and let freshly spawned cars drop the small distance from their
    spawn point onto the road and settle. Under --sync it became a 1 s pause in
    which the world did not move at all, so vehicles were handed to the traffic
    manager still overlapping geometry and mid-drop. One would get a violent
    depenetration impulse at the first tick, end up wrecked at the start line,
    and later fall out of the world and vanish.
    """
    if not sync_mode:
        time.sleep(seconds)
        return
    for _ in range(max(1, int(round(seconds / delta)))):
        world.tick()


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
    parser.add_argument("--tm-port", type=int, default=9000,
                        help="Traffic manager port. This process becomes the "
                             "TM-Server on it; Drive must NOT create its own on "
                             "a different port while in --sync.")
    parser.add_argument("--sync", action="store_true",
                        help="Put the world in synchronous mode with a fixed "
                             "time step and drive the clock from this process "
                             "(see the module docstring for why it has to be "
                             "this process). Drive must be launched with --sync "
                             "too, so it waits on the clock instead of "
                             "free-running. Without this flag the server "
                             "free-runs exactly as before.")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Fixed time step in seconds for --sync (default 0.05 "
                             "= 20 Hz). DO NOT raise this hoping for smoother "
                             "motion. It is a DEMAND on the server, not a quality "
                             "setting: every tick asks for a full rendered frame, "
                             "and if the server cannot deliver them that fast the "
                             "clock falls behind and the whole simulation runs in "
                             "slow motion -- the participant's own car included. "
                             "40 Hz was tried on this rig and did exactly that. "
                             "The ceiling is the server's frame rate, which the "
                             "[SYNC] report now measures and prints; only raise "
                             "--delta if it shows real headroom. Must stay <= "
                             "max_substep_delta_time * max_substeps.")
    parser.add_argument("--substep-delta", type=float,
                        default=DEFAULT_SUBSTEP_DELTA_TIME,
                        help="Maximum physics substep in seconds (default 0.01, "
                             "which is also CARLA's). THE MAIN LEVER ON SIMULATION "
                             "SPEED, because 1/this is the number of physics "
                             "substeps the server must run per simulated second "
                             "REGARDLESS of --delta -- so if the run is in slow "
                             "motion, lowering the tick rate will not help and this "
                             "will. 0.02 halves the physics work. The cost is "
                             "coarser integration, which is what made NPCs spin "
                             "out in the first place, so raise it a step at a time "
                             "and watch the traffic as well as the [SYNC] line.")
    parser.add_argument("--max-substeps", type=int, default=DEFAULT_MAX_SUBSTEPS,
                        help="Cap on physics substeps per tick (default 24). Only "
                             "binds when --delta is large; it is the substep SIZE "
                             "above, not this cap, that sets the cost.")
    parser.add_argument("--pause-file", default="",
                        help="Path Drive uses to ask for the clock to be held "
                             "(--sync only). While the file exists this process "
                             "stops ticking, which freezes the whole scene "
                             "mid-motion -- the honest way to pause a "
                             "simulation. Without it Drive falls back to holding "
                             "each vehicle at zero velocity, which makes traffic "
                             "stop dead and pull away from rest at every popup.")
    args = parser.parse_args()

    # Before connecting, so a bad --delta fails instantly instead of after a
    # CARLA connection attempt.
    if args.sync and args.substep_delta <= 0:
        parser.error("--substep-delta must be positive; got %.4f."
                     % args.substep_delta)
    if args.sync and args.max_substeps < 1:
        parser.error("--max-substeps must be at least 1; got %d."
                     % args.max_substeps)
    substep_budget = args.substep_delta * args.max_substeps
    if args.sync and args.delta > substep_budget:
        parser.error(
            "--delta %.4f exceeds the physics substep budget %.4f "
            "(--substep-delta %.4f * --max-substeps %d). CARLA would quietly "
            "integrate physics at a coarser step than requested, which is the "
            "failure --sync exists to remove."
            % (args.delta, substep_budget, args.substep_delta,
               args.max_substeps))
    if args.sync and args.delta <= 0:
        parser.error("--delta must be positive; got %.4f." % args.delta)

    CARLA_HOST = args.host
    CARLA_PORT = args.port

    TM_PORT = args.tm_port

    SEED = 42

    NUM_VEHICLES = 50

    SYNC_MODE = args.sync

    FIXED_DELTA_SECONDS = args.delta

    PAUSE_FILE = args.pause_file if SYNC_MODE else ""

    # A flag left behind by a Drive that died mid-popup would otherwise hold this
    # run's clock from the very first tick, which looks exactly like the rig
    # hanging on startup. Cleared before anything depends on it.
    if PAUSE_FILE:
        try:
            os.remove(PAUSE_FILE)
            print(f"Removed stale clock-pause flag {PAUSE_FILE}")
        except OSError:
            pass

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
    # Longer under --sync: world.tick() blocks until the server finishes the
    # frame, and this process is now the only thing advancing the simulation, so
    # a timeout here does not just fail a call -- it stops the world for the
    # participant and for Drive. A generous ceiling plus the retry in the tick
    # loop below rides out a server hitch instead of ending the session over one.
    client.set_timeout(60.0 if args.sync else 10.0)

    world = client.get_world()

    # =========================
    # SYNCHRONOUS MODE
    # =========================

    original_settings = world.get_settings()
    settings = world.get_settings()

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
    # A cap of 24 keeps substeps at or under --substep-delta down to ~4 FPS.
    #
    # CORRECTION, measured on this rig: an earlier version of this comment said
    # the CPU cost was "negligible with 11 vehicles -- rendering is the
    # bottleneck here, not vehicle dynamics". That was a guess and it was wrong.
    # Fitting the achieved rate at two different tick rates (20 Hz target -> ~15
    # Hz, 10 Hz target -> 8 Hz) separates the two costs: about 8 ms per frame of
    # rendering against about 12 ms per physics SUBSTEP. At 100 substeps per
    # simulated second that is roughly 1.2 s of physics for every 1.0 s of
    # simulated time, against 0.17 s of rendering -- physics is ~88% of the bill
    # and rendering is close to noise.
    #
    # Under --sync the substep count per tick is exact and constant, which is
    # what removes the async spin-outs. What it does NOT do is get cheaper at a
    # lower tick rate: see the header above DEFAULT_SUBSTEP_DELTA_TIME for why
    # the delta cancels out. --substep-delta is the only knob here that changes
    # the total.
    settings.substepping = True
    settings.max_substep_delta_time = args.substep_delta
    settings.max_substeps = args.max_substeps

    if SYNC_MODE:
        # --delta was already checked against the substep budget at parse time.
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    else:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None

    world.apply_settings(settings)

    # =========================
    # TRAFFIC MANAGER
    # =========================

    tm = client.get_trafficmanager(TM_PORT)

    # Deliberately right after apply_settings() and in the process that ticks --
    # both are documented requirements, and this is the call the earlier failed
    # attempt made from the wrong process. See the module docstring.
    tm.set_synchronous_mode(SYNC_MODE)

    # Only meaningful under --sync. With a variable time step the traffic
    # manager's own decisions are reproducible but the physics they act on is
    # not, so seeding buys nothing: async traffic differs run to run, which for
    # a between-participants study is an uncontrolled variable. Fixed step plus
    # this seed is what makes the traffic the same for every participant.
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
            # Never destroy the participant's car. In the normal launch order
            # this process runs before Drive and there is no ego yet, so this
            # guard is invisible -- but it makes restarting NPC traffic against
            # a live session survivable instead of ending the drive, and it
            # removes the ordering landmine from any future change that starts
            # Drive first. 'hero' is drive_improved.py's default --rolename.
            if vehicle.attributes.get('role_name') == 'hero':
                print(f"Keeping ego vehicle {vehicle.id} (role_name=hero)")
                continue
            vehicle.destroy()
        except:
            pass

    # Ticks under --sync, sleeps otherwise -- see _advance_world. A bare sleep
    # here is what left a wrecked car on the start line.
    _advance_world(world, SETTLE_S, SYNC_MODE, FIXED_DELTA_SECONDS)

    # =========================
    # SPAWN VEHICLES
    # =========================

    blueprint_library = world.get_blueprint_library()

    spawn_points = world.get_map().get_spawn_points()

    vehicles_list = []
    # (actor, blueprint_id) for the second pass. Separate from vehicles_list,
    # which exists only so the cleanup handler can destroy everything.
    spawned = []

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

            # NOT handed to the traffic manager yet -- that happens in a second
            # pass, after the settle below. A car is spawned a short distance
            # above the road and has to drop onto it; enabling autopilot here
            # means the traffic manager starts steering and throttling it while
            # it is still airborne, and it lands under power with the wheels
            # already turned. That is survivable for a hatchback and not for a
            # firetruck.
            vehicles_list.append(vehicle)
            spawned.append((vehicle, blueprint_id))

            print(
                f"Spawned: {vehicle.type_id} "
                f"at spawn point {spawn_index}"
            )

        except Exception as e:
            print(e)

    # Let the fleet drop, settle on its suspension and come to rest BEFORE the
    # traffic manager touches it.
    print("Settling %d vehicles..." % len(spawned))
    _advance_world(world, SETTLE_S, SYNC_MODE, FIXED_DELTA_SECONDS)

    # =========================
    # HAND THE FLEET TO THE TRAFFIC MANAGER
    # =========================

    print("Enabling autopilot...")

    for vehicle, blueprint_id in spawned:

        try:

            vehicle.set_autopilot(True, TM_PORT)

            # Lane changes: OFF under the free-running clock, ON under --sync.
            #
            # Two separate things made traffic-manager lane changes collide here.
            # One was the time step: it committed to a gap measured from a world
            # snapshot up to a full frame stale, and at the frame times this rig
            # hits in async the gap had already moved. --sync removes that
            # entirely -- the snapshot is exact and the reaction deterministic.
            # The other is that its gap check is imperfect in any mode.
            #
            # Under --sync the first cause is gone and the remaining risk is
            # worth taking, because switching them off had a consequence that
            # only shows up in a live drive: with no overtaking, one slow heavy
            # vehicle turns into a permanent rolling roadblock and the entire
            # fleet queues behind it. That reads as far more unnatural to a
            # participant than an occasional imperfect merge, and a queue of
            # stationary traffic is its own confound.
            tm.auto_lane_change(vehicle, SYNC_MODE)

            # 8 m, independent of the time step: 5 m is under a 0.4 s headway at
            # urban speeds, where real following distances are 1-2 s, so any
            # leader braking became a rear-end hit with or without exact physics.
            # --sync makes the controller's reaction deterministic; it does not
            # give it more room to react in.
            tm.distance_to_leading_vehicle(vehicle, 8.0)

            # Speed, and the reason the sign matters: in this API a NEGATIVE
            # percentage means FASTER than the posted limit. The original
            # random.uniform(-10, 10) ran the whole fleet at 90-110% of the
            # limit, while CARLA's own default for traffic-manager vehicles is
            # +30 (i.e. 70%). The NPCs were doing roughly a third more speed than
            # the controller is tuned for, everywhere, including into corners.
            #
            # The values are per-clock, because what they are compensating for is
            # different in each:
            #
            #   async  large penalties. The spin-outs there are an integration
            #          failure -- a coarse substep delta under load -- and the
            #          only lever this script has over that is to lower the
            #          speeds the solver has to cope with.
            #   sync   modest penalties. Physics is integrated exactly, so the
            #          only thing left to respect is that the lateral controller
            #          picks a corner entry speed from road geometry without
            #          consulting mass or centre-of-gravity height. Heavy
            #          vehicles still get more, but nothing needs to crawl, and
            #          the first --sync run showed that crawling is exactly how
            #          the previous async-tuned numbers felt.
            #
            # Whichever clock is used for the study, keep these fixed across
            # every participant and both arms, for the same reason
            # --decision-hz and --delta are fixed.
            if SYNC_MODE:
                speed_range = (20, 35) if blueprint_id in HEAVY_BLUEPRINTS \
                    else (0, 15)
            else:
                speed_range = (35, 55) if blueprint_id in HEAVY_BLUEPRINTS \
                    else (15, 35)

            tm.vehicle_percentage_speed_difference(
                vehicle, random.uniform(*speed_range))

        except Exception as e:
            print(e)

    # =========================
    # MAIN LOOP
    # =========================

    if SYNC_MODE:
        print("NPC traffic running -- SYNCHRONOUS, this process owns the clock "
              "(%.3f s step, %.1f Hz)."
              % (FIXED_DELTA_SECONDS, 1.0 / FIXED_DELTA_SECONDS))
    else:
        print("NPC traffic running -- asynchronous, the server free-runs.")

    # for bp in world.get_blueprint_library().filter('vehicle'):
    #     print(bp.id)

    try:

        if not SYNC_MODE:
            while True:
                world.wait_for_tick()

        # Wall-clock-paced tick loop.
        #
        # Each world.tick() advances simulated time by exactly
        # FIXED_DELTA_SECONDS, so the ONLY thing keeping simulated time equal to
        # real time is the rate we call it at. The deadline is advanced by a
        # fixed step rather than measured from "now" after each tick, so the
        # small per-tick overshoots do not accumulate into a permanent lag.
        #
        # When the server cannot keep up, world.tick() itself blocks longer than
        # the step and the loop falls behind. We do NOT try to catch up by
        # ticking faster: that would make the sim jump, and the participant is
        # driving it. We reset the deadline (dropping the debt) and report the
        # lag, because a rig that cannot hold the step is running the study in
        # slow motion and that is a finding, not something to hide. Time.sleep()
        # is accurate to about a millisecond on Python 3.11+ on Windows, which
        # is fine against a 50 ms step.
        LAG_REPORT_EVERY_S = 30.0
        FIRST_REPORT_S = 8.0
        report_every = FIRST_REPORT_S
        next_tick = time.perf_counter()
        ticks = 0
        lagged_ticks = 0
        worst_lag = 0.0
        tick_cost_sum = 0.0
        worst_tick_cost = 0.0
        last_report = time.perf_counter()

        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3
        paused = False

        while True:

            # Drive asking for a hold while a LoA popup is open. Not ticking IS
            # the pause: every vehicle stays exactly where and as it was, mid
            # corner, mid overtake, wheels turned, at whatever speed it had, and
            # resumes from there when the file goes away. Nothing is overridden,
            # no velocity is injected and no car has to pull away from rest, so
            # the resume is invisible -- from the simulation's point of view no
            # time passed, because none did.
            if _pause_requested(PAUSE_FILE):
                if not paused:
                    paused = True
                    print("[SYNC] clock held (Drive popup open).")
                time.sleep(FIXED_DELTA_SECONDS)
                # Deadline rebased so the hold is not then reported as lag.
                next_tick = time.perf_counter()
                continue
            if paused:
                paused = False
                print("[SYNC] clock released.")
                next_tick = time.perf_counter()

            try:
                # Timed, because how long the SERVER takes to complete a frame is
                # the ceiling on the tick rate and the number nobody had. It
                # includes rendering Drive's camera sensor, which on a fullscreen
                # viewport is usually the dominant cost -- so the ceiling is a
                # property of the render load, and a machine dedicated to CARLA
                # does not raise it much.
                t_before = time.perf_counter()
                world.tick()
                tick_cost = time.perf_counter() - t_before
                tick_cost_sum += tick_cost
                worst_tick_cost = max(worst_tick_cost, tick_cost)
                consecutive_failures = 0
            except RuntimeError as e:
                # A tick that times out or errors must not end this process by
                # default: it is the clock, so exiting freezes the scene for a
                # participant mid-drive. Retry a few times -- a server hitch or a
                # brief RPC stall recovers -- and only give up once it is clearly
                # not coming back, in which case the finally block restores
                # asynchronous mode so the rig is left usable rather than stuck
                # waiting for a tick nobody will send.
                consecutive_failures += 1
                print("[SYNC] tick failed (%d/%d): %s"
                      % (consecutive_failures, MAX_CONSECUTIVE_FAILURES, e))
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("[SYNC] giving up on the simulation clock.")
                    raise
                next_tick = time.perf_counter()
                continue

            ticks += 1

            next_tick += FIXED_DELTA_SECONDS
            slack = next_tick - time.perf_counter()

            if slack > 0:
                time.sleep(slack)
            else:
                lagged_ticks += 1
                worst_lag = max(worst_lag, -slack)
                next_tick = time.perf_counter()

            now = time.perf_counter()
            if now - last_report >= report_every:
                achieved = ticks / (now - last_report)
                sim_speed = achieved * FIXED_DELTA_SECONDS
                target = 1.0 / FIXED_DELTA_SECONDS
                mean_cost = tick_cost_sum / max(1, ticks)
                # What the server could actually sustain if we asked for nothing
                # more than it can render.
                ceiling = 1.0 / mean_cost if mean_cost > 0 else float('inf')

                # Sim speed is reported ALWAYS, not only when ticks run late,
                # because it is the one number that says whether the participant
                # is driving in real time -- and "on schedule" hid the answer.
                # Everything moving in slow motion, ego included, looks nothing
                # like a clock problem from the driver's seat; it looks like the
                # cars are slow. That misread cost a debugging cycle.
                # substeps/s is printed because it, not the tick rate, is what
                # predicts sim speed -- and it stays put when --delta changes,
                # which is the whole counter-intuitive point.
                print("[SYNC] %.1f Hz of %.1f Hz target | sim speed %.2fx | "
                      "server frame %.0f ms mean, %.0f ms worst (ceiling ~%.0f "
                      "Hz) | %.0f physics substeps/s"
                      % (achieved, target, sim_speed, mean_cost * 1000.0,
                         worst_tick_cost * 1000.0, ceiling,
                         1.0 / args.substep_delta))

                if sim_speed < 0.95:
                    # The advice deliberately does NOT say "lower --delta". That
                    # is the intuitive fix and it does not work: substeps per
                    # simulated second are 1/substep_delta whatever the tick rate,
                    # so a lower tick rate buys nothing. Measured here as a
                    # constant ~0.8x at both 20 Hz and 10 Hz targets.
                    suggested_sub = args.substep_delta / max(0.05, sim_speed)
                    print("[SYNC] *** SLOW MOTION: the simulation is running at "
                          "%.0f%% of real time -- everything the participant "
                          "sees, their own car included. THIS RUN IS NOT USABLE "
                          "PARTICIPANT DATA. ***"
                          % (sim_speed * 100.0))
                    print("[SYNC]     Lowering --delta will NOT fix this: the "
                          "server must run %.0f physics substeps per simulated "
                          "second at ANY tick rate, because --substep-delta is "
                          "%.4f. Try --substep-delta %.4f (%.0f substeps/s), or "
                          "reduce physics cost another way."
                          % (1.0 / args.substep_delta, args.substep_delta,
                             suggested_sub, 1.0 / suggested_sub))
                elif lagged_ticks:
                    print("[SYNC] %d/%d ticks late, worst %.0f ms -- holding real "
                          "time, but with little headroom."
                          % (lagged_ticks, ticks, worst_lag * 1000.0))

                ticks = 0
                lagged_ticks = 0
                worst_lag = 0.0
                tick_cost_sum = 0.0
                worst_tick_cost = 0.0
                last_report = now
                # First report comes early so a misconfigured rate is caught
                # before the participant has driven for half a minute.
                report_every = LAG_REPORT_EVERY_S

    except KeyboardInterrupt:
        pass

    finally:
        # This runs on ANY exit path, not just Ctrl-C/SIGBREAK, and that matters
        # much more under --sync than it used to: a server left in synchronous
        # mode with no client ticking it blocks every other client that connects
        # to it, so an unhandled exception here would take out the whole rig and
        # leave CARLA needing a restart. Restoring settings comes first for the
        # same reason -- if destroying vehicles throws, sync mode is already off.
        print("Restoring world settings...")

        try:
            world.apply_settings(original_settings)
        except Exception as e:
            print("[WARN] Could not restore world settings:", e)

        if SYNC_MODE:
            try:
                tm.set_synchronous_mode(False)
            except Exception as e:
                print("[WARN] Could not desynchronise the traffic manager:", e)

        print("Cleaning up vehicles...")

        for vehicle in vehicles_list:
            try:
                vehicle.destroy()
            except:
                pass

        print("Done.")

if __name__ == "__main__":
    main()