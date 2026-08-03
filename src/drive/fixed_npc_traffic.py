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

# Step the world is put on while frozen under the FREE-RUNNING clock, and the
# size of the single priming tick on release. Small and fixed, because the danger
# on resume is the server computing its first free-running delta from the WALL
# time that passed while frozen: after a 15 s deliberation that would be a 15 s
# physics step, which would fling every vehicle across the map. Ticking once at
# this step immediately before handing the clock back leaves the server with a
# fresh, recent frame reference instead.
FREEZE_STEP_S = 0.05


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

# Traffic seed used when --traffic-seed is not given. 42 is the value this
# script ran on before the seed was configurable, and start_experiment.py keeps
# it for the calibration and adaptation drives, so the study arms are unchanged
# by the seed becoming an argument. Data-collection runs pass their own.
DEFAULT_TRAFFIC_SEED = 42


# =========================
# PER-VEHICLE TRAFFIC MANAGER PARAMETERS
# =========================
#
# Every traffic manager setting that is PER VEHICLE and has a uniform
# (actor, value) signature, mapped to the method that applies it. This table is
# the single place a knob has to be declared: add a line and it is immediately
# settable from DRIVING_PROFILES and from VEHICLE_TM_OVERRIDES, and immediately
# validated at startup.
#
# Deliberately NOT in the table, because they need code rather than a value:
# collision_detection(reference, other, bool) is a PAIRWISE relation between two
# actors, set_path/set_route take a list of locations or turn codes, and
# force_lane_change is a one-shot command rather than a setting.
#
# The unit and sign traps below are CARLA's, not this script's, and every one of
# them has a wrong-looking-but-correct value somewhere in a config:
#   speed_difference   PERCENT, and NEGATIVE means FASTER than the posted limit.
#                      CARLA's own default is +30, i.e. 70% of the limit. The
#                      full history of this one is in the apply pass in main().
#   desired_speed      km/h, ABSOLUTE, and it REPLACES speed_difference rather
#                      than combining with it. Setting both on one vehicle is a
#                      configuration error and is rejected at startup.
#   distance_to_leading METRES.
#   lane_offset        METRES, positive = towards the right-hand lane boundary,
#                      which in these towns is the kerb and the parked cars. The
#                      margin it spends is roughly 0.8 m on a 3.5 m lane -- see
#                      the "lane_bias" profile for the arithmetic and for why
#                      this knob is not the free one it looks like.
#   *_lanechange, keep_right, ignore_*
#                      0-100, the percent of the time the behaviour applies.
TM_KNOBS = {
    "auto_lane_change":        "auto_lane_change",
    "distance_to_leading":     "distance_to_leading_vehicle",
    "speed_difference":        "vehicle_percentage_speed_difference",
    "desired_speed":           "set_desired_speed",
    "lane_offset":             "vehicle_lane_offset",
    "keep_right":              "keep_right_rule_percentage",
    "random_left_lanechange":  "random_left_lanechange_percentage",
    "random_right_lanechange": "random_right_lanechange_percentage",
    "ignore_lights":           "ignore_lights_percentage",
    "ignore_signs":            "ignore_signs_percentage",
    "ignore_vehicles":         "ignore_vehicles_percentage",
    "ignore_walkers":          "ignore_walkers_percentage",
    "update_lights":           "update_vehicle_lights",
}

# Named bundles of the knobs above, applied on top of the class-based baseline
# (see _baseline_profile) and themselves overridable per vehicle. A profile is a
# PARTIAL dict: anything it does not mention keeps the baseline value, so a
# profile only has to state what makes it different.
#
# Any numeric value may be written as a two-element (low, high) range instead of
# a number, in which case it is drawn once per vehicle from the seeded RNG at
# fleet-resolution time. That is how the baseline speeds have always worked and
# it is what keeps the fleet varied without making it vary BETWEEN PARTICIPANTS
# -- the draw order is fixed by the configuration, not by what happens at
# runtime, so the same config always produces the same fleet.
DRIVING_PROFILES = {
    # Hangs back, does not overtake, sits in the right lane. A rolling
    # roadblock, which is a real and useful thing to have on the route -- but
    # note the finding recorded in the apply pass below: switching overtaking
    # off for the WHOLE fleet made the traffic crawl, so use this in ones and
    # twos, not everywhere.
    "cautious": {
        "speed_difference": (15.0, 30.0),
        "distance_to_leading": 12.0,
        "auto_lane_change": False,
        "keep_right": 80.0,
    },
    # Follows close and will take a gap. Deliberately NOT a red-light runner:
    # everything here stays inside the rules of the road, so it changes how the
    # traffic feels without changing what it yields to.
    "assertive": {
        "speed_difference": (-10.0, 0.0),
        "distance_to_leading": 5.0,
        "auto_lane_change": True,
        "random_left_lanechange": 15.0,
        "random_right_lanechange": 15.0,
        "keep_right": 10.0,
    },
    # Sits off the lane centreline for the whole run. A CONSTANT bias, not a
    # wander -- the offset is drawn once, so this is the driver who habitually
    # hugs one side of the lane, not one who drifts about within it. Real
    # wandering would mean re-setting vehicle_lane_offset from the tick loop,
    # which nothing here does.
    #
    # NOT free, and the earlier claim in this file that it was is wrong. It is
    # free of YIELDING risk -- lane_offset changes where the vehicle sits, not
    # what it gives way to -- but the parked-car failure recorded under
    # LONG_BLUEPRINTS is a GEOMETRY failure, and this spends from exactly the
    # same budget. Measured off Town10HD's OpenDrive:
    #
    #   lane width          3.50 m, every driving lane in the map, all 168
    #   sedan width         ~1.9 m
    #   clearance per side   ~0.80 m before the body crosses the lane edge
    #
    # So 0.2 m spends a quarter of that margin and 0.4 m spends half. Two things
    # then eat the rest, neither of which the traffic manager can see:
    #
    #  - 86 of the map's 153 shoulders are only 0.5-0.6 m wide, which is not
    #    enough to park a ~1.9 m car clear of the road. Wherever those stretches
    #    are lined with parked cars, the meshes already overhang the driving
    #    lane, and the stock fleet only misses them by tracking the centreline.
    #  - 122 of 149 arcs in the map are under 20 m radius, the tightest 7.4 m.
    #    That is where off-tracking happens, and a constant bias adds to it
    #    rather than averaging out: a positive (rightward) offset through a
    #    right-hand bend puts the body toward the kerb for the whole corner.
    #
    # Hence 0.2 m rather than 0.4. Raising it is a real decision, and the thing
    # to watch on the drive is the kerbside on tight corners, not the straights
    # -- on a straight even 0.4 m keeps the body inside the lane. Positive is
    # the parked-car direction, so if contact does show up, biasing the range
    # negative is a cheaper fix than shrinking it further.
    "lane_bias": {
        "lane_offset": (-0.2, 0.2),
        "speed_difference": (5.0, 20.0),
    },
    # HAZARD, and the only profile here that can end a session. It runs lights
    # and periodically stops seeing other vehicles, which sooner or later puts a
    # car across the participant's path -- that is the point of it, but it means
    # it is not scene dressing. Assign it deliberately, to a vehicle whose route
    # relative to the participant's you actually know, and never to the whole
    # fleet. A collision with the ego does not just add a hazard, it invalidates
    # everything measured after it.
    "reckless": {
        "speed_difference": (-25.0, -10.0),
        "distance_to_leading": 3.0,
        "ignore_lights": 30.0,
        "ignore_signs": 50.0,
        "ignore_vehicles": 10.0,
        "auto_lane_change": True,
        "random_left_lanechange": 30.0,
        "random_right_lanechange": 30.0,
    },
}


def _is_range(value):
    """True for a two-element numeric (low, high) to be drawn from.

    Booleans are excluded explicitly: bool is a subclass of int in Python, so
    (True, False) would otherwise look like a numeric range and be silently
    turned into a meaningless uniform draw.
    """
    return (isinstance(value, (tuple, list))
            and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in value))


def _draw(value):
    """Resolve one configured value, drawing from the seeded RNG for a range."""
    return random.uniform(value[0], value[1]) if _is_range(value) else value


def _baseline_profile(heavy, sync_mode):
    """The class-based defaults every vehicle starts from.

    These are the values the fleet ran on before per-vehicle configuration
    existed, unchanged, so a run with an empty VEHICLE_TM_OVERRIDES behaves
    exactly as it did. The reasoning behind each one is in the apply pass in
    main(), kept there because it is a record of live-drive findings rather than
    a description of what the code does.
    """
    if sync_mode:
        speed = (20.0, 35.0) if heavy else (0.0, 15.0)
    else:
        speed = (35.0, 55.0) if heavy else (15.0, 35.0)
    return {
        "auto_lane_change": sync_mode and not heavy,
        "distance_to_leading": 8.0,
        "speed_difference": speed,
    }


def _layer_profile(heavy, sync_mode, override):
    """Baseline -> named profile -> explicit knobs, with the ranges UNDRAWN.

    Layered rather than replaced, so an override only has to state the one thing
    it changes and keeps tracking the baseline for everything else -- which
    matters because the baseline is per-clock, and an override that restated the
    speed would silently stop following --sync/--no-sync.

    Split out from _resolve_profile because it touches no randomness, so it can
    be used to ASK WHAT A VEHICLE WOULD GET without moving the RNG. Anything
    that inspects the configuration -- the off-tracking check before the spawn
    loop, and anything added later -- has to go through this one: a call to
    _resolve_profile for a look would consume draws and silently shift every
    vehicle resolved afterwards, which is the same class of bug as resolving
    the fleet after the spawns.
    """
    profile = _baseline_profile(heavy, sync_mode)

    named = override.get("profile")
    if named:
        profile.update(DRIVING_PROFILES[named])

    for knob, value in override.items():
        if knob != "profile":
            profile[knob] = value

    # An explicit desired_speed replaces the percentage rather than combining
    # with it; leaving both in would apply whichever CARLA happened to see last.
    # The startup validation rejects both being set in ONE override, but a
    # profile supplying one and an override the other is legitimate and lands
    # here.
    if "desired_speed" in profile and "speed_difference" in profile:
        del profile["speed_difference"]

    return profile


def _resolve_profile(heavy, sync_mode, override):
    """_layer_profile, then draw the ranges from the seeded RNG.

    CONSUMES RANDOMNESS -- one draw per range-valued knob. Call it once per
    vehicle, in configuration order; see _layer_profile for the read-only form.
    """
    # dict preserves insertion order, so the draw order -- and therefore the
    # whole fleet -- is fixed by the configuration alone.
    return {knob: _draw(value)
            for knob, value in _layer_profile(heavy, sync_mode, override).items()}


def _validate_overrides(overrides, configured_spawn_indices):
    """Reject a bad override before anything connects to CARLA.

    Worth failing loudly for: a misspelled knob name is not an error anywhere
    else in the chain. TM_KNOBS lookups would just skip it, the fleet would come
    up looking healthy, and the setting the session was arranged to test would
    quietly not be applied. That is a wasted participant, discovered afterwards
    if at all.
    """
    problems = []

    for spawn_index, override in overrides.items():
        where = "VEHICLE_TM_OVERRIDES[%r]" % (spawn_index,)

        if spawn_index not in configured_spawn_indices:
            problems.append(
                "%s: no vehicle is configured at spawn point %r. Keys are spawn "
                "point indices from VEHICLE_CONFIGS (%s)."
                % (where, spawn_index,
                   ", ".join(str(i) for i in sorted(configured_spawn_indices))))
            continue

        if not isinstance(override, dict):
            problems.append("%s: expected a dict of settings, got %s."
                            % (where, type(override).__name__))
            continue

        named = override.get("profile")
        if named is not None and named not in DRIVING_PROFILES:
            problems.append(
                "%s: unknown profile %r. Known profiles: %s."
                % (where, named, ", ".join(sorted(DRIVING_PROFILES))))

        for knob, value in override.items():
            if knob == "profile":
                continue
            if knob not in TM_KNOBS:
                problems.append(
                    "%s: unknown setting %r. Settable per vehicle: %s."
                    % (where, knob, ", ".join(sorted(TM_KNOBS))))
                continue
            if not _is_range(value) and isinstance(value, (tuple, list)):
                problems.append(
                    "%s: %s expects a number, a bool, or a two-element "
                    "(low, high) range; got %r."
                    % (where, knob, value))

        if "desired_speed" in override and "speed_difference" in override:
            problems.append(
                "%s: sets both desired_speed and speed_difference. They are two "
                "ways of saying the same thing and CARLA applies whichever it "
                "sees last -- pick one." % where)

    if problems:
        raise SystemExit(
            "Bad per-vehicle traffic manager configuration:\n  "
            + "\n  ".join(problems))


def _apply_profile(tm, vehicle, profile):
    """Apply a resolved profile, naming the knob if one of them fails.

    Per-knob rather than per-vehicle error handling, because the failure this is
    shaped around is a method missing from a particular CARLA build: that should
    cost the one setting, not the other ten and not the vehicle.
    """
    for knob, value in profile.items():
        try:
            getattr(tm, TM_KNOBS[knob])(vehicle, value)
        except Exception as e:
            print("  [WARN] %s: could not set %s=%r via TrafficManager.%s: %s"
                  % (vehicle.type_id, knob, value, TM_KNOBS[knob], e))


def _format_profile(profile):
    """One-line rendering of a resolved profile, for the fleet summary."""
    parts = []
    for knob, value in profile.items():
        if isinstance(value, bool):
            parts.append("%s=%s" % (knob, "on" if value else "off"))
        elif isinstance(value, (int, float)):
            parts.append("%s=%.1f" % (knob, value))
        else:
            parts.append("%s=%r" % (knob, value))
    return " ".join(parts)


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
    parser.add_argument("--allow-long-vehicles", action="store_true",
                        help="Keep the vans and trucks (sprinter, ambulance, "
                             "firetruck, fuso, carlacola) instead of substituting "
                             "cars for them. OFF by default because they "
                             "off-track: the traffic manager steers a reference "
                             "point down the lane centreline without modelling "
                             "the body behind it, so on tight corners their tails "
                             "sweep through the parked cars at the kerb, block the "
                             "road and collect the rest of the fleet behind them. "
                             "Parked cars are scenery rather than actors, so the "
                             "traffic manager cannot see them and no setting "
                             "prevents this. Substitution keeps the fleet at the "
                             "same size and spawn points, so traffic density is "
                             "unchanged.")
    parser.add_argument("--num-vehicles", type=int, default=0,
                        help="Spawn only the first N of the configured NPCs (0 = "
                             "all of them). THE DIAGNOSTIC FOR SLOW MOTION, and "
                             "currently the only physics lever that is reachable "
                             "from the Python API: measurements on this rig show "
                             "the server spending ~1.17 s of work per SIMULATED "
                             "second, which no tick rate can outrun, and CARLA "
                             "0.10's --substep-delta provably does not move it "
                             "(1 substep per tick performed exactly like 5). If "
                             "that cost is vehicle physics, halving the fleet "
                             "should roughly halve it. Run once with --num-vehicles "
                             "3 and compare the [SYNC] sim speed: a big jump "
                             "confirms vehicle physics, no change rules it out "
                             "and points at the map or the render path instead.")
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
    parser.add_argument("--traffic-seed", type=int, default=DEFAULT_TRAFFIC_SEED,
                        help="Seed for the traffic manager and for the "
                             "per-vehicle draws in VEHICLE_TM_OVERRIDES "
                             "(default %d). THE SCENARIO IDENTITY: the same "
                             "seed reproduces the same fleet behaviour, a "
                             "different one produces a different scenario from "
                             "the same configuration. Only meaningful under "
                             "--sync -- with a variable time step the traffic "
                             "manager's decisions are reproducible but the "
                             "physics they act on is not, so the seed buys "
                             "nothing (see set_random_device_seed below).\n"
                             "\n"
                             "Whatever is passed here MUST end up in the "
                             "collected data. It is the only record of which "
                             "scenario a session ran, it is not recoverable "
                             "from anything else afterwards, and a corpus that "
                             "mixes scenarios without saying which is which "
                             "cannot be split or held out by scenario. "
                             "start_experiment.py carries it to ProVoice for "
                             "exactly that reason; if this script is run by "
                             "hand, write the seed down."
                             % DEFAULT_TRAFFIC_SEED)
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

    # A non-default seed under the free-running clock is almost certainly a
    # mistake, and a silent one: the traffic manager's decisions would be
    # seeded, but the variable-step physics they act on would not be, so the
    # scenario is NOT reproducible and the seed recorded against the session
    # would not identify anything. Warn rather than refuse -- it is legitimate
    # while tuning, just never for a run whose data will be kept.
    if args.traffic_seed != DEFAULT_TRAFFIC_SEED and not args.sync:
        print("[WARN] --traffic-seed %d given without --sync. Under the "
              "free-running clock the seed does NOT make the scenario "
              "reproducible, so it does not identify anything in the collected "
              "data. Use --sync for any run you intend to keep."
              % args.traffic_seed)

    CARLA_HOST = args.host
    CARLA_PORT = args.port

    TM_PORT = args.tm_port

    # Feeds BOTH the traffic manager's own RNG and this script's random module
    # (the per-vehicle draws in VEHICLE_TM_OVERRIDES). One seed for both, so a
    # scenario is identified by a single number rather than by a pair that could
    # drift apart.
    SEED = args.traffic_seed

    NUM_VEHICLES = 50

    SYNC_MODE = args.sync

    FIXED_DELTA_SECONDS = args.delta

    # Honoured in BOTH modes. Under --sync the clock is held by not ticking;
    # under the free-running clock it is held by switching the world INTO
    # synchronous mode and never ticking it, which stops the server dead. Either
    # way the freeze is real -- vehicles keep their position, heading, speed and
    # wheel angle and carry on from there -- so traffic never has to stop at a
    # popup and pull away from rest afterwards.
    PAUSE_FILE = args.pause_file

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

    # =========================
    # PER-VEHICLE BEHAVIOUR -- EDIT THIS TABLE
    # =========================
    #
    # Keyed by SPAWN POINT INDEX, the left column of VEHICLE_CONFIGS above, and
    # not by blueprint id. Two reasons, both of which have bitten:
    #   - the same blueprint appears more than once in a fleet, so a blueprint
    #     key cannot address one vehicle;
    #   - --allow-long-vehicles is off by default, which SUBSTITUTES a car for
    #     each long vehicle. The blueprint at a given spawn point therefore
    #     changes with a command-line flag, while the spawn point does not.
    #
    # Each entry is a partial dict layered on top of the class-based baseline
    # (heavy/car x sync/async), so state only what should differ. Either form
    # works and they can be combined -- the named profile is applied first, then
    # the explicit knobs on top of it:
    #
    #     20: {"profile": "cautious"},
    #     25: {"profile": "assertive", "distance_to_leading": 4.0},
    #     30: {"speed_difference": (-5.0, 5.0), "lane_offset": 0.3},
    #
    # Profiles: see DRIVING_PROFILES at the top of the file -- cautious,
    # assertive, lane_bias, reckless. Individual knobs: see TM_KNOBS, next to it,
    # which also documents the units and the two sign traps.
    #
    # A two-element (low, high) range means "draw once from the seeded RNG",
    # which is how the baseline speeds already work.
    #
    # Empty by default: with no entries the fleet behaves exactly as it did
    # before this table existed, so turning it on is a deliberate act rather
    # than something a merge can do quietly.
    #
    # WHAT NOT TO DO, given what this rig is for. The fleet is meant to be
    # identical for every participant -- that is what the seed and the fixed
    # time step buy (see --sync). So treat this table as part of the STUDY
    # DESIGN, not as scenery: fix it before the first participant and change it
    # only between conditions, never between runs of the same condition. The
    # ignore_* knobs and the "reckless" profile additionally create real
    # collision risk for the participant's own car, which is a different and
    # larger decision than making the traffic look more varied.
    VEHICLE_TM_OVERRIDES = {
        0:  {"profile": "cautious"},
        5: {"profile": "assertive"},
        #10: {"profile": "assertive"},
        15: {"profile": "cautious"},
        20: {"profile": "assertive"},
        25: {"profile": "lane_bias"},
        30: {"profile": "cautious"},
        35: {"profile": "assertive"},
        # sp40 is the nissan.patrol -- the widest vehicle left in the fleet once
        # the long ones are substituted, so it has the least lane margin to give
        # away. It is the one to watch first on the kerbside of a tight corner,
        # and the first candidate to drop back to no offset if anything touches.
        40: {"profile": "assertive"},
        #45: {"profile": "lane_bias"},
        50: {"profile": "lane_bias"},
    }

    # Fails here, before the CARLA connection below, for the same reason the
    # --delta check does: a typo should cost a second, not a server start and a
    # timeout.
    _validate_overrides(VEHICLE_TM_OVERRIDES,
                        {spawn_index for spawn_index, _ in VEHICLE_CONFIGS})

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

    # LONG is a different property from HEAVY, and this is the one that made the
    # vans and trucks pile into parked cars at one particular corner.
    #
    # The traffic manager steers a vehicle's reference point down the lane
    # centreline. It does not model the body being dragged behind that point, so
    # a long vehicle OFF-TRACKS: on a tight corner the rear sweeps a tighter arc
    # than the front and ends up outside the lane while the front is still
    # tracking perfectly. Where the map has cars parked against the kerb, the
    # tail goes through them.
    #
    # Nothing in the traffic manager can prevent this. Parked cars in the CARLA
    # towns are scenery, not registered actors, so its collision avoidance is
    # blind to them by construction -- and even if it could see them, it has no
    # notion of its own swept path. It is also NOT a --sync problem: an exact
    # world snapshot does not make the planner length-aware, which is why it
    # persisted after the lane-change fix and happens with no lane change at all.
    #
    # So the only real remedy is not to run vehicles the map cannot accommodate.
    # nissan.patrol stays: it is tall and heavy (hence HEAVY above, for the speed
    # penalty) but it is car-length, so it does not off-track.
    LONG_BLUEPRINTS = {
        "vehicle.sprinter.mercedes",
        "vehicle.ambulance.ford",
        "vehicle.firetruck.actors",
        "vehicle.fuso.mitsubishi",
        "vehicle.carlacola.actors",
    }

    # Long vehicles are SUBSTITUTED rather than dropped, so the fleet stays at
    # eleven cars on the same eleven spawn points. Traffic density is part of the
    # scene the participant drives in and part of what they are judging the
    # assistant against, so it must not quietly change as a side effect of a
    # stability fix.
    SUBSTITUTE_CARS = [
        "vehicle.lincoln.mkz",
        "vehicle.dodge.charger",
        "vehicle.mini.cooper",
        "vehicle.taxi.ford",
        "vehicle.dodgecop.charger",
    ]

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
    # (actor, spawn_index, blueprint_id, resolved_profile) for the second pass.
    # Separate from vehicles_list, which exists only so the cleanup handler can
    # destroy everything.
    spawned = []

    wanted = list(VEHICLE_CONFIGS)

    if not args.allow_long_vehicles:
        swapped = []
        for i, (spawn_index, blueprint_id) in enumerate(wanted):
            if blueprint_id in LONG_BLUEPRINTS:
                replacement = SUBSTITUTE_CARS[len(swapped) % len(SUBSTITUTE_CARS)]
                wanted[i] = (spawn_index, replacement)
                swapped.append((blueprint_id, replacement))
        if swapped:
            print("Substituting %d long vehicle(s) that off-track into parked "
                  "cars (--allow-long-vehicles to keep them):" % len(swapped))
            for old, new in swapped:
                print(f"  {old} -> {new}")

    # lane_offset on a vehicle that already off-tracks, which is only reachable
    # with --allow-long-vehicles.
    #
    # Checked HERE rather than with the other override validation, because this
    # is the first point at which the FINAL blueprint of each spawn point is
    # known: the substitution above is what decides whether spawn point 45 is a
    # carlacola or a taxi, and the answer depends on a command-line flag. Still
    # before anything spawns.
    #
    # Refused rather than warned. The two failures compound in the same
    # direction and the result is the one already recorded under LONG_BLUEPRINTS
    # -- a tail through the parked cars, the road blocked, the rest of the fleet
    # piling into it. That ends a session, and it would be diagnosed as the old
    # off-tracking bug returning rather than as a line in this table.
    conflicts = [
        (spawn_index, blueprint_id)
        for spawn_index, blueprint_id in wanted
        if blueprint_id in LONG_BLUEPRINTS
        # _layer_profile, NOT _resolve_profile: this runs before the fleet is
        # resolved and must not consume a draw.
        and "lane_offset" in _layer_profile(
            blueprint_id in HEAVY_BLUEPRINTS, SYNC_MODE,
            VEHICLE_TM_OVERRIDES.get(spawn_index, {}))
    ]
    if conflicts:
        raise SystemExit(
            "Per-vehicle configuration sets lane_offset on %d vehicle(s) that "
            "off-track:\n  %s\n"
            "These are the vehicles whose tails already sweep through the "
            "parked cars on tight corners (see LONG_BLUEPRINTS); a lateral "
            "offset spends what little lane margin they have left. Either drop "
            "the offset for those spawn points, or drop --allow-long-vehicles "
            "so they are substituted for cars."
            % (len(conflicts),
               "\n  ".join("spawn point %d: %s" % c for c in conflicts)))

    if args.num_vehicles and args.num_vehicles < len(wanted):
        # Evenly spaced, NOT the first N. The list is ordered with the biggest
        # vehicles first, so a prefix would hand back a fleet of nothing but
        # trucks -- the opposite of a representative sample, and useless for the
        # physics-load comparison --num-vehicles exists to support.
        step = len(wanted) / float(args.num_vehicles)
        wanted = [wanted[int(i * step)] for i in range(args.num_vehicles)]
        print("Spawning only %d of %d configured NPCs, evenly spaced "
              "(--num-vehicles)." % (len(wanted), len(VEHICLE_CONFIGS)))

    # =========================
    # RESOLVE PER-VEHICLE BEHAVIOUR
    # =========================
    #
    # Done for the whole fleet HERE, before the first spawn, rather than in the
    # apply pass -- and the reason is determinism, not tidiness.
    #
    # The ranges are drawn from the seeded RNG, so the draws only reproduce if
    # the SEQUENCE of draws does. Drawing inside the apply pass ties that
    # sequence to which vehicles actually spawned, and try_spawn_actor returns
    # None whenever a spawn point happens to be occupied -- by leftover traffic,
    # or by a participant who parked on one. One such failure would shift every
    # subsequent vehicle's draw, so the fleet would differ between participants
    # for a reason nothing records and nobody would think to look for.
    #
    # Resolving against the CONFIGURED list makes the draw depend on the
    # configuration alone. The blueprint is read after the long-vehicle
    # substitution above, so the heavy/car baseline follows the vehicle that
    # will really be there.
    fleet = [
        (spawn_index, blueprint_id,
         _resolve_profile(blueprint_id in HEAVY_BLUEPRINTS, SYNC_MODE,
                          VEHICLE_TM_OVERRIDES.get(spawn_index, {})))
        for spawn_index, blueprint_id in wanted
    ]

    print("Spawning fixed NPC vehicles...")

    for spawn_index, blueprint_id, profile in fleet:

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
            spawned.append((vehicle, spawn_index, blueprint_id, profile))

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

    # WHY THE BASELINE IN _baseline_profile() IS WHAT IT IS. Each of these three
    # numbers is a record of a live drive, so they are kept next to the fleet
    # they shaped rather than next to the function that returns them. Anything in
    # VEHICLE_TM_OVERRIDES is layered on top of them, which means an override is
    # also a decision to leave one of these findings behind -- read the relevant
    # one first.
    #
    # LANE CHANGES: allowed for CARS under --sync, never for the long heavy
    # vehicles, never under the free-running clock.
    #
    # Three findings, each from a live drive, and the split is what satisfies all
    # of them:
    #
    #  1. Async collided because the traffic manager committed to a gap measured
    #     from a world snapshot up to a full frame stale. --sync removes that
    #     cause: the snapshot is exact.
    #  2. Switching them off entirely made the traffic crawl. With no overtaking,
    #     one slow heavy vehicle becomes a permanent rolling roadblock and the
    #     whole fleet queues behind it.
    #  3. Switching them on for EVERYTHING then had the van/truck sized vehicles
    #     sideswiping cars in the next lane and blocking the road, with the rest
    #     of the fleet piling into the wreck.
    #
    # (3) is the decisive one and it is not a staleness problem, so --sync does
    # not help: the traffic manager plans a lane change from a waypoint path
    # without properly accounting for the length of what it is steering, so a
    # long vehicle's tail is still in the old lane when its nose commits to the
    # new one. Short vehicles have the slack to get away with it; a Sprinter or a
    # Fuso does not.
    #
    # Cars keep the ability to overtake, which is what fixes (2) -- and since the
    # heavy vehicles are also the slow ones, they are exactly what the cars now
    # flow around instead of queueing behind.
    #
    # (2) is also the trap in the "cautious" profile, and (3) the trap in setting
    # auto_lane_change True on a heavy vehicle by hand.
    #
    # FOLLOWING DISTANCE: 8 m, independent of the time step. 5 m is under a 0.4 s
    # headway at urban speeds, where real following distances are 1-2 s, so any
    # leader braking became a rear-end hit with or without exact physics. --sync
    # makes the controller's reaction deterministic; it does not give it more
    # room to react in. Overrides that shorten it are re-entering that region.
    #
    # SPEED, and the reason the sign matters: in this API a NEGATIVE percentage
    # means FASTER than the posted limit. The original random.uniform(-10, 10)
    # ran the whole fleet at 90-110% of the limit, while CARLA's own default for
    # traffic-manager vehicles is +30 (i.e. 70%). The NPCs were doing roughly a
    # third more speed than the controller is tuned for, everywhere, including
    # into corners.
    #
    # The values are per-clock, because what they are compensating for is
    # different in each:
    #
    #   async  large penalties. The spin-outs there are an integration failure --
    #          a coarse substep delta under load -- and the only lever this
    #          script has over that is to lower the speeds the solver has to cope
    #          with.
    #   sync   modest penalties. Physics is integrated exactly, so the only thing
    #          left to respect is that the lateral controller picks a corner
    #          entry speed from road geometry without consulting mass or
    #          centre-of-gravity height. Heavy vehicles still get more, but
    #          nothing needs to crawl, and the first --sync run showed that
    #          crawling is exactly how the previous async-tuned numbers felt.
    #
    # This is the one an override most easily gets wrong in a way that only shows
    # up as a wreck on a corner: a per-clock baseline replaced by a single number
    # stops following --sync/--no-sync, and a negative number applied to a heavy
    # vehicle undoes the whole finding.
    #
    # Whichever clock is used for the study, keep these fixed across every
    # participant and both arms, for the same reason --decision-hz and --delta
    # are fixed.
    print("Enabling autopilot...")

    for vehicle, spawn_index, blueprint_id, profile in spawned:

        try:
            vehicle.set_autopilot(True, TM_PORT)
        except Exception as e:
            print(e)
            continue

        _apply_profile(tm, vehicle, profile)

    # The fleet as it actually ended up, printed once. Under --sync the log is
    # the only record of which behaviour each participant drove among: the
    # ranges are drawn at runtime, so the configuration alone does not say what
    # any individual vehicle was doing, and "the seed makes it reproducible" is
    # only useful if the numbers can be checked against a session afterwards.
    print("Fleet behaviour (seed %d, %s clock):"
          % (SEED, "sync" if SYNC_MODE else "async"))
    for vehicle, spawn_index, blueprint_id, profile in spawned:
        override = VEHICLE_TM_OVERRIDES.get(spawn_index)
        if override:
            named = override.get("profile")
            tag = " [%s]" % (named if named else "override")
        else:
            tag = ""
        print("  sp%-3d %-28s%-12s %s"
              % (spawn_index, blueprint_id, tag, _format_profile(profile)))

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
            # Free-running clock, with an honest pause.
            #
            # There is no tick of ours to withhold here, so the hold is done by
            # putting the world INTO synchronous mode and never ticking it: a
            # synchronous server advances only when a client says so, and no
            # client will. That freezes the scene exactly as it stands, which is
            # what the constant-velocity hold in Drive could never do -- that one
            # pinned every vehicle to 0 m/s, so traffic stopped dead at each
            # popup and pulled away from rest afterwards.
            async_paused = False
            while True:
                want_pause = _pause_requested(PAUSE_FILE)

                if want_pause and not async_paused:
                    try:
                        s = world.get_settings()
                        s.synchronous_mode = True
                        s.fixed_delta_seconds = FREEZE_STEP_S
                        world.apply_settings(s)
                        async_paused = True
                        print("[PAUSE] world held (Drive popup open).")
                    except Exception as e:
                        print("[WARN] could not hold the world:", e)

                elif not want_pause and async_paused:
                    try:
                        # Prime before releasing -- see FREEZE_STEP_S. This tick
                        # advances FREEZE_STEP_S of simulated time, not the
                        # deliberation time, and leaves the server with a current
                        # frame reference so its first free-running delta is
                        # small rather than the whole pause.
                        try:
                            world.tick()
                        except RuntimeError:
                            pass
                        s = world.get_settings()
                        s.synchronous_mode = False
                        s.fixed_delta_seconds = None
                        world.apply_settings(s)
                        print("[PAUSE] world released.")
                    except Exception as e:
                        print("[WARN] could not release the world:", e)
                    finally:
                        # Cleared even if the release throws, so a failure cannot
                        # strand this loop believing the world is still frozen
                        # and leave it held for the rest of the session.
                        async_paused = False

                if async_paused:
                    time.sleep(0.02)
                    continue

                try:
                    # Bounded, and a timeout is NOT an error here: this process
                    # has nothing to do between ticks, and the world may legally
                    # be held by a pause we are about to notice.
                    world.wait_for_tick(2.0)
                except RuntimeError:
                    pass

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
                # DUTY is the number that says WHERE the time goes, and it is the
                # one to read first when the rate is short of target:
                #
                #   duty ~100%  every millisecond is spent inside world.tick().
                #               The server genuinely cannot go faster; the
                #               ceiling below is real and the fix is to make the
                #               server's frame cheaper.
                #   duty  <80%  the loop is idle while still missing the target,
                #               which the pacing code should make impossible.
                #               That points at this script, not at CARLA.
                #
                # Worth having explicitly because achieved-rate ratios alone
                # cannot tell those apart -- two plausible-looking diagnoses
                # (render-bound, then physics-bound) were argued from ratios and
                # both turned out to be wrong.
                duty = min(999.0, 100.0 * mean_cost * achieved)
                print("[SYNC] %.1f Hz of %.1f Hz target | sim speed %.2fx | "
                      "tick %.0f ms mean, %.0f ms worst -> ceiling ~%.0f Hz | "
                      "duty %.0f%% | %.0f substeps/s"
                      % (achieved, target, sim_speed, mean_cost * 1000.0,
                         worst_tick_cost * 1000.0, ceiling, duty,
                         1.0 / args.substep_delta))

                if sim_speed < 0.95 and duty < 80.0:
                    print("[SYNC]     NOTE: only %.0f%% of wall time is spent "
                          "inside world.tick(), yet the target is being missed. "
                          "The server is NOT the limit here -- the time is going "
                          "somewhere else in this loop." % duty)

                if sim_speed < 0.95:
                    print("[SYNC] *** SLOW MOTION: the simulation is running at "
                          "%.0f%% of real time -- everything the participant "
                          "sees, their own car included. THIS RUN IS NOT USABLE "
                          "PARTICIPANT DATA. ***"
                          % (sim_speed * 100.0))
                    if duty >= 80.0:
                        # Deliberately does NOT suggest a --delta. Measured on
                        # this rig, the server's cost is dominated by a term
                        # proportional to SIMULATED time (~1.17 s of work per
                        # simulated second), not to frames: 20 Hz asked gave 15,
                        # 10 Hz asked gave 8, a near-constant ratio. Fitting
                        # cost = A + B*delta puts A at ~8 ms/frame and B at ~1.17,
                        # so sim speed tends to 1/B = 0.86x as delta grows and
                        # NEVER reaches 1.0. Recommending "lower the tick rate"
                        # here was wrong once already; the work has to shrink.
                        print("[SYNC]     %.0f%% of wall time is inside "
                              "world.tick() (%.0f ms/frame). If lowering --delta "
                              "did NOT improve sim speed, the cost scales with "
                              "SIMULATED time, not frames -- no tick rate can fix "
                              "that, only less work per simulated second. Try "
                              "--num-vehicles 3 to test whether it is vehicle "
                              "physics; --render-scale 0.5 on Drive and CARLA "
                              "-RenderOffScreen to test the render path."
                              % (duty, mean_cost * 1000.0))
                    else:
                        print("[SYNC]     Not server-bound (duty %.0f%%): the "
                              "loop is idle yet still missing the target, which "
                              "points at this script rather than at CARLA."
                              % duty)
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