#!/usr/bin/env python3
"""Verify the brake path end to end: pedal -> VehicleControl.brake -> tyres.

"The car brakes slowly" has three possible causes and they need different fixes,
so each is measured separately:

    pedal    Does a full press of the brake pedal actually produce brake = 1.0?
             The mapping in drive_improved assumes the axis rests at exactly
             +1.0 and travels to exactly -1.0. Real potentiometers rarely span
             the full range, and the deadzone is chopped but NOT rescaled, so a
             pedal that only reaches -0.6 caps the brake at 0.8 forever.
             Needs the wheel, does not need CARLA.

    physics  What the simulator will do with brake = 1.0: per-wheel
             max_brake_torque, ABS, tyre friction, mass -> theoretical peak
             deceleration. Needs CARLA, does not touch the running session.

    test     Ground truth. Accelerates a throwaway car to a target speed, slams
             the brake, and reports deceleration, stopping time and stopping
             distance against real-car references. Needs CARLA.

Run with no mode to do all three (pedal first, then whichever need CARLA):

    uv run python scripts/check_braking.py
    uv run python scripts/check_braking.py pedal
    uv run python scripts/check_braking.py test --speed 50 --trace

The pedal mapping constants are imported from src/drive/drive_improved.py, not
copied, so this cannot drift away from what the participant actually drives.
"""
import os
import sys

# Must precede the pygame import that comes in with drive_improved: the wheel is
# polled with no window, and axes have to keep updating without window focus.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import argparse
import math
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import carla
import pygame

from src.drive.drive_improved import (
    WHEEL_AXIS_BRAKE,
    WHEEL_AXIS_COMBINED,
    WHEEL_AXIS_STEER,
    WHEEL_AXIS_THROTTLE,
    WHEEL_COMBINED_THROTTLE_SIGN,
    WHEEL_PEDAL_DEADZONE,
    WHEEL_STEER_DEADZONE,
)

VEHICLE_BP = 'vehicle.dodge.charger'
G = 9.81


# ==============================================================================
# -- pedal ---------------------------------------------------------------------
# ==============================================================================


def map_pedals(js, combined):
    """The exact mapping KeyboardControl._parse_vehicle_wheel applies.

    Returns (throttle, brake) after the deadzone, plus the raw axis readings so
    a pedal that never reaches its end stop is visible rather than inferred.
    """
    if combined:
        raw = js.get_axis(WHEEL_AXIS_COMBINED)
        signed = raw * WHEEL_COMBINED_THROTTLE_SIGN
        throttle, brake = max(0.0, signed), max(0.0, -signed)
        raws = {'combined': raw}
    else:
        raw_t = js.get_axis(WHEEL_AXIS_THROTTLE)
        raw_b = js.get_axis(WHEEL_AXIS_BRAKE)
        throttle = 1.0 - (raw_t + 1.0) / 2.0
        brake = 1.0 - (raw_b + 1.0) / 2.0
        raws = {'throttle': raw_t, 'brake': raw_b}
    if throttle < WHEEL_PEDAL_DEADZONE:
        throttle = 0.0
    if brake < WHEEL_PEDAL_DEADZONE:
        brake = 0.0
    return min(1.0, throttle), min(1.0, brake), raws


def run_pedal(args):
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("[FAIL] No wheel detected. Plug it in, or the participant is on "
              "the keyboard (where brake is a 0.2-per-frame ramp on S, not a "
              "pedal at all).")
        return False

    js = pygame.joystick.Joystick(0)
    js.init()
    axes = js.get_numaxes()
    combined = axes < 3
    print("Wheel : %r" % js.get_name())
    print("Axes  : %d  ->  %s pedals" % (axes, "COMBINED" if combined else "separate"))
    if combined:
        print()
        print("[FAIL] COMPATIBILITY mode. Throttle and brake are summed onto one")
        print("       axis, so they cannot be pressed independently and any")
        print("       residual throttle SUBTRACTS from the brake. This alone")
        print("       explains a car that brakes weakly. Install Logitech Gaming")
        print("       Software / G HUB to get native mode with four axes.")
        print()

    print()
    print("Press the BRAKE pedal to the floor a few times, then the THROTTLE,")
    print("then release both and let them rest. Ctrl-C when done (%ds max).\n"
          % args.seconds)

    lo = {}
    hi = {}
    peak_brake = 0.0
    peak_throttle = 0.0
    rest_brake = []
    samples = 0
    t_start = time.monotonic()
    t_last_print = 0.0

    try:
        while time.monotonic() - t_start < args.seconds:
            pygame.event.pump()
            throttle, brake, raws = map_pedals(js, combined)
            steer = js.get_axis(WHEEL_AXIS_STEER)
            samples += 1
            for name, val in raws.items():
                lo[name] = min(lo.get(name, val), val)
                hi[name] = max(hi.get(name, val), val)
            peak_brake = max(peak_brake, brake)
            peak_throttle = max(peak_throttle, throttle)
            if brake == 0.0 and throttle == 0.0:
                rest_brake.append(raws.get('brake', raws.get('combined', 0.0)))

            now = time.monotonic() - t_start
            if now - t_last_print >= 0.05:
                t_last_print = now
                raw_txt = "  ".join("%s %+.3f" % (k, v) for k, v in raws.items())
                sys.stdout.write(
                    "\r  %s | steer %+.3f | throttle %.3f %-11s | brake %.3f %-11s"
                    % (raw_txt, steer, throttle, _bar(throttle), brake, _bar(brake)))
                sys.stdout.flush()
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - t_start
    print("\n")
    print("-" * 68)
    print("Polling rate      : %.0f Hz over %.1f s" % (samples / max(elapsed, 1e-6), elapsed))
    for name in sorted(lo):
        print("Axis %-9s    : travelled %+.3f .. %+.3f  (span %.3f of the 2.000 available)"
              % (name, lo[name], hi[name], hi[name] - lo[name]))
    print("Peak brake mapped : %.3f" % peak_brake)
    print("Peak throttle     : %.3f" % peak_throttle)
    print("-" * 68)

    ok = True
    if peak_brake < 0.05:
        print("[FAIL] The brake pedal never moved. Wrong axis index: brake is")
        print("       read from axis %d. Press ONLY the brake and watch which"
              % WHEEL_AXIS_BRAKE)
        print("       raw value above changes, then fix WHEEL_AXIS_BRAKE in")
        print("       src/drive/drive_improved.py.")
        ok = False
    elif peak_brake < 0.95:
        deficit = (1.0 - peak_brake) * 100.0
        print("[FAIL] Full pedal travel only reaches brake = %.2f, so the car is "
              "permanently" % peak_brake)
        print("       missing %.0f%% of its braking force. The mapping assumes the "
              "axis" % deficit)
        print("       reaches -1.000; yours stops at %+.3f."
              % lo.get('brake', lo.get('combined', float('nan'))))
        print("       Fix by calibrating the axis range instead of assuming it")
        print("       (see the note this script prints at the end).")
        ok = False
    else:
        print("[ OK ] Full pedal travel reaches brake = %.2f." % peak_brake)

    if rest_brake and not combined:
        rest = sum(rest_brake) / len(rest_brake)
        resting_brake = max(0.0, 1.0 - (rest + 1.0) / 2.0)
        if resting_brake > WHEEL_PEDAL_DEADZONE:
            print("[WARN] The brake pedal rests at %+.3f, i.e. brake = %.3f with your"
                  % (rest, resting_brake))
            print("       foot off it. Dragging brakes.")
        else:
            print("[ OK ] Brake pedal rests inside the deadzone (no brake drag).")

    if peak_throttle < 0.05:
        print("[WARN] The throttle pedal never moved either -- if you did press it,")
        print("       WHEEL_AXIS_THROTTLE (%d) is wrong too." % WHEEL_AXIS_THROTTLE)
    return ok


def _bar(v, width=10):
    n = int(round(v * width))
    return "[" + "#" * n + "." * (width - n) + "]"


# ==============================================================================
# -- CARLA helpers -------------------------------------------------------------
# ==============================================================================


def connect(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    client.get_server_version()
    return client


def speed_kmh(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def find_hero(world):
    for a in world.get_actors().filter('vehicle.*'):
        if a.attributes.get('role_name') == 'hero':
            return a
    try:
        with open(os.path.join(_ROOT, 'vehicle_id.txt')) as f:
            return world.get_actor(int(f.read().strip()))
    except Exception:
        return None


class Ticker:
    """One step of the world, whoever owns the clock.

    Under --sync the clock owner is src/drive/fixed_npc_traffic.py, never this
    script: calling tick() from a second client would double-step the world and
    corrupt the measurement (and the running session, if there is one).
    """

    def __init__(self, world, self_tick=False):
        self.world = world
        settings = world.get_settings()
        self.sync = settings.synchronous_mode
        self.dt = settings.fixed_delta_seconds or 0.0
        self.self_tick = self_tick and self.sync

    def step(self, timeout=5.0):
        if self.self_tick:
            self.world.tick()
            return self.world.get_snapshot()
        try:
            return self.world.wait_for_tick(timeout)
        except RuntimeError:
            return None


# ==============================================================================
# -- write probe ---------------------------------------------------------------
# ==============================================================================


def _probe_one(vehicle, ticker, client, label, mutate, read, use_batch=False):
    """Write one field, tick, read it back, restore. Reports whether it stuck.

    CARLA 0.10 moved vehicle physics onto Chaos and not every field of
    VehiclePhysicsControl survives the round trip. Which ones do is a property
    of the build, so it is measured rather than assumed.
    """
    original_pc = vehicle.get_physics_control()
    before = read(vehicle.get_physics_control())

    pc = vehicle.get_physics_control()
    want = mutate(pc)
    if use_batch:
        client.apply_batch_sync(
            [carla.command.ApplyVehiclePhysicsControl(vehicle.id, pc)], False)
    else:
        vehicle.apply_physics_control(pc)
    for _ in range(3):
        ticker.step()
    after = read(vehicle.get_physics_control())

    if isinstance(want, bool):
        stuck = (after == want)
    else:
        stuck = abs(float(after) - float(want)) <= max(1e-3, abs(want) * 1e-3)
    print("  %-34s %-10s -> %-10s  want %-10s  %s"
          % (label, _fmt(before), _fmt(after), _fmt(want),
             "STICKS" if stuck else "IGNORED"))

    vehicle.apply_physics_control(original_pc)
    for _ in range(2):
        ticker.step()
    return stuck


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    return "%.3f" % float(v)


def probe_writes(vehicle, ticker, client):
    """Which physics fields can actually be written on this CARLA build?"""
    print("Probing which VehiclePhysicsControl fields survive a write+tick.")
    print("Each field is written, ticked, read back, then restored.\n")

    def wheels_set(pc, field, value):
        ws = []
        for w in pc.wheels:
            setattr(w, field, value)
            ws.append(w)
        pc.wheels = ws
        return value

    results = {}
    print("TOP-LEVEL FIELDS")
    results['mass'] = _probe_one(
        vehicle, ticker, client, 'mass',
        lambda pc: setattr(pc, 'mass', pc.mass * 0.5) or pc.mass,
        lambda pc: pc.mass)
    results['drag_coefficient'] = _probe_one(
        vehicle, ticker, client, 'drag_coefficient',
        lambda pc: setattr(pc, 'drag_coefficient', pc.drag_coefficient + 1.0) or pc.drag_coefficient,
        lambda pc: pc.drag_coefficient)
    results['use_sweep_wheel_collision'] = _probe_one(
        vehicle, ticker, client, 'use_sweep_wheel_collision',
        lambda pc: setattr(pc, 'use_sweep_wheel_collision',
                           not pc.use_sweep_wheel_collision) or pc.use_sweep_wheel_collision,
        lambda pc: pc.use_sweep_wheel_collision)

    print("\nPER-WHEEL FIELDS (apply_physics_control)")
    results['max_brake_torque'] = _probe_one(
        vehicle, ticker, client, 'wheels[].max_brake_torque',
        lambda pc: wheels_set(pc, 'max_brake_torque', 4000.0),
        lambda pc: pc.wheels[0].max_brake_torque)
    results['friction_force_multiplier'] = _probe_one(
        vehicle, ticker, client, 'wheels[].friction_force_multiplier',
        lambda pc: wheels_set(pc, 'friction_force_multiplier', 3.5),
        lambda pc: pc.wheels[0].friction_force_multiplier)
    results['abs_enabled'] = _probe_one(
        vehicle, ticker, client, 'wheels[].abs_enabled',
        lambda pc: wheels_set(pc, 'abs_enabled',
                              not pc.wheels[0].abs_enabled),
        lambda pc: pc.wheels[0].abs_enabled)
    results['max_hand_brake_torque'] = _probe_one(
        vehicle, ticker, client, 'wheels[].max_hand_brake_torque',
        lambda pc: wheels_set(pc, 'max_hand_brake_torque', 6000.0),
        lambda pc: pc.wheels[0].max_hand_brake_torque)
    results['wheel_radius'] = _probe_one(
        vehicle, ticker, client, 'wheels[].wheel_radius',
        lambda pc: wheels_set(pc, 'wheel_radius', pc.wheels[0].wheel_radius * 1.5),
        lambda pc: pc.wheels[0].wheel_radius)

    print("\nPER-WHEEL VIA BATCH COMMAND (different server code path)")
    results['max_brake_torque_batch'] = _probe_one(
        vehicle, ticker, client, 'wheels[].max_brake_torque (batch)',
        lambda pc: wheels_set(pc, 'max_brake_torque', 4000.0),
        lambda pc: pc.wheels[0].max_brake_torque,
        use_batch=True)

    print()
    print("=" * 68)
    top_ok = any(results[k] for k in
                 ('mass', 'drag_coefficient', 'use_sweep_wheel_collision'))
    brake_ok = results['max_brake_torque'] or results['max_brake_torque_batch']
    if brake_ok:
        print("[ OK ] max_brake_torque is writable -- the brake fix is a physics")
        print("       override applied at spawn.")
    elif top_ok:
        print("[FAIL] Top-level physics writes work but PER-WHEEL ones are dropped")
        print("       by this build. max_brake_torque cannot be raised through the")
        print("       Python API, so the brake strength has to come from elsewhere")
        print("       (different vehicle blueprint, or a patched CARLA build).")
    else:
        print("[FAIL] NO physics write of any kind survives on this build. Either")
        print("       the world is not ticking, or apply_physics_control is inert")
        print("       here. Check the tick warnings above before concluding.")
    print("=" * 68)
    return results


# ==============================================================================
# -- physics -------------------------------------------------------------------
# ==============================================================================


def report_physics(vehicle):
    """Print the brake-relevant physics and the deceleration they imply."""
    pc = vehicle.get_physics_control()
    print("Vehicle : %s (id %d)" % (vehicle.type_id, vehicle.id))
    print("Mass    : %.0f kg" % pc.mass)
    print("brake_effect : %.2f   (engine braking off-throttle, NOT the pedal)"
          % pc.brake_effect)
    print()
    print("  wheel  max_brake_torque  handbrake   radius   ABS    TC    by_brake  friction_x")
    total_force = 0.0
    for i, w in enumerate(pc.wheels):
        radius_m = w.wheel_radius / 100.0  # CARLA 0.10 reports wheel_radius in cm
        force = (w.max_brake_torque / radius_m) if (radius_m > 0 and w.affected_by_brake) else 0.0
        total_force += force
        print("  %5d  %16.0f  %9.0f  %6.1fcm  %-5s  %-5s  %-8s  %.2f"
              % (i, w.max_brake_torque, w.max_hand_brake_torque, w.wheel_radius,
                 w.abs_enabled, w.traction_control_enabled, w.affected_by_brake,
                 w.friction_force_multiplier))
    print()

    if pc.mass > 0 and total_force > 0:
        a_torque = total_force / pc.mass
        print("Brake torque implies a peak deceleration of %.1f m/s^2 (%.2f g)"
              % (a_torque, a_torque / G))
        print("  -> if torque-limited, 50 km/h stops in %.1f s / %.0f m"
              % (13.89 / a_torque, 13.89 ** 2 / (2 * a_torque)))
        print("Reference: a real car on dry asphalt does 8-10 m/s^2 (0.8-1.0 g),")
        print("           50 km/h -> ~1.5 s and ~14 m of braking.")
        if a_torque < 6.0:
            print()
            print("[FAIL] %.1f m/s^2 is well under what a car does. The brakes are "
                  "underpowered" % a_torque)
            print("       in physics, independent of the pedal.")
        elif a_torque < 8.0:
            print()
            print("[WARN] %.1f m/s^2 is at the low end -- noticeably softer than a "
                  "real car," % a_torque)
            print("       which is exactly the 'brakes slowly' feeling.")
        else:
            print()
            print("[ OK ] Brake torque is in the normal range; if it still feels")
            print("       slow the limit is tyre grip or the pedal, not torque.")
    if any(w.abs_enabled for w in pc.wheels):
        print()
        print("[NOTE] ABS is enabled. Chaos modulates brake torque to keep slip")
        print("       below threshold, so measured deceleration will be under the")
        print("       torque figure above. Disable it with --set-abs off to see")
        print("       how much it is costing you.")
    return pc


def apply_overrides(vehicle, ticker, args):
    """Runtime-only physics changes, so a candidate fix can be felt immediately.

    The write is read back and verified, because an override that never landed
    looks exactly like an override that made no difference. The readback has to
    come AFTER a tick: apply_physics_control is documented as taking effect "for
    the next tick", so reading immediately always returns the old values.
    """
    if (args.set_brake_torque is None and args.set_abs is None
            and args.set_friction is None):
        return False
    pc = vehicle.get_physics_control()
    wheels = []
    for w in pc.wheels:
        if args.set_brake_torque is not None:
            w.max_brake_torque = args.set_brake_torque
        if args.set_abs is not None:
            w.abs_enabled = (args.set_abs == 'on')
        if args.set_friction is not None:
            w.friction_force_multiplier = args.set_friction
        wheels.append(w)
    pc.wheels = wheels
    vehicle.apply_physics_control(pc)
    print("[INFO] Requested override: brake_torque=%s abs=%s friction=%s "
          "(this process only, reverts when the actor is respawned)"
          % (args.set_brake_torque, args.set_abs, args.set_friction))

    ticked = all(ticker.step() is not None for _ in range(3))
    if not ticked:
        print("[WARN] The world did not tick, so the override cannot have been "
              "applied yet and the readback below is stale. Start the clock "
              "owner (or pass --self-tick) and run again.")
    back = vehicle.get_physics_control()
    landed = True
    for i, w in enumerate(back.wheels):
        if (args.set_brake_torque is not None
                and abs(w.max_brake_torque - args.set_brake_torque) > 1.0):
            print("[FAIL] wheel %d max_brake_torque stayed at %.0f -- the write "
                  "was ignored." % (i, w.max_brake_torque))
            landed = False
        if args.set_abs is not None and w.abs_enabled != (args.set_abs == 'on'):
            print("[FAIL] wheel %d abs_enabled stayed at %s -- the write was "
                  "ignored." % (i, w.abs_enabled))
            landed = False
        if (args.set_friction is not None
                and abs(w.friction_force_multiplier - args.set_friction) > 0.01):
            print("[FAIL] wheel %d friction_force_multiplier stayed at %.2f -- the "
                  "write was ignored." % (i, w.friction_force_multiplier))
            landed = False
    if landed:
        print("[ OK ] Override verified on all %d wheels." % len(back.wheels))
    return landed


# ==============================================================================
# -- brake test ----------------------------------------------------------------
# ==============================================================================


def run_brake_test(world, vehicle, ticker, args):
    """Accelerate to args.speed, brake at args.level, measure what happens."""
    target_ms = args.speed / 3.6

    collided = []
    bp = world.get_blueprint_library().find('sensor.other.collision')
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    sensor.listen(lambda e: collided.append(e))

    try:
        print("Accelerating to %.0f km/h ..." % args.speed)
        deadline = time.monotonic() + args.accel_timeout
        while True:
            if ticker.step() is None:
                print("[FAIL] The world is not ticking. Under --sync the clock owner")
                print("       (src/drive/fixed_npc_traffic.py --sync) must be running,")
                print("       or pass --self-tick if nothing else owns the clock.")
                return None
            vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0, steer=0.0))
            v = speed_kmh(vehicle) / 3.6
            if v >= target_ms:
                break
            if time.monotonic() > deadline:
                print("[WARN] Only reached %.1f km/h in %.0f s; braking from there."
                      % (v * 3.6, args.accel_timeout))
                break
            if collided:
                print("[FAIL] Collided while accelerating -- no usable measurement. "
                      "Try another --spawn-index.")
                return None

        snap = world.get_snapshot()
        t0 = snap.timestamp.elapsed_seconds
        p0 = vehicle.get_location()
        v0 = speed_kmh(vehicle) / 3.6
        print("Braking from %.1f km/h at brake=%.2f ...\n" % (v0 * 3.6, args.level))

        trace = [(0.0, v0, 0.0)]
        t_first_drop = None
        deadline = time.monotonic() + args.brake_timeout
        while True:
            snap = ticker.step()
            if snap is None:
                print("[FAIL] World stopped ticking mid-measurement.")
                return None
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, brake=args.level, steer=0.0, hand_brake=False))
            t = snap.timestamp.elapsed_seconds - t0
            v = speed_kmh(vehicle) / 3.6
            loc = vehicle.get_location()
            dist = math.sqrt((loc.x - p0.x) ** 2 + (loc.y - p0.y) ** 2)
            trace.append((t, v, dist))
            if t_first_drop is None and v < v0 * 0.98:
                t_first_drop = t
            if v <= 0.15:
                break
            if time.monotonic() > deadline:
                print("[WARN] Still moving at %.1f km/h after %.0f s of full brake."
                      % (v * 3.6, args.brake_timeout))
                break
            if collided:
                print("[FAIL] Collided while braking -- measurement discarded. "
                      "Try another --spawn-index.")
                return None
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    finally:
        sensor.stop()
        sensor.destroy()

    return _report_trace(trace, t_first_drop, v0, args)


def _report_trace(trace, t_first_drop, v0, args):
    t_end, v_end, d_end = trace[-1]
    mean_a = (v0 - v_end) / t_end if t_end > 0 else 0.0

    peak_a = 0.0
    for (ta, va, _), (tb, vb, _) in zip(trace, trace[1:]):
        dt = tb - ta
        if dt > 1e-4:
            peak_a = max(peak_a, (va - vb) / dt)

    if args.trace:
        print("      t (s)   speed (km/h)   distance (m)")
        for t, v, d in trace:
            print("   %7.3f   %12.1f   %12.2f" % (t, v * 3.6, d))
        print()

    print("=" * 68)
    print("Braking from %.1f km/h at brake=%.2f" % (v0 * 3.6, args.level))
    print("  time to stop        : %.2f s" % t_end)
    print("  distance to stop    : %.1f m" % d_end)
    print("  mean deceleration   : %.2f m/s^2  (%.2f g)" % (mean_a, mean_a / G))
    print("  peak deceleration   : %.2f m/s^2  (%.2f g)" % (peak_a, peak_a / G))
    if t_first_drop is not None:
        print("  lag before slowing  : %.3f s" % t_first_drop)
    print()
    ref_t = v0 / 9.0
    print("  a real car (0.9 g)  : %.2f s, %.1f m" % (ref_t, v0 * v0 / (2 * 9.0)))
    print("=" * 68)

    if mean_a < 5.0:
        print("[FAIL] %.2f m/s^2 is far below a real car. This is the 'brakes"
              "\n       slowly' complaint, and it is in the physics, not the pedal."
              % mean_a)
    elif mean_a < 7.0:
        print("[WARN] %.2f m/s^2 is soft -- perceptibly slower than a real car."
              % mean_a)
    else:
        print("[ OK ] %.2f m/s^2 is in the normal range for a car." % mean_a)
    if t_first_drop is not None and t_first_drop > 0.25:
        print("[WARN] %.0f ms passed before the car started slowing. That delay is"
              % (t_first_drop * 1000))
        print("       felt as unresponsive brakes regardless of how hard they bite.")
    return mean_a


def run_carla_modes(args, do_physics, do_test, do_probe=False):
    try:
        client = connect(args)
    except Exception as e:
        print("[SKIP] CARLA is not reachable at %s:%d (%s). Start CarlaUnreal.exe "
              "to run the physics and test checks." % (args.host, args.port, e))
        return None

    world = client.get_world()
    settings = world.get_settings()
    print("Server %s | map %s | sync=%s | fixed_delta=%s"
          % (client.get_server_version(), world.get_map().name,
             settings.synchronous_mode, settings.fixed_delta_seconds))
    print()

    ticker = Ticker(world, self_tick=args.self_tick)
    spawned = None
    try:
        vehicle = find_hero(world) if args.use_hero else None
        if vehicle is None:
            if args.use_hero:
                print("[WARN] No hero vehicle found; spawning a throwaway one.")
            spawn_points = world.get_map().get_spawn_points()
            if not spawn_points:
                print("[FAIL] Map has no spawn points.")
                return None
            bp = world.get_blueprint_library().find(VEHICLE_BP)
            idx = args.spawn_index % len(spawn_points)
            vehicle = world.try_spawn_actor(bp, spawn_points[idx])
            if vehicle is None:
                print("[FAIL] Spawn point %d is occupied; try another --spawn-index "
                      "(0..%d)." % (idx, len(spawn_points) - 1))
                return None
            spawned = vehicle
            vehicle.set_autopilot(False)
            print("[INFO] Spawned %s at spawn point %d." % (VEHICLE_BP, idx))
            for _ in range(20):  # let it settle onto its suspension
                ticker.step()
                vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        elif args.set_brake_torque is not None or args.set_abs is not None:
            print("[WARN] Overriding physics on the LIVE hero vehicle.")

        if do_physics:
            print("-" * 68)
            report_physics(vehicle)
            print()

        if do_probe:
            if spawned is None:
                print("[SKIP] Refusing to probe writes on the LIVE hero vehicle -- "
                      "it would perturb the participant's car. Drop --use-hero.")
                return None
            print("-" * 68)
            return probe_writes(vehicle, ticker, client)

        apply_overrides(vehicle, ticker, args)

        if do_test:
            if args.use_hero and spawned is None:
                print("[SKIP] Refusing to drive the hero vehicle for the brake test -- "
                      "it is the participant's car. Drop --use-hero to test on a "
                      "throwaway spawn.")
                return None
            print("-" * 68)
            return run_brake_test(world, vehicle, ticker, args)
    finally:
        if spawned is not None:
            spawned.destroy()
            print("\n[INFO] Test vehicle destroyed.")
    return None


# ==============================================================================
# -- main ----------------------------------------------------------------------
# ==============================================================================


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', nargs='?', default='all',
                   choices=['all', 'pedal', 'physics', 'test', 'probe'],
                   help='which check to run (default: all)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--timeout', type=float, default=10.0)
    p.add_argument('--seconds', type=float, default=30.0,
                   help='pedal: how long to sample before reporting (default 30)')
    p.add_argument('--speed', type=float, default=50.0,
                   help='test: speed in km/h to brake from (default 50)')
    p.add_argument('--level', type=float, default=1.0,
                   help='test: brake value to apply, 0..1 (default 1.0). Run it at '
                        '0.25/0.5/0.75/1.0 to check the pedal response is linear')
    p.add_argument('--spawn-index', type=int, default=0,
                   help='test: which spawn point to run from (needs clear road ahead)')
    p.add_argument('--use-hero', action='store_true',
                   help='inspect the running session vehicle instead of spawning one')
    p.add_argument('--self-tick', action='store_true',
                   help='tick the world from here. ONLY when nothing else owns the '
                        'clock -- never while fixed_npc_traffic.py --sync is running')
    p.add_argument('--trace', action='store_true',
                   help='test: print the full speed/distance trace')
    p.add_argument('--accel-timeout', type=float, default=30.0)
    p.add_argument('--brake-timeout', type=float, default=20.0)
    p.add_argument('--set-brake-torque', type=float, default=None,
                   help='override max_brake_torque on every wheel before testing')
    p.add_argument('--set-abs', choices=['on', 'off'], default=None,
                   help='override abs_enabled on every wheel before testing')
    p.add_argument('--set-friction', type=float, default=None,
                   help='override friction_force_multiplier on every wheel '
                        '(the tyre-grip ceiling) before testing')
    args = p.parse_args()

    if args.mode in ('all', 'pedal'):
        print("=" * 68)
        print("PEDAL  -- does a full press reach brake = 1.0?")
        print("=" * 68)
        run_pedal(args)
        print()

    if args.mode in ('all', 'physics', 'test', 'probe'):
        print("=" * 68)
        print("CARLA  -- what the simulator does with the brake input")
        print("=" * 68)
        run_carla_modes(args,
                        do_physics=args.mode in ('all', 'physics', 'test', 'probe'),
                        do_test=args.mode in ('all', 'test'),
                        do_probe=args.mode == 'probe')


if __name__ == '__main__':
    main()
