"""
Windows console wheel+pedal controller for CARLA (with random interruptions).

Controls:
  Wheel        : steer
  Throttle     : accelerate
  Brake pedal  : brake
  Button 0     : acknowledge random interruption and continue
  R (keyboard) : reset to nearest spawn point
  Q or ESC     : quit

Notes:
- Uses CARLA simulator window only. Single controller, no HUD.
- Fixed vehicle: 'vehicle.dodge.charger_2020'. Map: "default" or "Mine_01".
- Driver-seat view with camera smoothing.
- Uses synchronous mode to reduce jitter between physics and camera.
"""

import os
import sys
import time
import math
import ctypes
import random

import pygame  # NEW: for wheel + pedals

try:
    import carla
except ImportError:
    print("[ERROR] Cannot import CARLA. Put the PythonAPI egg in PYTHONPATH.")
    sys.exit(1)

# --------------
# Configuration
# --------------
MAP_SELECTION = "default"  # "default" or "Mine_01"
TARGET_FPS = 30  # control loop pacing

# Camera: driver seat offset (vehicle local: x fwd, y right, z up)
DRIVER_OFFSET_X = 0.25
DRIVER_OFFSET_Y = -0.33
DRIVER_OFFSET_Z = 1.21
CAMERA_PITCH = -2.0

# Camera smoothing (0 = off, 1 = very slow). Keep small for responsiveness.
CAMERA_SMOOTH_POS = 0.55
CAMERA_SMOOTH_YAW = 0.85

# Use synchronous mode to avoid jitter between physics and camera updates.
ENABLE_SYNC_MODE = True

# VK codes for GetAsyncKeyState (keyboard utility controls)

VK_SHIFT = 0x10
VK_SPACE = 0x20
VK_ESCAPE = 0x1B
# ---------- WHEEL CONFIG (Logitech G29/G920 style) ----------
WHEEL_DEADZONE = 0.05
PEDAL_DEADZONE = 0.02

AXIS_STEER = 0  # Wheel (Left/Right)
AXIS_THROTTLE = 1  # Accelerator Pedal (1.0 = no press, -1.0 = full)
AXIS_BRAKE = 2  # Brake Pedal      (1.0 = no press, -1.0 = full)

BUTTON_CONTINUE = 0  # Wheel button index to acknowledge interruption

# --- INTERRUPTION CONFIG ---
INTERRUPT_MIN_DELAY = 10.0  # seconds
INTERRUPT_MAX_DELAY = 25.0  # seconds


# -----------
# Utilities
# -----------


def is_windows():
    return os.name == "nt"


def key_down(vk):
    """Check a Windows virtual key state (for ESC/Q/R only)."""
    return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def load_world(client, selection):
    s = (selection or "").strip().lower()
    if s == "mine_01":
        print("[INFO] Loading map: Mine_01 ...")
        return client.load_world("Mine_01")
    else:
        world = client.get_world()
        print("[INFO] Using current map: " + world.get_map().name)
        return world


def get_spawn_points(world):
    return world.get_map().get_spawn_points()


def spawn_vehicle(world, spawn_points):
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find("vehicle.dodge.charger_2020")
    if bp is None:
        raise RuntimeError("Blueprint 'vehicle.dodge.charger_2020' not found")
    if bp.has_attribute("color"):
        bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
    if not spawn_points:
        raise RuntimeError("No spawn points available")
    
    vehicle = world.spawn_actor(bp, spawn_points[0])
    vehicle.set_autopilot(False)
    print("[INFO] Spawned 'vehicle.dodge.charger_2020' (id={})".format(vehicle.id))

    # === Persist the vehicle id to a file for other processes to read ===
    try:
        id_file = os.path.join(os.getcwd(), "vehicle_id.txt")
        tmp_file = id_file + ".tmp"
        with open(tmp_file, "w") as f:
            f.write(str(vehicle.id))
        os.replace(tmp_file, id_file)
        print(f"[INFO] Written vehicle id to {id_file}")
    except Exception as e:
        print("[WARN] Failed to write vehicle id file:", e)

    return vehicle


def nearest_spawn_index(vehicle, spawn_points):
    vloc = vehicle.get_location()
    best_i = 0
    best_d2 = 1e30
    for i in range(len(spawn_points)):
        t = spawn_points[i]
        dx = t.location.x - vloc.x
        dy = t.location.y - vloc.y
        dz = t.location.z - vloc.z
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def reset_to_nearest_spawn(vehicle, spawn_points):
    try:
        i = nearest_spawn_index(vehicle, spawn_points)
        target = spawn_points[i]
        vehicle.set_transform(target)
        # stop drift after teleport
        try:
            vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except Exception:
            pass
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, reverse=False)
        )
        print("[INFO] Reset to nearest spawn (#" + str(i) + ")")
    except Exception as e:
        print("[WARN] Reset failed: " + str(e))


# Camera smoothing state
CAM_SMX = None
CAM_SMY = None
CAM_SMZ = None
CAM_SMYAW = None


def update_spectator(world, vehicle):
    global CAM_SMX, CAM_SMY, CAM_SMZ, CAM_SMYAW

    sp = world.get_spectator()
    tf = vehicle.get_transform()
    loc = tf.location
    rot = tf.rotation
    yaw = rot.yaw * math.pi / 180.0
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    tx = loc.x + DRIVER_OFFSET_X * cos_y - DRIVER_OFFSET_Y * sin_y
    ty = loc.y + DRIVER_OFFSET_X * sin_y + DRIVER_OFFSET_Y * cos_y
    tz = loc.z + DRIVER_OFFSET_Z
    tyaw = rot.yaw

    if CAM_SMX is None:
        CAM_SMX = tx
        CAM_SMY = ty
        CAM_SMZ = tz
        CAM_SMYAW = tyaw
    else:
        # small low-pass to reduce perceived head bob
        CAM_SMX = CAM_SMX + CAMERA_SMOOTH_POS * (tx - CAM_SMX)
        CAM_SMY = CAM_SMY + CAMERA_SMOOTH_POS * (ty - CAM_SMY)
        CAM_SMZ = CAM_SMZ + CAMERA_SMOOTH_POS * (tz - CAM_SMZ)
        dyaw = tyaw - CAM_SMYAW
        CAM_SMYAW = CAM_SMYAW + CAMERA_SMOOTH_YAW * dyaw

    cam_loc = carla.Location(x=CAM_SMX, y=CAM_SMY, z=CAM_SMZ)
    cam_rot = carla.Rotation(pitch=CAMERA_PITCH, yaw=CAM_SMYAW, roll=0.0)
    sp.set_transform(carla.Transform(cam_loc, cam_rot))


# --- Wheel init & interruption helpers ---


def init_wheel():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("[WARN] No wheel detected. Falling back to keyboard control.")
        return None  
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[INFO] Detected wheel: {js.get_name()}")
    return js


def next_interrupt_time(now: float) -> float:
    """Pick a random time in the future for the next interruption."""
    delay = random.uniform(INTERRUPT_MIN_DELAY, INTERRUPT_MAX_DELAY)
    return now + delay


# -----------
# Main
# -----------


def main():
    if not is_windows():
        print("[ERROR] Windows only. This script uses GetAsyncKeyState.")
        return

    print("\n=== CARLA Wheel+Pedal Console Controller (with interruptions) ===")
    print("Wheel / pedals control driving.")
    print("Button", BUTTON_CONTINUE, ": acknowledge random interruption and continue.")
    print("Keyboard:")
    print("  R         : reset to nearest spawn")
    print("  Q or ESC  : quit")
    print(
        "Notes: Synchronous mode and camera smoothing are enabled to reduce view shake."
    )
    print("Map selection: " + MAP_SELECTION)

    # Init wheel (or None)
    wheel = init_wheel()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = load_world(client, MAP_SELECTION)
    original_settings = world.get_settings()

    if ENABLE_SYNC_MODE:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(TARGET_FPS)
        world.apply_settings(settings)

    spawn_points = get_spawn_points(world)
    vehicle = None

    # Interruption state
    interrupted = False
    now = time.time()
    next_int_at = next_interrupt_time(now)

    id_file = os.path.join(os.getcwd(), "vehicle_id.txt")

    try:
        vehicle = spawn_vehicle(world, spawn_points)

        # init camera smoothing state at first pose
        update_spectator(world, vehicle)

        r_prev = False

        while True:
            # Pump pygame events (required for joystick updates)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Keyboard utility (quit/reset)
            if key_down(VK_ESCAPE) or key_down(ord("Q")):
                break

            now = time.time()

            # Trigger interruption when time is up
            if (not interrupted) and (now >= next_int_at):
                interrupted = True
                print(
                    "=== INTERRUPTION! Press wheel button",
                    BUTTON_CONTINUE,
                    "to continue driving. ===",
                )

            # While interrupted: full brake, ignore wheel axes
            if interrupted:
                # check physical wheel
                if wheel is not None:
                    for event in events:
                        if event.type == pygame.JOYBUTTONDOWN and event.button == BUTTON_CONTINUE:
                            interrupted = False
                            next_int_at = next_interrupt_time(now)
                            print("Continuing simulation...")
                            break
                else:
                    if key_down(ord('E')):
                        interrupted = False
                        next_int_at = next_interrupt_time(now)
                        print("Continuing simulation via keyboard E...")

                # stay full brake
                ctrl = carla.VehicleControl()
                ctrl.throttle = 0.0
                ctrl.brake = 1.0
                ctrl.steer = 0.0
                ctrl.reverse = False
                vehicle.apply_control(ctrl)

                update_spectator(world, vehicle)

                if ENABLE_SYNC_MODE:
                    world.tick()
                else:
                    world.wait_for_tick()
                    time.sleep(1.0 / float(TARGET_FPS))
                continue  # skip normal wheel control this frame

            # --- NORMAL WHEEL/ PEDAL DRIVING ---

            # Reset to nearest spawn with 'R' (keyboard)
            r_now = key_down(ord("R"))
            if r_now and not r_prev:
                reset_to_nearest_spawn(vehicle, spawn_points)
            r_prev = r_now

            if wheel is not None:
                # Steering from wheel axis
                steer = wheel.get_axis(AXIS_STEER)
                if abs(steer) < WHEEL_DEADZONE:
                    steer = 0.0
                steer = clamp(steer, -1.0, 1.0)

                # Throttle: Logitech style 1.0 (not pressed) to -1.0 (fully pressed)
                raw_throttle = wheel.get_axis(AXIS_THROTTLE)
                throttle = 1.0 - (raw_throttle + 1.0) / 2.0  # map to 0..1
                if throttle < PEDAL_DEADZONE:
                    throttle = 0.0
                throttle = clamp(throttle, 0.0, 1.0)

                # Brake pedal
                raw_brake = wheel.get_axis(AXIS_BRAKE)
                brake = 1.0 - (raw_brake + 1.0) / 2.0  # map to 0..1
                if brake < PEDAL_DEADZONE:
                    brake = 0.0
                brake = clamp(brake, 0.0, 1.0)
                
                forward = True   # no reverse button assigned for wheel
                reverse = False
                
            else:
                # === Full WASD fallback (consistent with drive.py) ===
                # Use GetAsyncKeyState instead of pygame.key to keep behavior consistent with drive.py
                forward = key_down(ord('W'))
                reverse = key_down(ord('S'))
                left = key_down(ord('A'))
                right = key_down(ord('D'))
                brake_key = key_down(VK_SPACE)
                boost = key_down(VK_SHIFT)
                acknowledge = key_down(ord('E'))

                if not hasattr(main, "_fallback_steer"):
                    main._fallback_steer = 0.0

                if left and not right:
                    main._fallback_steer -= 0.03     # STEER_STEP
                elif right and not left:
                    main._fallback_steer += 0.03
                else:
                    main._fallback_steer *= (1.0 - 0.18)  # STEER_DECAY
                    if abs(main._fallback_steer) < 1e-3:
                        main._fallback_steer = 0.0

                steer = clamp(main._fallback_steer, -1.0, 1.0)

                if brake_key:
                    throttle = 0.0
                    brake = 1.0
                else:
                    throttle = 0.0
                    brake = 0.0

                    if forward and not reverse:
                        throttle = 0.55    # BASE_THROTTLE
                    elif reverse and not forward:
                        throttle = 0.55

                    # boost
                    if boost and throttle > 0.0:
                        throttle *= 1.45   # BOOST_MULTIPLIER

                    throttle = clamp(throttle, 0.0, 1.0)

            ctrl = carla.VehicleControl()
            ctrl.steer = steer
            ctrl.throttle = throttle
            ctrl.brake = brake
            ctrl.reverse = reverse and not forward


            vehicle.apply_control(ctrl)

            update_spectator(world, vehicle)

            if ENABLE_SYNC_MODE:
                world.tick()
            else:
                world.wait_for_tick()
                time.sleep(1.0 / float(TARGET_FPS))

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        # restore settings
        if ENABLE_SYNC_MODE:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass

        # delete vehicle_id 
        try:
            if os.path.exists(id_file):
                os.remove(id_file)
                print(f"[INFO] Removed vehicle id file: {id_file}")
        except Exception as e:
            print("[WARN] Could not remove vehicle id file:", e)

        # destory vehicle
        if vehicle is not None:
            try:
                vehicle.set_autopilot(False)
            except Exception:
                pass
            try:
                vehicle.destroy()
                print("[INFO] Vehicle destroyed.")
            except Exception as e:
                print("[WARN] Could not destroy vehicle: " + str(e))

        pygame.quit()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
