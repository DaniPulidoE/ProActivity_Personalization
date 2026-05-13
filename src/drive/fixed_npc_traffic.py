import carla
import random
import time

# =========================
# CONFIG
# =========================

CARLA_HOST = "localhost"
CARLA_PORT = 2000

TM_PORT = 9000

SEED = 42

NUM_VEHICLES = 50

SYNC_MODE = True

FIXED_DELTA_SECONDS = 0.05

# fixed npc vehicle config
# (spawn_point_index, blueprint_id)

VEHICLE_CONFIGS = [
    (0,  "vehicle.tesla.model3"),
    (5,  "vehicle.audi.tt"),
    (10, "vehicle.bmw.grandtourer"),
    (15, "vehicle.lincoln.mkz_2020"),
    (20, "vehicle.mercedes.coupe"),
    (25, "vehicle.nissan.patrol"),
    (30, "vehicle.toyota.prius"),
    (35, "vehicle.ford.mustang"),
    (40, "vehicle.chevrolet.impala"),
    (45, "vehicle.dodge.charger"),
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

tm.set_hybrid_physics_mode(True)

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

    # 恢复异步模式
    settings = world.get_settings()

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None

    world.apply_settings(settings)

    print("Done.")