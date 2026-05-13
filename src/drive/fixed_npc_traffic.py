import carla
import random
import time


class FixedTrafficManager:
    def __init__(
        self,
        host="localhost",
        port=2000,
        tm_port=9000,
        seed=42,
        sync_mode=True,
        fixed_delta_seconds=0.05,
        vehicle_configs=None
    ):
        self.host = host
        self.port = port
        self.tm_port = tm_port
        self.seed = seed
        self.sync_mode = sync_mode
        self.fixed_delta_seconds = fixed_delta_seconds

        self.client = None
        self.world = None
        self.tm = None

        self.blueprint_library = None
        self.spawn_points = None

        self.vehicles = []

        # default configs
        self.vehicle_configs = vehicle_configs or []

    # =========================
    # CONNECT
    # =========================
    def connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        random.seed(self.seed)

    # =========================
    # WORLD SETUP
    # =========================
    def setup_world(self):
        settings = self.world.get_settings()

        settings.synchronous_mode = self.sync_mode

        if self.sync_mode:
            settings.fixed_delta_seconds = self.fixed_delta_seconds
        else:
            settings.fixed_delta_seconds = None

        self.world.apply_settings(settings)

        # traffic manager
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(self.sync_mode)
        self.tm.set_random_device_seed(self.seed)
        self.tm.set_hybrid_physics_mode(True)

    # =========================
    # SPAWN VEHICLES
    # =========================
    def spawn_vehicles(self):
        print("Destroying old vehicles...")

        actors = self.world.get_actors()
        for actor in actors.filter("vehicle.*"):
            try:
                actor.destroy()
            except:
                pass

        time.sleep(1)

        print("Spawning fixed NPC vehicles...")

        for spawn_index, blueprint_id in self.vehicle_configs:

            if spawn_index >= len(self.spawn_points):
                print(f"Spawn point {spawn_index} not available")
                continue

            try:
                bp = self.blueprint_library.find(blueprint_id)
                transform = self.spawn_points[spawn_index]

                vehicle = self.world.try_spawn_actor(bp, transform)

                if vehicle is None:
                    print(f"Failed spawn at {spawn_index}")
                    continue

                # autopilot
                vehicle.set_autopilot(True, self.tm_port)

                # traffic manager rules
                self.tm.auto_lane_change(vehicle, True)

                self.tm.distance_to_leading_vehicle(vehicle, 5.0)

                self.tm.vehicle_percentage_speed_difference(
                    vehicle,
                    random.uniform(-10, 10)
                )

                self.vehicles.append(vehicle)

                print(f"Spawned {vehicle.type_id} at {spawn_index}")

            except Exception as e:
                print("Spawn error:", e)

    # =========================
    # RUN LOOP
    # =========================
    def run(self, tick_callback=None):
        print("NPC traffic running...")

        try:
            while True:

                if self.sync_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()

                if tick_callback:
                    tick_callback(self.world, self.vehicles)

        except KeyboardInterrupt:
            print("Stopping traffic...")
            self.cleanup()

    # =========================
    # CLEANUP
    # =========================
    def cleanup(self):
        print("Cleaning up vehicles...")

        for v in self.vehicles:
            try:
                v.destroy()
            except:
                pass

        self.vehicles.clear()

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        print("Done.")