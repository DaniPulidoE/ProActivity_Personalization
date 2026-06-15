import carla
import time
import random

def get_env_info(world, ego_vehicle, collision_sensor=None):
    weather = world.get_weather()
    vehicles = world.get_actors().filter('vehicle.*')
    walkers = world.get_actors().filter('walker.*')

    # Compute vehicle speed (km/h)
    velocity = ego_vehicle.get_velocity()
    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

    # Compute vehicle acceleration (m/s²)
    acceleration = ego_vehicle.get_acceleration()
    acc = (acceleration.x**2 + acceleration.y**2 + acceleration.z**2) ** 0.5

    weather_type = "Clear"
    if weather.precipitation > 60:
        weather_type = "Rainy"
    elif weather.fog_density > 30:
        weather_type = "Foggy"
    elif weather.cloudiness > 60:
        weather_type = "Cloudy"

    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    traffic_light_state = "None"
    min_dist = float('inf')
    ego_loc = ego_vehicle.get_location()
    for t in traffic_lights:
        dist = ego_loc.distance(t.get_location())
        if dist < min_dist:
            min_dist = dist
            traffic_light_state = t.state

    lane_info = ego_vehicle.get_transform().location

    collision_event = "None"
    if collision_sensor and getattr(collision_sensor, 'triggered', False):
        collision_event = f"Collided! {collision_sensor.last_event}"

    info = {
        "WeatherType": weather_type,
        "Cloudiness": weather.cloudiness,
        "WindIntensity": weather.wind_intensity,
        "Wetness": weather.wetness,
        "TrafficDensity": len(vehicles),
        "PedestrianDensity": len(walkers),
        "TrafficLightState": str(traffic_light_state),
        "LaneInformation": lane_info,
        "CollisionEvent": collision_event,
        "Speed": speed,        # km/h
        "Acceleration": acc    # m/s^2
    }
    return info

def draw_info_panel(world, ego_vehicle, info, context="low"):
    right_offset = carla.Location(y=2.8, z=2.3)
    right_base = ego_vehicle.get_transform().location + right_offset
    text_color = carla.Color(0, 255, 255)

    panel_text = (
        f"WeatherType: {info['WeatherType']}\n"
        f"Cloudiness: {info['Cloudiness']:.1f}\n"
        f"WindIntensity: {info['WindIntensity']:.1f}\n"
        f"Wetness: {info['Wetness']:.1f}\n"
        f"TrafficDensity: {info['TrafficDensity']}\n"
        f"PedestrianDensity: {info['PedestrianDensity']}\n"
        f"TrafficLightState: {info['TrafficLightState']}\n"
        f"LaneInfo: ({info['LaneInformation'].x:.1f},{info['LaneInformation'].y:.1f})\n"
        f"Speed: {info['Speed']:.1f} km/h" + ("  <<LOW SPEED>>\n" if info['Speed'] < 10 else "\n") +
        f"Acceleration: {info['Acceleration']:.2f} m/s²\n"
        f"CollisionEvent: {info['CollisionEvent']}"
    )
    world.debug.draw_string(right_base, panel_text, life_time=0.1, color=text_color, persistent_lines=False)

    left_offset = carla.Location(y=-2.8, z=2.3)
    left_base = ego_vehicle.get_transform().location + left_offset
    if context == "low":
        left_text = (
            "Traffic Density: LOW\n"
            "Driver Workload: LOW\n"
            "ProVoice LoA (Prediction): 1-2 (Advisory)\n"
        )
    else:
        left_text = (
            "Traffic Density: HIGH\n"
            "Event: Drinking\n"
            "Driver Workload: HIGH\n"
            "ProVoice LoA (Prediction): 3-4 (Autonomous)\n"
            "Driver slows down and reacts...\n"
        )
    world.debug.draw_string(left_base, left_text, life_time=0.1, color=carla.Color(255, 128, 0), persistent_lines=False)

def spawn_ego_vehicle(world, blueprint_library, spawn_point):
    vehicle_bp = blueprint_library.find('vehicle.dodge.charger')
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Spawned ego vehicle:", ego_vehicle)
    return ego_vehicle

def spawn_traffic(world, blueprint_library, spawn_points, count):
    vehicle_bps = blueprint_library.filter('vehicle.*')
    traffic_vehicles = []
    for i in range(count):
        bp = random.choice(vehicle_bps)
        pt = random.choice(spawn_points)
        try:
            v = world.try_spawn_actor(bp, pt)
            if v:
                v.set_autopilot(True)
                traffic_vehicles.append(v)
        except:
            continue
    print(f"Spawned {len(traffic_vehicles)} traffic vehicles")
    return traffic_vehicles

def attach_collision_sensor(world, ego_vehicle):
    blueprint = world.get_blueprint_library().find('sensor.other.collision')
    sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=ego_vehicle)
    sensor.triggered = False
    sensor.last_event = ""
    def on_collision(event):
        sensor.triggered = True
        sensor.last_event = f"{event.other_actor.type_id}"
    sensor.listen(on_collision)
    return sensor

def set_camera_once(world, ego_vehicle):
    spectator = world.get_spectator()
    transform = ego_vehicle.get_transform()
    loc = transform.location + carla.Location(x=-8, y=0, z=6)
    rot = carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
    spectator.set_transform(carla.Transform(loc, rot))

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world("Mine_01")
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    ego_vehicle = spawn_ego_vehicle(world, blueprint_library, random.choice(spawn_points))
    set_camera_once(world, ego_vehicle)
    traffic = spawn_traffic(world, blueprint_library, spawn_points, count=2)
    ego_vehicle.set_autopilot(True)

    collision_sensor = attach_collision_sensor(world, ego_vehicle)

    context = "low"
    phase = 0
    t0 = time.time()
    manual_control = False

    try:
        while True:
            info = get_env_info(world, ego_vehicle, collision_sensor)
            draw_info_panel(world, ego_vehicle, info, context=context)

            # Auto decelerate in high-load zone
            if manual_control:
                control = carla.VehicleControl(throttle=0.18, steer=0.0, brake=0.0)
                ego_vehicle.apply_control(control)
            time.sleep(0.08)

            # Switch to high-load scenario ~20 seconds in, spawn more traffic and slow down
            if phase == 0 and time.time()-t0 > 20:
                print("Switching to high-traffic scenario and slowing down...")
                for v in traffic:
                    v.destroy()
                traffic = spawn_traffic(world, blueprint_library, spawn_points, count=5)
                context = "high"
                phase = 1
                ego_vehicle.set_autopilot(False)   # Switch to manual slow mode
                manual_control = True

    finally:
        ego_vehicle.destroy()
        for v in traffic:
            v.destroy()
        collision_sensor.destroy()

if __name__ == "__main__":
    main()
