import carla
from agents.navigation.basic_agent import BasicAgent
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.route_manipulation import interpolate_trajectory


class AutopilotAgent(AutonomousAgent):

    def __init__(self, role_name: str, carla_host, carla_port, debug=False, opt_dict={}):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None
        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None
        self._carla_host = carla_host
        self._carla_port = carla_port
        self._client = carla.Client(self._carla_host, self._carla_port)
        self.role_name = role_name
        self._agent: BasicAgent = None
        self._plan = None
        self._debug = debug
        self._opt_dict = opt_dict

    def setup(self, path_to_conf_file, route=None, trajectory=None):
        actors = CarlaDataProvider.get_world().get_actors()
        vehicle = [actor for actor in actors if actor.attributes.get('role_name') == self.role_name][0]

        if vehicle:
            self._agent = BasicAgent(vehicle, target_speed=50, opt_dict=self._opt_dict)
        world = CarlaDataProvider.get_world()
        if self._agent and route:
            plan = []
            map = CarlaDataProvider.get_map()
            for loc, option in route:
                wp = (map.get_waypoint(loc), option)
                plan.append(wp)
                if self._debug:
                    world.debug.draw_point(loc, size=0.1, color=carla.Color(0, 255, 0), life_time=120.0)
            self._agent.set_global_plan(plan)

        elif self._agent and trajectory:
            plan = []
            waypoints = []
            for item in trajectory:
                if isinstance(item, tuple):
                    wp, _ = item
                else:
                    wp = item

                if self._debug:
                    world.debug.draw_point(wp, size=0.2, color=carla.Color(255, 0, 0), life_time=120.0)

                waypoints.append(wp)

            _, route = interpolate_trajectory(waypoints)
            map = CarlaDataProvider.get_map()
            for tf, option in route:
                wp = (map.get_waypoint(tf.location), option)
                plan.append(wp)
                if self._debug:
                    world.debug.draw_point(tf.location, size=0.1, color=carla.Color(0, 255, 0), life_time=120.0)

            self._agent.set_global_plan(plan)


    def sensors(self):
        sensors = [
            {
                "type": "sensor.camera.rgb",
                "x": 0.7,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "Center",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.7,
                "y": -0.4,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -45.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "Left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.7,
                "y": 0.4,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 45.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "Right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 0.7,
                "y": -0.4,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -45.0,
                "id": "LIDAR",
            },
            {"type": "sensor.other.gnss", "x": 0.7, "y": -0.4, "z": 1.60, "id": "GPS"},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        control = self._agent.run_step()
        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        if self._agent:
            self._agent.set_global_plan(global_plan_world_coord)
