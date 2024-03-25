import os
from xml.etree import ElementTree

import carla
import gymnasium
import numpy as np
import srunner
from pettingzoo.utils.env import AgentID, ObsType
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, \
    ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.route_scenario import RouteScenario

import mats_gym
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.scenarios.actor_configuration import ActorConfiguration

class RouteScenarioEnv(BaseScenarioEnvWrapper):

    def __init__(
            self,
            route_file: str,
            actor_configuration: ActorConfiguration,
            reset_progress_threshold: float = None,
            scenario_runner_path: str = None,
            debug_mode: int = 0,
            **kwargs
    ):
        if scenario_runner_path is None:
            scenario_runner_path = f"{os.path.dirname(srunner.__file__)}/../"
        os.environ["SCENARIO_RUNNER_ROOT"] = scenario_runner_path
        configs = self._parse_routes_file(route_filename=route_file)
        for config in configs:
            config.ego_vehicles = [actor_configuration]
        self._ego_role_name = actor_configuration.rolename or "hero"
        self._current_route = 0
        self._debug_mode = debug_mode
        self._configs = configs
        self._progress = {self._ego_role_name: 0.0}
        self._info = {}
        self._reset_progress_threshold = reset_progress_threshold

        env = mats_gym.raw_env(
            config=configs[0],
            scenario_fn=self._scenario_fn,
            **kwargs
        )
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        obs_space = self.env.observation_space(agent)
        obs_space["progress"] = gymnasium.spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
        return obs_space

    def _scenario_fn(self, client, config):
        return RouteScenario(world=client.get_world(), config=config, debug_mode=self._debug_mode)

    def step(self, action: dict) -> tuple[dict[AgentID, ObsType], dict[AgentID, float], dict, dict]:
        obs, reward, term, trun, info = self.env.step(action)
        for agent in self.env.agents:
            progress = self._get_progress(info, agent=agent)
            self._progress[agent] = progress
        obs = self._add_progress(obs)
        return obs, reward, term, trun, info


    def _add_progress(self, obs: dict) -> dict:
        for agent in self.agents:
            obs[agent]["progress"] = np.array(self._progress.get(agent, 0), dtype=np.float32)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        options = options or {}
        if options.get("reload", True):
            if "route" not in options:
                config = self._configs[self._current_route]
                self._current_route = (self._current_route + 1) % len(self._configs)
            else:
                self._current_route = int(options["route"])
                config = next(
                    filter(
                        lambda c: c.name.replace("RouteScenario_", "") == str(self._current_route),
                        self._configs
                    ))
            options["scenario_config"] = config
            obs, self._info = self.env.reset(seed=seed, options=options)
        else:
            progress = options.get("progress", self._progress[self._ego_role_name])
            route = self.current_scenario.route
            start_idx = int(len(route) * progress)
            start, _ = route[start_idx]
            map = CarlaDataProvider.get_map()
            wp = map.get_waypoint(start.location, project_to_road=True)
            self.actors[self._ego_role_name].set_transform(wp.transform)
            CarlaDataProvider.get_world().tick()
            self._progress = {agent: options.get("progress", progress) for agent in self.agents}
            obs = {self._ego_role_name: self.env.observe(self._ego_role_name)}

        obs = self._add_progress(obs)


        current_route = self.current_scenario.route
        actor_config = self.current_scenario.config.ego_vehicles[0]
        actor_config = ActorConfiguration(
            route=current_route[int(len(current_route) * self._progress[self._ego_role_name]):],
            transform=self.actors[self._ego_role_name].get_transform(),
            rolename=self._ego_role_name,
            random=False,
            autopilot=False,
            color=actor_config.color,
            model=actor_config.model
        )
        self.current_scenario.config.ego_vehicles[0] = actor_config
        return obs, self._info

    @property
    def current_actor_config(self) -> ActorConfiguration:
        return self.current_scenario.config.ego_vehicles[0]

    def _get_progress(self, info: dict, agent: str):
        events = info.get(agent, {}).get("events", [])
        completion = list(filter(lambda e: e["event"] == "ROUTE_COMPLETION", events))
        if completion:
            return float(completion[0]["route_completed"]) / 100
        else:
            return self._progress[agent]

    def _parse_routes_file(self, route_filename: str, single_route_id: str = None):
        """
        Returns a list of route configuration elements.
        :param route_filename: the path to a set of routes.
        :param single_route: If set, only this route shall be returned
        :return: List of dicts containing the waypoints, id and town of the routes
        """

        route_configs = []
        tree = ElementTree.parse(route_filename)
        for route in tree.iter("route"):

            route_id = route.attrib["id"]
            if single_route_id and route_id != single_route_id:
                continue

            route_config = RouteScenarioConfiguration()
            route_config.town = route.attrib["town"]
            route_config.name = "RouteScenario_{}".format(route_id)
            route_config.weather = self._parse_weather(route)

            # The list of carla.Location that serve as keypoints on this route
            positions = []
            for position in route.find('waypoints').iter('position'):
                loc = carla.Location(
                    x=float(position.attrib['x']),
                    y=float(position.attrib['y']),
                    z=float(position.attrib['z'])
                )
                positions.append(loc)
            route_config.keypoints = positions

            # The list of ScenarioConfigurations that store the scenario's data
            scenario_configs = []
            for scenario in route.find("scenarios").iter("scenario"):
                scenario_config = ScenarioConfiguration()
                scenario_config.name = scenario.attrib.get("name")
                scenario_config.type = scenario.attrib.get("type")

                for elem in scenario:
                    if elem.tag == "trigger_point":
                        tf = carla.Transform(
                            carla.Location(
                                float(elem.attrib.get('x')),
                                float(elem.attrib.get('y')),
                                float(elem.attrib.get('z'))
                            ),
                            carla.Rotation(
                                roll=0.0,
                                pitch=0.0,
                                yaw=float(elem.attrib.get('yaw'))
                            )
                        )
                        scenario_config.trigger_points.append(tf)
                    elif elem.tag == "other_actor":
                        actor_config = self._parse_actor_config(elem)
                        scenario_config.other_actors.append(actor_config)
                    else:
                        scenario_config.other_parameters[elem.tag] = elem.attrib

                scenario_configs.append(scenario_config)
            route_config.scenario_configs = scenario_configs
            route_configs.append(route_config)
        return route_configs

    def _parse_actor_config(self, elem):
        model = elem.attrib.get('model', 'vehicle.*')
        pos_x = float(elem.attrib.get('x', 0))
        pos_y = float(elem.attrib.get('y', 0))
        pos_z = float(elem.attrib.get('z', 0))
        yaw = float(elem.attrib.get('yaw', 0))
        transform = carla.Transform(
            carla.Location(x=pos_x, y=pos_y, z=pos_z),
            carla.Rotation(yaw=yaw)
        )

        rolename = elem.attrib.get('rolename', 'other')
        speed = elem.attrib.get('speed', 0)
        autopilot = False
        if 'autopilot' in elem.keys():
            autopilot = True

        random_location = False
        if 'random_location' in elem.keys():
            random_location = True

        color = elem.attrib.get('color', None)
        return ActorConfigurationData(
            model=model,
            rolename=rolename,
            transform=transform,
            speed=speed,
            autopilot=autopilot,
            random=random_location,
            color=color
        )

    def _parse_weather(self, route):
        """
        Parses all the weather information as a list of [position, carla.WeatherParameters],
        where the position represents a % of the route.
        """
        weathers = []

        weathers_elem = route.find("weathers")
        if weathers_elem is None:
            return [[0, carla.WeatherParameters(sun_altitude_angle=70, cloudiness=50)]]

        for weather_elem in weathers_elem.iter('weather'):
            route_percentage = float(weather_elem.attrib['route_percentage'])

            weather = carla.WeatherParameters(sun_altitude_angle=70, cloudiness=50)  # Base weather
            for weather_attrib in weather_elem.attrib:
                if hasattr(weather, weather_attrib):
                    setattr(weather, weather_attrib, float(weather_elem.attrib[weather_attrib]))
                elif weather_attrib != 'route_percentage':
                    print(f"WARNING: Ignoring '{weather_attrib}', as it isn't a weather parameter")

            weathers.append([route_percentage, weather])

        weathers.sort(key=lambda x: x[0])
        return weathers

