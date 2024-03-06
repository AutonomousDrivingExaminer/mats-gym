import os

import gymnasium
from pettingzoo.utils.env import AgentID, ObsType
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.tools.route_parser import RouteParser

from mats_gym import BaseScenarioEnv
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.scenarios.actor_configuration import ActorConfiguration


class RouteScenarioWrapper(BaseScenarioEnvWrapper):

    def __init__(
            self,
            env: BaseScenarioEnv | BaseScenarioEnvWrapper,
            route_file: str,
            ego_role_name: str = "hero",
            scenario_runner_root: str = f"{os.getcwd()}/.venv/lib/python3.10/site-packages/",
            debug_mode: int = 0
    ):
        os.environ["SCENARIO_RUNNER_ROOT"] = scenario_runner_root
        configs = RouteParser.parse_routes_file(route_filename=route_file)
        for config in configs:
            config.ego_vehicles = [
                # Use our version of ActorConfiguration which allows to have actor-specific routes
                ActorConfiguration(
                    route=None,
                    model="vehicle.lincoln.mkz2017",
                    rolename=ego_role_name,
                    transform=None,
                )
            ]
        self._ego_role_name = ego_role_name
        self._current_route = 0
        self._debug_mode = debug_mode
        self._configs = configs
        self._progress = {}
        self._info = {}
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        obs_space = self.env.observation_space(agent)
        obs_space["progress"] = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        return obs_space

    def _scenario_fn(self, client, config):
        return RouteScenario(world=client.get_world(), config=config, debug_mode=self._debug_mode)

    def step(self, action: dict) -> tuple[dict[AgentID, ObsType], dict[AgentID, float], dict, dict]:
        obs, reward, term, trun, info = self.env.step(action)
        for agent in self.env.agents:
            events = info.get(agent, {}).get("events", [])
            completion = list(filter(lambda e: e["event"] == "ROUTE_COMPLETION", events))
            if completion:
                self._progress[agent] = float(completion[0]["route_completed"]) / 100
                obs[agent]["progress"] = self._progress[agent]
        return obs, reward, term, trun, info

    def observe(self, agent: str) -> dict:
        obs = self.env.observe(agent)
        obs["progress"] = self._progress.get(agent, 0)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[
        dict[AgentID, ObsType], dict[AgentID, dict]]:

        options = options or {}
        if "route" not in options and not options.get("soft_reset", False):
            config = self._configs[self._current_route]
            self._current_route = (self._current_route + 1) % len(self._configs)
            options["scenario_config"] = config
        elif "route" in options and options["route"] is not None:
            self._current_route = int(options["route"])
            config = next(filter(lambda c: c.name.replace("RouteScenario_", "") == str(self._current_route), self._configs))
            options["scenario_config"] = config


        if options.get("start_progress", None) is not None:
            route = self.current_scenario.route
            start_idx = int(len(route) * options["start_progress"])
            start, _ = route[start_idx]
            map = CarlaDataProvider.get_map()
            wp = map.get_waypoint(start.location, project_to_road=True)
            self.actors[self._ego_role_name].set_transform(wp.transform)
            CarlaDataProvider.get_world().tick()
            self._progress = {agent: options["start_progress"] for agent in self.agents}
            obs = {self._ego_role_name: self.observe(self._ego_role_name)}
        else:
            obs, self._info = self.env.reset(seed=seed, options=options)
            self.current_scenario.config.ego_vehicles[0].route = self.current_scenario.route



        self._progress = {agent: 0 for agent in self.agents}

        return obs, self._info



