from __future__ import annotations

from typing import Any

import carla
import gymnasium
import gymnasium.spaces
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper


class RoadObservationWrapper(BaseScenarioEnvWrapper):
    """
    Wrapper to add road information to the observation.
    Road information includes:
    - Identification of the current lane (road, section, lane)
    - Lane type and width
    - Lane change possibility
    """

    def __init__(self, env: BaseScenarioEnvWrapper):
        """
        @param env: The environment to wrap.
        """
        super().__init__(env)
        self._lane_type_values = {
            v: i for i, v in enumerate(sorted(carla.LaneType.values))
        }
        self._lane_change = {
            carla.LaneChange.Both: 0,
            carla.LaneChange.Left: 1,
            carla.LaneChange.Right: 2,
            carla.LaneChange.NONE: 3,
        }

    def observation_space(self, agent: str) -> gymnasium.spaces.Dict:
        obs_space: gymnasium.spaces.Dict = self.env.observation_space(agent)
        obs_space["lane_id"] = gymnasium.spaces.Box(
            low=-10, high=10, shape=(1,), dtype=np.int32
        )
        obs_space["road_id"] = gymnasium.spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.int32
        )
        obs_space["section_id"] = gymnasium.spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.int32
        )
        obs_space["on_junction"] = gymnasium.spaces.Discrete(2)
        obs_space["lane_width"] = gymnasium.spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )
        obs_space["lane_type"] = gymnasium.spaces.Discrete(len(self._lane_type_values))
        obs_space["lane_change"] = gymnasium.spaces.Discrete(len(self._lane_change))
        return obs_space

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        obs, info = super().reset(seed, options)
        for agent in obs:
            obs[agent].update(self._get_road_information(agent))
        return obs, info

    def step(
        self, actions: dict
    ) -> tuple[
        dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]
    ]:
        obs, *rest = super().step(actions)
        for agent in obs:
            obs[agent].update(self._get_road_information(agent))
        return obs, *rest

    def _get_road_information(self, agent: str) -> dict:
        location = self.env.actors[agent].get_location()
        map = CarlaDataProvider.get_map()
        waypoint: carla.Waypoint = map.get_waypoint(location)
        return {
            "lane_id": waypoint.lane_id,
            "road_id": waypoint.road_id,
            "section_id": waypoint.section_id,
            "on_junction": waypoint.is_junction,
            "lane_width": waypoint.lane_width,
            "lane_type": self._lane_type_values[waypoint.lane_type],
            "lane_change": self._lane_change[waypoint.lane_change],
        }

    def observe(self, agent: str) -> dict:
        observation = self.env.observe(agent)
        observation.update(self._get_road_information(agent))
        return observation
