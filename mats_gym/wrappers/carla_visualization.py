from __future__ import annotations

from typing import Callable

import carla
from pettingzoo.utils.env import AgentID, ActionType, ObsType
from srunner.scenarios.basic_scenario import BasicScenario

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper

VisualizationCallback = Callable[[BasicScenario, carla.World], None]


class CarlaVisualizationWrapper(BaseScenarioEnvWrapper):
    """
    A visualization helper for Carla environments that calls a list of callbacks after each step.
    """

    def __init__(
        self, env: BaseScenarioEnvWrapper, callbacks: list[VisualizationCallback]
    ):
        """
        @param env: The environment to wrap.
        @param callbacks: A list of callbacks that are called after each step.
        """
        super().__init__(env)
        self._callbacks = callbacks

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        step = super().step(actions)
        for callback in self._callbacks:
            callback(self.env.current_scenario, self._world)
        return step

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        obs, info = super().reset(seed=seed, options=options)
        self._world = self.env.client.get_world()
        for callback in self._callbacks:
            callback(self.env.current_scenario, self._world)
        return obs, info
