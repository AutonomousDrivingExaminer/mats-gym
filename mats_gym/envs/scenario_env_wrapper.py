from __future__ import annotations

import carla
from pettingzoo.utils.wrappers import BaseParallelWrapper
from srunner.scenarios.basic_scenario import BasicScenario

from mats_gym.envs.base_env import BaseScenarioEnv
from mats_gym.envs.replays import SimulationHistory


class BaseScenarioEnvWrapper(BaseParallelWrapper):
    """
    A wrapper for a scenario environment.
    """

    def __init__(self, env: BaseScenarioEnv):
        self.agents = env.agents[:]
        super().__init__(env)

    @property
    def history(self) -> SimulationHistory:
        """
        Returns the simulation history.
        @return: The simulation history of the current episode.
        """
        return self.env.history

    @property
    def client(self) -> carla.Client:
        """
        Returns the carla client.
        @return: The carla client.
        """
        return self.env.client

    @property
    def current_scenario(self) -> BasicScenario:
        """
        Returns the current scenario.
        @return: The current scenario.
        """
        return self.env.current_scenario

    @property
    def actors(self) -> dict[str, carla.Actor]:
        """
        Returns the current carla actor for each agent.
        @return: A dictionary mapping agent names to carla actors.
        """
        return self.env.actors
