from __future__ import annotations
import carla
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper
from srunner.scenarios.basic_scenario import BasicScenario

from mats_gym.envs.base_env import BaseScenarioEnv


class BaseScenarioEnvWrapper(BaseParallelWrapper):

    def __init__(self, env: BaseScenarioEnv):
        self.agents = env.agents[:]
        super().__init__(env)

    @property
    def history(self) -> dict:
        return self.env.history
    
    @property
    def client(self) -> carla.Client:
        return self.env.client
    
    @property
    def current_scenario(self) -> BasicScenario:
        return self.env.current_scenario
    
    @property
    def actors(self) -> dict[str, carla.Actor]:
        return self.env.actors