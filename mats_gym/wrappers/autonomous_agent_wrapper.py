from __future__ import annotations

import logging
from collections import defaultdict
from typing import SupportsFloat, Any

import carla
import gymnasium
import gymnasium.spaces
import numpy as np
from gymnasium import Env
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.timer import GameTime

from mats_gym import BaseScenarioEnv
from mats_gym.envs import renderers
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper


class AutonomousAgentWrapper(BaseScenarioEnvWrapper):
    def __init__(
        self,
        env: BaseScenarioEnvWrapper,
        agent_name: str,
        agent: AutonomousAgent,
        agent_config: str = None,
    ) -> None:
        super().__init__(env)
        self.possible_agents = [a for a in self.env.agents if a != agent_name]
        self.agents = self.possible_agents[:]
        self._agent_name = agent_name
        self._agent = agent
        self._agent_config = agent_config
        self._agent_obs = None
    

    def step(self, actions: dict) -> tuple[dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]]:
        logging.debug(f"Compute vehicle control for {self._agent_name}.")
        action = actions.copy()
        sut_action = self._agent.run_step(
            self._agent_obs, timestamp=GameTime.get_time()
        )
        action[self._agent_name] = np.array(
            [sut_action.throttle, sut_action.steer, sut_action.brake]
        )
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._agent_obs = obs.pop(self._agent_name)
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict[Any, dict]]:
        options = options or {}
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        if "path_to_conf_file" in options:
            logging.info(
                f"Setup with agent config file: {options['path_to_conf_file']}"
            )
            self._agent.setup(path_to_conf_file=options["path_to_conf_file"])
        else:
            logging.debug("Setting up autonomous agent.")
            self._agent.setup(path_to_conf_file=self._agent_config)

        if "route" in options:
            logging.debug("Resetting route for autonomous agent.")
            gps, map = options["route"]
            self._agent.set_global_plan(gps, map)

        self._agent_obs = obs.pop(self._agent_name)
        return obs, info
