from __future__ import annotations

import numpy as np
import optree
from pettingzoo.utils.env import AgentID, ActionType, ObsType

from mats_gym import BaseScenarioEnv
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper


class BlackDeathWrapper(BaseScenarioEnvWrapper):

    def __init__(self, env: BaseScenarioEnv, default_action: dict[str, ActionType] = None):
        super().__init__(env)
        if default_action is None:
            default_action = {agent: np.zeros_like(self.action_space(agent).low) for agent in self.agents}
        self._default_action = default_action
        self._terminated = {agent: False for agent in self.agents}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[
        dict[AgentID, ObsType], dict[AgentID, dict]]:
        obs, infos = super().reset(seed, options)
        self._terminated = {agent: False for agent in self.agents}
        return obs, infos

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        actions = {
            agent: actions[agent] if not self._terminated[agent] else self._default_action[agent]
            for agent
            in self.agents
        }
        obs, rewards, terminated, truncated, infos = super().step(actions)
        for agent in obs:
            self._terminated[agent] = self._terminated[agent] or terminated[agent]
            if self._terminated[agent]:
                obs[agent] = optree.tree_map(lambda x: np.zeros_like(x), obs[agent])
                rewards[agent] = 0.0
        return obs, rewards, terminated, truncated, infos