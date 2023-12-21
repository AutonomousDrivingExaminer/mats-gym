from __future__ import annotations

import logging
import typing
from collections import defaultdict
from typing import Any, SupportsFloat

import carla
import gymnasium
import gymnasium.spaces
import numpy as np
from pettingzoo.utils import BaseParallelWrapper

from mats_gym.agents.meta_actions_agent import MetaActionsAgent, Action
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper


def get_vehicle_by_rolename(
        rolename: str, world: carla.World
) -> typing.Optional[carla.Vehicle]:
    actor_list = world.get_actors()
    for vehicle in actor_list.filter("vehicle.*"):
        if vehicle.attributes.get("role_name") == rolename:
            return vehicle
    return None


class MetaActionWrapper(BaseScenarioEnvWrapper):
    """
    Wrapper for the scenario environment that allows for discrete high-level navigation actions.
    """

    def __init__(
            self,
            env: BaseScenarioEnvWrapper,
            agent_names: list[str] = None,
            planner_options: dict[str, Any] = None,
            action_frequency: int = 20
    ):
        """
        Constructor method.
        :param env: The scenario environment to wrap.
        :param action_frequency: The number of simulation steps to take at each call to step().
        """
        super().__init__(env)
        self.env = env
        self.world = None
        self.map = None

        if planner_options is None:
            planner_options = {}
        self._planner_options = planner_options
        self._meta_action_agents: typing.Dict[str, MetaActionsAgent] = {}
        self._meta_action_agent_names = agent_names or env.agents
        action_spaces = {}
        for agent in env.agents:
            if agent in self._meta_action_agent_names:
                action_spaces[agent] = gymnasium.spaces.Discrete(len(Action))
            else:
                action_spaces[agent] = env.action_space(agent)

        observation_spaces = {}
        for agent in env.agents:
            agent_obs_space = env.observation_space(agent)
            if agent in self._meta_action_agent_names:
                agent_obs_space["action_mask"] = gymnasium.spaces.Box(
                    low=0, high=1, shape=(len(Action),), dtype=np.float32
                )
            observation_spaces[agent] = agent_obs_space
        self._observation_spaces = gymnasium.spaces.Dict(observation_spaces)
        self._action_spaces = gymnasium.spaces.Dict(action_spaces)

        self._action_frequency = action_frequency
        self._steps = 0
        self._frames = []
    
    def action_space(self, agent: Any) -> gymnasium.spaces.Dict:
        return self._action_spaces[agent]
    
    def observation_space(self, agent: Any) -> gymnasium.spaces.Dict:
        return self._observation_spaces[agent]
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict[Any, dict]]:
        logging.debug("Resetting MetaActionWrapper.")
        obs, info = super().reset(seed=seed, options=options)
        logging.debug("Finished child environment reset.")
        self._steps = 0
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        for agent in self._meta_action_agent_names:
            vehicle = self.env.actors[agent]
            logging.debug(f"Creating meta-action agent for '{agent}'.")
            planner_options = self._planner_options.get(agent, {})
            self._meta_action_agents[agent] = MetaActionsAgent(
                vehicle=vehicle,
                target_speed=planner_options.get("target_speed", 30),
                opt_dict=planner_options,
                carla_map=self.map,
            )
            actions = self._meta_action_agents[agent].get_available_actions()
            mask = self._get_action_mask(actions)
            info[agent]["action_mask"] = mask
            obs[agent]["action_mask"] = mask
        return obs, info

    def step(self, actions: dict) -> tuple[dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]]:
        action = actions.copy()
        # Only update the meta-action agents every _action_frequency steps.
        if self._steps % self._action_frequency == 0:
            for agent, act in action.items():
                if agent in self._meta_action_agents:
                    agent_action = Action(act)
                    logging.debug(f"Agent '{agent}' requested maneuver '{agent_action.name}'.")
                    self._meta_action_agents[agent].update_action(action=agent_action)

        for name, agent in self._meta_action_agents.items():
            control = agent.run_step()
            action[name] = np.array(
                [control.throttle, control.steer, control.brake]
            )

        obs, reward, terminated, truncated, info = self.env.step(action)

        self._steps += 1
        for agent in self._meta_action_agents:
            available_actions = self._meta_action_agents[agent].get_available_actions()
            if self._steps % self._action_frequency == 0:
                action_mask = self._get_action_mask(available_actions)
            else:
                action_mask = np.zeros(len(Action), dtype=np.float32)
            
            action_mask = action_mask * (1 - terminated[agent]) * (1 - truncated[agent])
            obs[agent]["action_mask"] = action_mask
            info[agent]["action_mask"] = action_mask

        return obs, reward, terminated, truncated, info

    def _get_action_mask(self, actions: list[Action]) -> np.ndarray:
        mask = np.zeros(len(Action), dtype=bool)
        for action in actions:
            mask[action] = True
        return mask
