import logging
from typing import Any, Callable

import gymnasium
import numpy as np
from mats_gym.envs import renderers
import mats_gym

"""
This example shows how to use the Scenic scenario adapter.
"""

NUM_EPISODES = 3


def make_policy(agent_name: str, action_space: gymnasium.Space) -> Callable[[], Any]:
    """
    Creates a policy for an agent. It distinguishes between pedestrians and vehicles.
    :param agent_name: agent name
    :param action_space: the action space of the agent
    :return: a policy that can be used to control the agent
    """

    def walker_policy():
        """
        A simple policy that makes the agent run forward.
        """
        action = action_space.sample()
        action["jump"] = 0
        action["direction"] = np.array([1.0, 0, 0])
        return action

    def vehicle_policy():
        """
        A simple policy that drives the agent forward and turns left or right randomly.
        """
        return np.array(
            [
                0,  # throttle
                np.random.rand() - 0.5,  # steer
                0.0,  # brake
            ]
        )

    if agent_name.startswith("ped"):
        return walker_policy
    else:
        return vehicle_policy


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )
    env = mats_gym.scenic_env(
        host="localhost",
        port=2000,
        scenario_specification="scenarios/scenic/intersection_pedestrians.scenic",
        scenes_per_scenario=2,
        resample_scenes=False,
        agent_name_prefixes=[
            "ped",
            "veh",
        ],  # we want to control pedestrians and vehicles
        render_mode="human",
        render_config=renderers.camera_pov(agent="ped_1"),
    )

    # Create a policy for each agent. The policy is a function that returns an action for the agent.
    policies = {
        agent: make_policy(agent, env.action_space(agent)) for agent in env.agents
    }

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        while not done:
            actions = {agent: policies[agent]() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    main()
