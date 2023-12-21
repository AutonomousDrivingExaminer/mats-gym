import logging

import cv2
import gymnasium
import numpy as np

from mats_gym.envs import renderers
from mats_gym.wrappers import AutonomousAgentWrapper
from examples.example_agents import AutopilotAgent
import mats_gym

"""
The AutonomousAgentWrapper allows to use an autonomous agent to take over the control of an agent
in a scenario environment.
"""

NUM_EPISODES = 3


def policy():
    """
    A simple policy that drives the agent forward and turns left or right randomly.
    """
    return np.array(
        [
            0.5 + np.random.rand() / 2,  # throttle
            np.random.rand() - 0.5,  # steer
            0.0,  # brake
        ]
    )


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )
    env = mats_gym.scenic_env(
        host="localhost",
        port=2000,
        scenario_specification="scenarios/scenic/carla_challenge_08.scenic",
        scenes_per_scenario=2,
        resample_scenes=False,
        agent_name_prefixes=["adv", "sut"],
        render_mode="human",
        render_config=renderers.camera_pov(agent="sut"),
    )

    # Instantiate an autonomous agent that will take over the control of the agent named "sut".
    agent = AutopilotAgent(
        role_name="sut",
        carla_host="localhost",
        carla_port=2000,
    )

    # Wrap the environment with the AutonomousAgentWrapper. You need to provide the name of the
    # agent that should be controlled by the autonomous agent and the autonomous agent itself.
    # Additionally, you can specify the path to a config file for the agent.
    env = AutonomousAgentWrapper(
        env=env,
        agent_name="sut",  # name of the agent to be controlled by the autonomous agent
        agent=agent,  # the autonomous agent
        agent_config=None,  # optional path to a config file for the agent
    )

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()

        # The agent is now controlled by the autonomous agent, thus not available as an agent in the
        # environment anymore.
        assert "sut" not in env.agents

        done = False
        while not done:
            # Take an action for the other agents.
            actions = {agent: policy() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    main()
