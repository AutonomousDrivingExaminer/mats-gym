import logging
import time
import carla
import gymnasium
import numpy as np
from srunner.scenarios.cut_in import CutIn
from srunner.scenarios.open_scenario import OpenScenario

from mats_gym.envs import renderers
from mats_gym.envs.renderers import camera_pov
import mats_gym

"""
This example shows how to use the ScenarioRunnerEnv adapter.
"""

NUM_EPISODES = 3


def policy():
    """
    A simple policy that drives the agent forward and turns left or right randomly.
    """
    return np.array(
        [1, 0, 0.0]  # + np.random.rand() / 2,  # throttle  # steer  # brake
    )


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    # Traffic scenarios that are defined in OpenSCENARIO format can be loaded using the OpenScenario environment adapter.
    # You can provide either a single scenario or a list of scenarios. Note that you have to pass a Carla client
    # or a host and port at initialization.
    env = mats_gym.openscenario_env(
        scenario_files="scenarios/open_scenario/FollowLeadingVehicle.xosc",
        host="localhost",  # The host to connect to
        port=2000,  # The port to connect to
        render_mode="human",  # The render mode. Can be "human", "rgb_array", "rgb_array_list".
        timeout=10,  # The timeout in seconds.
        render_config=camera_pov(
            agent="hero"
        ),  # See adex_gym.envs.renderers for more render configs.
    )

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        while not done:
            actions = {agent: policy() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    main()
