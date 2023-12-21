import logging
import time
import carla
import gymnasium
import numpy as np
from srunner.scenarios.cut_in import CutIn

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

    # There exists a gymnasium factory with id "ScenarioRunnerEnv-v0" that can be used to create a ScenarioEnv.
    # Scenarios are specified their XML configuration file, the scenario name and a scenario factory function.
    env = mats_gym.srunner_env(
        scenario_name="group:srunner.scenarios.cut_in.CutIn",  # Name of the scenario
        config_file="scenarios/scenario-runner/CutIn.xml",  # Path to the scenario configurations
        render_mode="human",  # The render mode. Can be "human", "rgb_array", "rgb_array_list".
        timeout=10,  # The timeout in seconds.
        render_config=camera_pov(
            agent="scenario"
        ),  # See adex_gym.envs.renderers for more render configs.
    )

    for _ in range(NUM_EPISODES):
        obs, info = env.reset(options={"client": carla.Client("localhost", 2000)})
        done = False
        while not done:
            actions = {agent: policy() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    main()
