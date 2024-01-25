import logging
import math

import carla
import numpy as np
from srunner.scenarios.basic_scenario import BasicScenario

import mats_gym
from mats_gym.envs import renderers
from mats_gym.wrappers import CarlaVisualizationWrapper
from mats_gym.wrappers.road_graph import RoadGraphObservationWrapper

"""
This example shows how to use the CarlaVisualizationWrapper to create visualizations
inside the CARLA simulator. The visualization is done by adding a callback to the wrapper.
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
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    env = mats_gym.scenic_env(
        host="localhost",
        port=2000,
        scenario_specification="scenarios/scenic/four_way_route_scenario.scenic",
        scenes_per_scenario=2,
        resample_scenes=False,
        agent_name_prefixes=["vehicle"],
        render_mode="human",
        render_config=renderers.camera_pov(agent="vehicle_1"),
    )
    env = RoadGraphObservationWrapper(env=env)

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
