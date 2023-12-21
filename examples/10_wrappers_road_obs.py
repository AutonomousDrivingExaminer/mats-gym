import logging

import cv2
import gymnasium
import numpy as np

from mats_gym.envs import renderers
from mats_gym.wrappers import BirdViewObservationWrapper
from mats_gym.wrappers.birdseye_view.birdseye import BirdViewProducer
from mats_gym.wrappers.birdview import ObservationConfig
from mats_gym.wrappers.road_obs_wrapper import RoadObservationWrapper
import mats_gym

"""
This example shows how to use the MetaActionsWrapper class to enable discrete, high-level actions.
"""

NUM_EPISODES = 3


def policy():
    """
    A simple policy that drives the agent forward and turns left or right randomly.
    """
    return np.array([
        0.5 + np.random.rand() / 2,  # throttle
        np.random.rand() - 0.5,  # steer
        0.0  # brake
    ])

def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s")
    env = mats_gym.scenic_env(
        host="localhost",
        port=2000,
        scenario_specification="scenarios/scenic/carla_challenge_08.scenic",
        scenes_per_scenario=2,
        resample_scenes=False,
        agent_name_prefixes=["adv", "sut"],
        render_mode="human",
        render_config=renderers.camera_pov(agent="sut")
    )
    env = RoadObservationWrapper(env=env)
    print(env.action_space("sut"))
    print(env.observation_space("sut"))

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
