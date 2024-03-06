import logging
import carla

import cv2
import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from mats_gym.envs import renderers
from mats_gym.wrappers import BirdViewObservationWrapper, RoadGraphObservationWrapper
from mats_gym.wrappers.birdseye_view.birdseye import BirdViewProducer
from mats_gym.wrappers.birdview import ObservationConfig
import mats_gym
from mats_gym.wrappers.roadgraph_wrapper import RoadGraphTypes

"""
This example shows how to use the MetaActionsWrapper class to enable discrete, high-level actions.
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

def plot_roadgraph(ax, road_graph):
    xyz, dir, type = road_graph["xyz"], road_graph["dir"], road_graph["type"]
    for id in np.unique(road_graph["id"]):
        idx = (road_graph["id"] == id).reshape(-1)
        x = xyz[idx][:, 0]
        y = xyz[idx][:, 1]
        dx = dir[idx][:, 0] * 2
        dy = dir[idx][:, 1] *2
        ax.quiver(x, y, dx, dy, angles="xy", scale_units='xy', scale=10)
        ax.set_aspect('equal', adjustable='box')

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )
    env = mats_gym.scenic_env(
        scenario_specification="scenarios/scenic/carla_challenge_08.scenic",
        scenes_per_scenario=2,
        resample_scenes=False,
        agent_name_prefixes=["adv", "sut"],
        render_mode="human",
        render_config=renderers.camera_pov(agent="sut"),
    )

    # Wrap the environment with the roadgraph observation wrapper.
    # This will add an entry to the observations that contains the road layout in vector representation.
    env = RoadGraphObservationWrapper(
        env=env,
        max_samples=20000,
        sampling_resolution=2
    )

    print(env.observation_space(agent=env.agents[0]))

    for _ in range(NUM_EPISODES):
        obs, info = env.reset(options={"client": carla.Client("localhost", 2000)})

        fig, axs = plt.subplots(ncols=len(env.agents), nrows=1, figsize=(40, 40), layout="constrained")
        for i, agent in enumerate(env.agents):
            road_graph = obs[agent]["roadgraph"]
            plot_roadgraph(axs[i], road_graph)
        plt.show()
        done = False
        while not done:
            actions = {agent: policy() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    main()
