import logging
import carla

import cv2
import gymnasium
import numpy as np

from mats_gym.envs import renderers
from mats_gym.wrappers import BirdViewObservationWrapper
from mats_gym.wrappers.birdseye_view.birdseye import BirdViewProducer
from mats_gym.wrappers.birdview import ObservationConfig
import mats_gym

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

    # Wrap the environment with the BirdViewObservationWrapper.
    # This wrapper adds a multi-layer occupancy grid observation to the agent observations.
    # The observation config can be set for each agent individually or for all agents at once. This allows to configure
    # agents with different observation spaces.
    # The layers are:
    # - 0: Road
    # - 1: Lanes
    # - 2: Centerlines
    # - 3: Green traffic lights
    # - 4: Yellow traffic lights
    # - 5: Red traffic lights
    # - 6: Pedestrians
    # - 7: Ego vehicle
    # - 8: Route information
    # - 9+: Other vehicles (clustered by rolename prefixes)
    env = BirdViewObservationWrapper(
        env=env,
        obs_config={
            "sut": ObservationConfig(
                as_rgb=True, vehicle_class_prefixes=None, width=84, height=84
            ),
            "adv_1": ObservationConfig(
                as_rgb=True,
                vehicle_class_prefixes=["sut", "adv"],
                width=84,
                height=84,
                pixels_per_meter=2,
            ),
        },
    )

    print(env.observation_space)

    for _ in range(NUM_EPISODES):
        obs, info = env.reset(options={"client": carla.Client("localhost", 2000)})
        done = False
        while not done:
            actions = {agent: policy() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            for agent in obs:
                if "birdview" in obs[agent]:
                    cv2.imshow(f"{agent} birdview", obs[agent]["birdview"])
                    cv2.waitKey(1)
            env.render()
    env.close()


if __name__ == "__main__":
    main()
