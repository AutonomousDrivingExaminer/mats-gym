import logging

import numpy as np
from scenic.domains.driving.roads import ManeuverType

import mats_gym
from mats_gym.envs import renderers

"""
This example shows how to use the Scenic scenario adapter.
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

    # There exists a gymnasium factory with id "ScenicEnv-v0" that can be used to create a ScenicEnv.
    # You can provide either a single scenario or a list of scenarios. Moreover, you can control how many scenes per
    # scenario should be generated.
    env = mats_gym.scenic_env(
        host="localhost",  # The host to connect to
        port=2000,  # The port to connect to
        scenario_specification="scenarios/scenic/carla_challenge_08.scenic",
        # Path to the scenario specification
        scenes_per_scenario=5,  # How many scenes should be generated per scenario
        resample_scenes=False,
        # if True, the scenes are resampled after all initial scenes have been used.
        agent_name_prefixes=["sut", "adv"],
        # Each actor whose role-name starts with one of the prefixes is an agent.
        render_mode="human",  # The render mode. Can be "human", "rgb_array", "rgb_array_list".
        render_config=renderers.camera_pov(agent="sut"),
        # See adex_gym.envs.renderers for more render configs.
        params={
            "MANEUVER_TYPE": ManeuverType.LEFT_TURN.value,
            "NPC_MANEUVER_CONFLICT_ONLY": True,
            "NPC_PARAMS": {
                "ignore_traffic_lights": False,
                "ignore_vehicles": False,
                "target_speed": 30,
            },
            "NUM_NPCS": 1,
        },
    )

    for _ in range(5):
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
