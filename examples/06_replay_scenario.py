import logging
import gymnasium
import numpy as np
from srunner.scenarios.cut_in import CutIn

from mats_gym import BaseScenarioEnv
from mats_gym.envs.renderers import camera_pov
import mats_gym

"""
This example shows how to use the replay functionality of the scenario environment.
"""

NUM_EPISODES = 3


def run_env(env, joint_policy):
    """
    Run the environment with a joint policy until the scenario is finished.
    """
    done = False
    while not done:
        actions = joint_policy()
        obs, reward, done, truncated, info = env.step(actions)
        done = all(done.values())
        env.render()
    return info


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )
    env = mats_gym.srunner_env(
        host="localhost",
        port=2000,
        seed=42,  # Set the seed to make the scenario deterministic.
        scenario_name="CutInFrom_left_Lane",
        config_file="scenarios/scenario-runner/CutIn.xml",
        render_mode="human",
        render_config=camera_pov(agent="scenario"),
        timeout=10,
    )
    obs, info = env.reset(seed=42)

    # Run the environment with a starting policy to generate a history of the scenario.
    info = run_env(env, lambda: {agent: np.array([0.75, 0, 0]) for agent in env.agents})

    # the environment has a history attribute that contains the history of the scenario.
    history = env.history

    # On the next reset, we can provide the history and the number of frames to replay. The environment will start from
    # the last frame of replay with the exact same state.
    replay = {"history": str(history), "num_frames": 120}

    # Replay the environment to frame 100 and then continue with a different policy.
    policies = [
        lambda: {agent: np.array([1.0, 0.0, 0.0]) for agent in env.agents},
        lambda: {agent: np.array([0.0, 0.0, 1.0]) for agent in env.agents},
        lambda: {agent: np.array([0.8, -0.3, 0.0]) for agent in env.agents},
    ]

    for policy in policies:
        obs, info = env.reset(seed=42, options={"replay": replay})
        run_env(env, policy)
    env.close()


if __name__ == "__main__":
    main()
