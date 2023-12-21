import logging
import carla
import gymnasium
import numpy as np
from mats_gym.envs import renderers
from mats_gym.wrappers import MetaActionWrapper
import mats_gym

"""
This example shows how to use the MetaActionsWrapper class to enable discrete, high-level actions.
The environment will return an action mask on every timestep. If a high-level action can be taken,
the corresponding action index will be set to True. Otherwise, it will be set to False.
"""

NUM_EPISODES = 3


def policy(obs: dict, agent: str):
    """
    A simple policy selects a random action from the available actions, but never stops.
    """
    # Get the available actions for the agent: an array of booleans for each action index.
    mask = obs[agent]["action_mask"]
    mask[-1] = False  # never brake
    probs = mask / np.sum(mask) # uniform distribution over the available actions
    return np.random.choice(range(len(mask)), p=probs)


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s")
    env = mats_gym.scenic_env(
        scenario_specification="scenarios/scenic/carla_challenge_08.scenic",
        scenes_per_scenario=2,
        resample_scenes=False,
        agent_name_prefixes=["adv", "sut"],
        render_mode="human",
        render_config=renderers.camera_pov(agent="sut")
    )

    # Wrap the environment with the MetaActionWrapper.
    # This wrapper allows to specify the frequency of the meta actions.
    # Possible actions are:
    # - 0: Accelerate
    # - 1: Decelerate
    # - 2: Keep current speed and lane
    # - 3: Lane change left
    # - 4: Lane change right
    # - 5: Go straight (only at intersections)
    # - 6: Turn left (only at intersections)
    # - 7: Turn right (only at intersections)
    # - 8: Stop
    env = MetaActionWrapper(
        env=env,
        agent_names=["sut"], # agent names which are controlled by the meta actions
        action_frequency=20,
        planner_options={
            "sut": {
                "target_speed": 50.0,
                "ignore_traffic_lights": True,
                "ignore_stops": False
            }
        }
    )

    for _ in range(NUM_EPISODES):
        # Info dicts contain information about the actions that are available to the agent.
        obs, info = env.reset(options={"client": carla.Client("localhost", 2000)})
        done = False
        while not done:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}

            # Check if sut can take an action
            action_mask = obs["sut"]["action_mask"]
            if any(action_mask):
                action = policy(obs, "sut")
                actions["sut"] = action

            # Step the environment with the selected actions.
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    main()
