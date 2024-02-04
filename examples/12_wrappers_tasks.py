import logging
import gymnasium
import numpy as np
from mats_gym.envs import renderers
from mats_gym.tasks.tasks import TaskCombination
from mats_gym.tasks.traffic_event_tasks import (
    InfractionAvoidanceTask,
    RouteFollowingTask,
)
import cv2
from srunner.scenariomanager.traffic_events import TrafficEventType
from mats_gym.wrappers.birdseye_view.birdseye import BirdViewProducer
from mats_gym.wrappers.birdview import BirdViewObservationWrapper, ObservationConfig
from mats_gym.wrappers.meta_actions_wrapper import MetaActionWrapper

from mats_gym.wrappers.task_wrapper import TaskWrapper
import mats_gym

"""
This example shows how to use the TaskWrapper class conveniently define new tasks.
A task defines how the reward and termination condition of an agent is defined.
Tasks can be combined using the TaskCombination class.
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


def show_obs(obs, agent):
    """
    Displays the birdview observation of the given agent. The layers are collapsed into a single RGB image.
    """
    img = obs[agent]["birdview"]
    obs = BirdViewProducer.as_rgb(img)
    cv2.imwrite("img.png", obs)
    cv2.waitKey(10)


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
    env = BirdViewObservationWrapper(env=env)

    tasks = {}
    for agent in env.agents:
        task = TaskCombination(
            agent=agent,
            tasks=[
                RouteFollowingTask(agent=agent),
                InfractionAvoidanceTask(
                    agent=agent,
                    infractions=[
                        TrafficEventType.COLLISION_VEHICLE.name,
                        TrafficEventType.COLLISION_STATIC.name,
                        TrafficEventType.COLLISION_PEDESTRIAN.name,
                        TrafficEventType.ON_SIDEWALK_INFRACTION.name,
                        TrafficEventType.ROUTE_DEVIATION.name,
                        TrafficEventType.OUTSIDE_LANE_INFRACTION.name,
                        TrafficEventType.WRONG_WAY_INFRACTION.name,
                        TrafficEventType.TRAFFIC_LIGHT_INFRACTION.name,
                        TrafficEventType.STOP_INFRACTION.name,
                    ],
                ),
            ],
            weights=[0.01, 1.0],
        )
        tasks[agent] = task

    env = TaskWrapper(
        env=env,
        tasks=tasks,
        ignore_wrapped_env_reward=True,
        ignore_wrapped_env_termination=False,
    )

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        rewards = {agent: 0.0 for agent in env.agents}
        while not done:
            actions = {agent: policy() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            show_obs(obs, "vehicle_1")
            for agent, reward in reward.items():
                rewards[agent] += reward
            print(
                f"Cum. Rewards: {', '.join([f'{agent}={reward}' for agent, reward in rewards.items()])}"
            )
            done = all(done.values())
            env.render()

        for agent in env.agents:
            print(f"Agent {agent}: reward={rewards[agent]}, events:")
            for event in info[agent]["events"]:
                text = f"  - {event['event']} at frame {event.get('frame', 'N/A')}"
                if (
                    event["event"] == TrafficEventType.ROUTE_COMPLETION.name
                    and "route_completed" in event
                ):
                    text += f" completion={event['route_completed']:.2f}"
                print(text)
    env.close()


if __name__ == "__main__":
    main()
