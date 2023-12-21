import logging
import math

import carla
import numpy as np
from srunner.scenarios.basic_scenario import BasicScenario

import mats_gym
from mats_gym.envs import renderers
from mats_gym.wrappers import CarlaVisualizationWrapper

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

    # Route visualization callbacks are called after each step and after each reset.
    # The callback receives the current scenario and the world as arguments.
    def route_vis_callback(scenario: BasicScenario, world: carla.World) -> None:
        map = world.get_map()
        colors = [(5, 0, 0), (0, 5, 0), (0, 0, 5), (5, 5, 0)]
        for i, route in enumerate([cfg.route for cfg in scenario.config.ego_vehicles]):
            actor = scenario.ego_vehicles[i]
            rot = actor.get_transform().rotation
            box = actor.bounding_box
            box.location += actor.get_transform().location
            world.debug.draw_box(
                box=box,
                rotation=rot,
                thickness=0.1,
                color=carla.Color(*colors[i]),
                life_time=0.1
            )
            for tf, _ in route:
                begin = tf.location + carla.Location(z=0.1)
                angle = math.radians(tf.rotation.yaw)
                end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
                world.debug.draw_arrow(
                    begin=begin,
                    end=end,
                    arrow_size=0.05,
                    color=carla.Color(*colors[i]),
                    life_time=0.1
                )

    # The callback is then passed to the CarlaVisualizationWrapper. You can register multiple
    # callbacks by passing a list of callbacks.
    env = CarlaVisualizationWrapper(env=env, callbacks=[route_vis_callback])

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
