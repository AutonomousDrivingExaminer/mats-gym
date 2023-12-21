import logging
import math

import carla
import cv2
import numpy as np
from srunner.scenarios.basic_scenario import BasicScenario

import mats_gym
from mats_gym.envs import renderers
from mats_gym.wrappers import BirdViewObservationWrapper, MetaActionWrapper
from mats_gym.wrappers.birdview import BirdViewCropType
from mats_gym.wrappers.birdview import ObservationConfig

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
        scenario_specification="scenarios/scenic/demo.scenic",
        scenes_per_scenario=1,
        resample_scenes=False,
        agent_name_prefixes=["student", "npc"],
        render_mode="rgb_array",
        traffic_manager_port=9230,
        render_config=renderers.camera_pov(
            agent="student",
            height=720,
            width=1280,
            camera_transform=carla.Transform(
                carla.Location(x=0, y=0, z=30), carla.Rotation(pitch=-80)
            ),
        ),
        sensor_specs={
            "student":
                [
                    {"id": "rgb-center", "type": "sensor.camera.rgb", "x": 0.7, "y": 0.0, "z": 1.60},
                    {"id": "depth-center", "type": "sensor.camera.depth", "x": 0.7, "y": 0.0, "z": 1.60},
                    {"id": "gps", "type": "sensor.other.gnss", "x": 0.7, "y": -0.4, "z": 1.60},
                ]

        }
    )

    env = BirdViewObservationWrapper(
        env=env,
        obs_config=ObservationConfig(
            as_rgb=True,
            width=256,
            height=256,
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY
        )
    )
    print(env.observation_space)

    def route_vis_callback(scenario: BasicScenario, world: carla.World) -> None:
        map = world.get_map()
        colors = [(5, 0, 0), (0, 5, 0), (0, 0, 5), (5, 5, 0), (0, 5, 5), (5, 0, 5), (5, 5, 5)]

        for i, route in enumerate([cfg.route for cfg in scenario.config.ego_vehicles]):
            actor = scenario.ego_vehicles[i]
            rolename = actor.attributes.get("role_name")
            if rolename.startswith("npc"):
                color = colors[0]
            else:
                color = colors[1]
            rot = actor.get_transform().rotation
            box = actor.bounding_box
            box.location += actor.get_transform().location
            actor_loc = actor.get_location()
            for loc, _ in route:
                wpt = map.get_waypoint(loc)
                wpt_t = wpt.transform

                begin = wpt_t.location
                begin.z = actor_loc.z + 0.2
                angle = math.radians(wpt_t.rotation.yaw)
                end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
                world.debug.draw_arrow(
                    begin=begin,
                    end=end,
                    arrow_size=0.05,
                    color=carla.Color(*color),
                    life_time=100000000000000000000
                )

    env = MetaActionWrapper(
        env=env,
        agent_names=["student"],  # agent names which are controlled by the meta actions
        action_frequency=20,
        planner_options={
            "student": {
                "target_speed": 50.0,
                "ignore_traffic_lights": True,
                "ignore_stops": False
            }
        }
    )
    # The callback is then passed to the CarlaVisualizationWrapper. You can register multiple
    # callbacks by passing a list of callbacks.
    # env = CarlaVisualizationWrapper(env=env, callbacks=[route_vis_callback])
    for _ in range(6):
        client = carla.Client("localhost", 2000)
        obs, info = env.reset(options={"client": client})
        cv2.imwrite("birdview.jpeg", obs["student"]["birdview"])
        cv2.imwrite("rgb.jpeg", obs["student"]["rgb-center"])
        depth = obs["student"]["depth-center"]
        R, G, B = 0, 1, 2

        done = False
        while not done:
            actions = {agent: 0 for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            # 
            done = all(done.values())
            # for agent in obs:
            # if "birdview" in obs[agent]:

            img = env.render()
            cv2.imshow(f"birdview", img[..., ::-1])
            cv2.waitKey(1)
    env.close()


def freeze(client, active=True):
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = active
    world.apply_settings(settings)


if __name__ == "__main__":
    main()
