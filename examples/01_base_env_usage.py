import logging
import carla
import gymnasium
import numpy as np
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.cut_in import CutIn
from srunner.tools.scenario_parser import ScenarioConfigurationParser

from mats_gym.envs import renderers
import mats_gym

"""
This example shows how to use the BaseScenarioEnv class directly by passing a scenario factory function.
"""

NUM_EPISODES = 3
SENSOR_SPECS = [
    {"id": "rgb-center", "type": "sensor.camera.rgb", "x": 0.7, "y": 0.0, "z": 1.60},
    {
        "id": "lidar",
        "type": "sensor.lidar.ray_cast",
        "range": 100,
        "channels": 32,
        "x": 0.7,
        "y": -0.4,
        "z": 1.60,
        "yaw": -45.0,
    },
    {"id": "gps", "type": "sensor.other.gnss", "x": 0.7, "y": -0.4, "z": 1.60},
]


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


def scenario_fn(client: carla.Client, config: ScenarioConfiguration):
    """
    This function is called by the environment to create the scenario.
    :param client: The carla client.
    :param config: The scenario configuration.
    :return: A base scenario instance.
    """
    world = client.get_world()
    ego_vehicles = []
    for vehicle in config.ego_vehicles:
        actor = CarlaDataProvider.request_new_actor(
            model=vehicle.model,
            spawn_point=vehicle.transform,
            rolename=vehicle.rolename,
            color=vehicle.color,
            actor_category=vehicle.category,
        )
        ego_vehicles.append(actor)
    scenario = CutIn(
        world=world,
        ego_vehicles=ego_vehicles,
        config=config,
        debug_mode=False,
        timeout=10,
    )
    return scenario


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    # The base environment can be used directly by providing a scenario factory function. This function takes two
    # arguments (client and config) and returns a scenario instance. The scenario instance must be a subclass of BaseScenario.
    # Furthermore, the function is responsible creating the ego vehicles. For concrete examples, see
    # the adapter wrappers in adex_gym.envs.adapters (for Scenic and ScenarioRunner scenarios).

    # By default, the observation space for each agent is a dictionary with one key "state" which holds a
    # vector containing the position and velocity of the agent. If sensor specs are provided, the observation space
    # will contain an entry for each sensor with its corresponding id. Sensor specs are of the same format as in
    # the autonomous driving challenge (https://leaderboard.carla.org/get_started/#33-override-the-sensors-method).

    # If you wish to run the base environment, you need to set up the scenario configuration manually.
    config = ScenarioConfiguration()
    config.name = "cut_in"
    config.town = "Town04"
    ego_vehicle = ActorConfigurationData(
        rolename="ego",
        model="vehicle.lincoln.mkz2017",
        transform=carla.Transform(
            carla.Location(x=284.4, y=16.4, z=2.5),
            carla.Rotation(yaw=180)
        )
    )
    config.ego_vehicles = [ego_vehicle]
    config.trigger_points = [ego_vehicle.transform]
    config.other_actors = [ActorConfigurationData(
        model="vehicle.tesla.model3",
        rolename="scenario",
        transform=carla.Transform(
            carla.Location(x=324.2, y=20.7, z=2.5 - 105), # Original scenario let the car fall from the sky (for some reason)
            carla.Rotation(yaw=180)
        )
    )]
    config.weather = carla.WeatherParameters(sun_altitude_angle=75)

    # This config is then passed to the base environment.
    env = mats_gym.raw_env(
        config=config,  # The scenario configuration.
        scenario_fn=scenario_fn,  # A function that takes a carla client and a scenario config to instantiate a scenario.
        render_mode="human",  # The render mode. Can be "human", "rgb_array", "rgb_array_list".
        render_config=renderers.camera_pov(
            agent="scenario",  # The agent to follow with the camera.
        ),  # See adex_gym.envs.renderers for more render configs.
        sensor_specs={"ego": SENSOR_SPECS},  # sensor specs for each agent
    )

    for _ in range(NUM_EPISODES):
        # The client can be passed as late as on the first call to reset.
        obs, info = env.reset(options={"client": carla.Client("localhost", 2000)})
        done = False
        while not done:
            actions = {agent: policy() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    main()
