import logging

import carla
import numpy as np
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    Criterion,
)
from srunner.scenariomanager.traffic_events import TrafficEvent
from srunner.scenarios.cut_in import CutIn

import mats_gym
from mats_gym.envs.renderers import camera_pov
from mats_gym.wrappers import CriteriaWrapper

"""
This example shows how to use the ScenarioRunnerEnv adapter.
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


class MaxVelocityTest(Criterion):
    def __init__(self, actor: carla.Actor, max_velocity: float, optional=False):
        self.max_velocity = max_velocity
        super(MaxVelocityTest, self).__init__(
            "MaximumVelocityTest", actor, optional, terminate_on_failure=False
        )

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        velocity = CarlaDataProvider.get_velocity(self.actor)
        self.actual_value = velocity
        if velocity > self.max_velocity:
            self.test_status = "FAILURE"
            event = TrafficEvent(
                event_type="velocity_limit",
                message="Velocity limit exceeded",
                dictionary={"velocity": velocity, "max_velocity": self.max_velocity},
                frame=GameTime.get_frame(),
            )
            self.events.append(event)
        else:
            self.test_status = "SUCCESS"
        return new_status


def scenario_fn(world, config):
    return CutIn(
        world=world, config=config, ego_vehicles=config.ego_vehicles, timeout=10
    )


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    # If you want to add criteria to the scenario, you can use the CriteriaWrapper.
    # The CriteriaWrapper allows you to specify a list of criterion functions that are added to the scenario.
    # For instance, in this scenario, we also want to register a velocity test for the ego vehicle.
    env = mats_gym.srunner_env(
        host="localhost",
        port=2000,
        scenario_name="CutInFrom_left_Lane",
        config_file="scenarios/scenario-runner/CutIn.xml",
        render_mode="human",
        render_config=camera_pov(agent="scenario"),
        timeout=10,
    )
    env = CriteriaWrapper(
        env=env,
        criteria_fns=[lambda s: MaxVelocityTest(s.ego_vehicles[0], max_velocity=10)],
    )
    for _ in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        while not done:
            actions = {agent: policy() for agent in env.agents}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            if len(info["hero"]["events"]) > 0:
                print(info["hero"]["events"][-1])
            env.render()
    env.close()


if __name__ == "__main__":
    main()
