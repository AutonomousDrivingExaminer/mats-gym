from __future__ import annotations

import logging
import os

import carla
import numpy as np
from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.route_parser import RouteParser

import mats_gym
from mats_gym.envs import renderers
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from examples.example_agents import AutopilotAgent

"""
This example shows how run a leaderboard route scenario with a custom agent.
"""


def get_policy_for_agent(agent: AutonomousAgent):
    def policy(obs):
        control = agent.run_step(input_data=obs, timestamp=GameTime.get_time())
        action = np.array([control.throttle, control.steer, control.brake])
        return action

    return policy


def scenario_fn(client: carla.Client, config: ScenarioConfiguration):
    scenario = RouteScenario(world=client.get_world(), config=config, debug_mode=1)
    return scenario

def make_configs(agent: AutopilotAgent, route_scenarios: str, routes: str) -> list[RouteScenarioConfiguration]:
    configs = RouteParser.parse_routes_file(
        route_filename=routes
    )
    for config in configs:
        config.ego_vehicles = [
            # Use our version of ActorConfiguration which allows to have actor-specific routes
            ActorConfiguration(
                route=config.route,
                model="vehicle.lincoln.mkz2017",
                rolename=agent.role_name,
                transform=None
            )
        ]
    return configs

def main():
    # Set environment variable for the scenario runner root. It can be found in the virtual environment.
    os.environ["SCENARIO_RUNNER_ROOT"] = os.path.join(os.getcwd(), "venv/lib/python3.10/site-packages/srunner/scenarios")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    agent = AutopilotAgent(role_name="hero", carla_host="localhost", carla_port=2000)
    configs = make_configs(
        agent=agent,
        route_scenarios="scenarios/routes/all_towns_traffic_scenarios_public.json",
        routes="scenarios/routes/training.xml"
    )

    env = mats_gym.raw_env(
        config=configs[0],
        scenario_fn=scenario_fn,
        render_mode="human",
        render_config=renderers.camera_pov(agent="hero"),
    )
    client = carla.Client("localhost", 2000)
    client.set_timeout(120.0)

    for config in configs[0:]:
        config.agent = AutopilotAgent(role_name="hero", carla_host="localhost", carla_port=2000)
        obs, info = env.reset(options={"scenario_config": config, "client": client})
        config.agent.setup(path_to_conf_file="", trajectory=config.keypoints)
        policy = get_policy_for_agent(config.agent)
        done = False
        while not done:
            # Use agent to get control for the current step
            actions = {agent: policy(o) for agent, o in obs.items()}
            obs, reward, done, truncated, info = env.step(actions)
            done = done["hero"]
            env.render()
            print("EVENTS: ", info["hero"]["events"])

    env.close()


if __name__ == "__main__":
    main()
