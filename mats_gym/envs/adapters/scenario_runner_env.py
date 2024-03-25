from __future__ import annotations

import importlib
import logging
from typing import Any

import carla
import gymnasium
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper

from mats_gym.envs.base_env import BaseScenarioEnv
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from pettingzoo.utils.wrappers import BaseParallelWrapper

# !/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides access to a scenario configuration parser
"""

import glob
import os
import xml.etree.ElementTree as ET

import carla

from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
from srunner.scenarioconfigs.route_scenario_configuration import RouteConfiguration


def parse_scenario_configuration(scenario_name, additional_config_file_name):
    """
    Parse all scenario configuration files at srunner/examples and the additional
    config files, providing a list of ScenarioConfigurations @return

    If scenario_name starts with "group:" all scenarios that
    have that type are parsed and returned. Otherwise only the
    scenario that matches the scenario_name is parsed and returned.
    """

    if scenario_name.startswith("group:"):
        scenario_group = True
        scenario_name = scenario_name[6:]
    else:
        scenario_group = False

    scenario_configurations = []

    list_of_config_files = glob.glob("{}/srunner/examples/*.xml".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
    if additional_config_file_name != '':
        list_of_config_files.append(additional_config_file_name)

    for file_name in list_of_config_files:
        tree = ET.parse(file_name)

        for scenario in tree.iter("scenario"):

            scenario_config_name = scenario.attrib.get('name', None)
            scenario_config_type = scenario.attrib.get('type', None)

            # Check that the scenario is the correct one
            if not scenario_group and scenario_config_name != scenario_name:
                continue
            # Check that the scenario is of the correct type
            elif scenario_group and scenario_config_type != scenario_name:
                continue

            config = ScenarioConfiguration()
            config.town = scenario.attrib.get('town')
            config.name = scenario_config_name
            config.type = scenario_config_type

            for elem in scenario:
                # Elements with special parsing
                if elem.tag == 'ego_vehicle':
                    config.ego_vehicles.append(ActorConfigurationData.parse_from_node(elem, 'hero'))
                    config.trigger_points.append(config.ego_vehicles[-1].transform)
                elif elem.tag == 'other_actor':
                    config.other_actors.append(ActorConfigurationData.parse_from_node(elem, 'scenario'))
                elif elem.tag == 'weather':
                    for weather_attrib in elem.attrib:
                        if hasattr(config.weather, weather_attrib):
                            setattr(config.weather, weather_attrib, float(elem.attrib[weather_attrib]))
                        else:
                            print(f"WARNING: Ignoring '{weather_attrib}', as it isn't a weather parameter")

                elif elem.tag == 'route':
                    route_conf = RouteConfiguration()
                    route_conf.parse_xml(elem)
                    config.route = route_conf

                # Any other possible element, add it as a config attribute
                else:
                    config.other_parameters[elem.tag] = elem.attrib

            scenario_configurations.append(config)
    return scenario_configurations


def get_list_of_scenarios(additional_config_file_name):
    """
    Parse *all* config files and provide a list with all scenarios @return
    """

    list_of_config_files = glob.glob("{}/srunner/examples/*.xml".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
    list_of_config_files += glob.glob("{}/srunner/examples/*.xosc".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
    if additional_config_file_name != '':
        list_of_config_files.append(additional_config_file_name)

    scenarios = []
    for file_name in list_of_config_files:
        if ".xosc" in file_name:
            tree = ET.parse(file_name)
            scenarios.append("{} (OpenSCENARIO)".format(tree.find("FileHeader").attrib.get('description', None)))
        else:
            tree = ET.parse(file_name)
            for scenario in tree.iter("scenario"):
                scenarios.append(scenario.attrib.get('name', None))

    return scenarios


class ScenarioRunnerEnv(BaseScenarioEnvWrapper):
    def __init__(
            self,
            client: carla.Client,
            scenario_name: str,
            config_file: str,
            scenario_module: str = None,
            timeout: float = 60.0,
            **kwargs,
    ) -> None:
        self._client = client
        logging.debug(f"Loading scenario runner scenarios.")
        self._config_file = config_file
        self._scenario_module = scenario_module
        self._configs = parse_scenario_configuration(
            scenario_name=scenario_name, additional_config_file_name=config_file
        )
        for config in self._configs:
            ego_vehicles = []
            for vehicle_config in config.ego_vehicles:
                ego_vehicles.append(ActorConfiguration(**vehicle_config.__dict__))
            config.ego_vehicles = ego_vehicles

        self._current_config = 0
        self._timeout = timeout

        env = BaseScenarioEnv(
            client=client,
            config=self._configs[0],
            scenario_fn=self._make_scenario,
            **kwargs,
        )
        super().__init__(env)

    def reset(
            self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        options = options or {}
        options["scenario_config"] = self._configs[self._current_config]
        self._current_config = (self._current_config + 1) % len(self._configs)
        return super().reset(seed=seed, options=options)

    def _make_scenario(self, client: carla.Client, config: ScenarioConfiguration):
        logging.debug(f"Loading scenario class {config.type}.")
        class_name = config.type.split(".")[-1]
        module_name = config.type[: -len(class_name) - 1]
        module = importlib.import_module(module_name)
        scenario_class = getattr(module, class_name)
        logging.debug(f"Spawning ego actors.")
        ego_vehicles = CarlaDataProvider.request_new_actors(
            config.ego_vehicles, tick=False
        )
        logging.debug(f"Creating scenario runner scenario.")
        scenario = scenario_class(
            world=client.get_world(),
            ego_vehicles=ego_vehicles,
            config=config,
            timeout=self._timeout,
        )
        return scenario
