from __future__ import annotations

import importlib
import logging
from typing import Any

import carla
import gymnasium
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper

from mats_gym.envs.base_env import BaseScenarioEnv
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from pettingzoo.utils.wrappers import BaseParallelWrapper

class ScenarioRunnerEnv(BaseScenarioEnvWrapper):

    def __init__(
            self,
            client: carla.Client,
            scenario_name: str,
            config_file: str,
            scenario_module: str = None,
            timeout: float = 60.0,
            **kwargs
    ) -> None:
        self._client = client
        logging.debug(f"Loading scenario runner scenarios.")
        self._config_file = config_file
        self._scenario_module = scenario_module
        self._configs = ScenarioConfigurationParser.parse_scenario_configuration(
            scenario_name=scenario_name,
            additional_config_file_name=config_file
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
            **kwargs
        )
        super().__init__(env)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict[Any, dict]]:
        options = options or {}
        options["scenario_config"] = self._configs[self._current_config]
        self._current_config = (self._current_config + 1) % len(self._configs)
        return super().reset(seed=seed, options=options)

    def _make_scenario(self, client: carla.Client, config: ScenarioConfiguration):
        logging.debug(f"Loading scenario class {config.type}.")
        class_name = config.type.split(".")[-1]
        module_name = config.type[:-len(class_name) - 1]
        module = importlib.import_module(module_name)
        scenario_class = getattr(module, class_name)
        logging.debug(f"Spawning ego actors.")
        ego_vehicles = CarlaDataProvider.request_new_actors(config.ego_vehicles, tick=False)
        logging.debug(f"Creating scenario runner scenario.")
        scenario = scenario_class(
            world=client.get_world(),
            ego_vehicles=ego_vehicles,
            config=config,
            timeout=self._timeout
        )
        return scenario

