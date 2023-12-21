from __future__ import annotations
import logging
from typing import Any, SupportsFloat, Callable

import carla
import gymnasium
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.open_scenario import OpenScenario
from srunner.tools.openscenario_parser import OpenScenarioParser
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper

from mats_gym.envs.base_env import BaseScenarioEnv
from srunner.tools.scenario_parser import ScenarioConfigurationParser

from mats_gym.envs.renderers import RenderConfig
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from pettingzoo.utils.wrappers import BaseParallelWrapper

ScenarioConstructor = Callable[[carla.World, ScenarioConfiguration], BasicScenario]


class OpenScenarioEnv(BaseScenarioEnvWrapper):

    def __init__(
            self,
            client: carla.Client,
            scenario_files: str | list[str],
            timeout: int = 60,
            **kwargs
    ) -> None:
        if isinstance(scenario_files, str):
            scenario_files = [scenario_files]

        self._configs = [
            OpenScenarioConfiguration(filename=file, client=client, custom_params=None)
            for file in scenario_files
        ]
        for config in self._configs:
            ego_vehicles = []
            for vehicle_config in config.ego_vehicles:
                ego_vehicles.append(ActorConfiguration(**vehicle_config.__dict__))
            config.ego_vehicles = ego_vehicles
            
        self._config_files = scenario_files
        self._next_scenario = 0
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
        options["scenario_config"] = self._configs[self._next_scenario]
        self._next_scenario = (self._next_scenario + 1) % len(self._configs)
        return super().reset(seed=seed, options=options)

    def _make_scenario(self, client: carla.Client, config: OpenScenarioConfiguration):
        logging.debug(f"Creating OpenSCENARIO instance.")
        config_file = self._config_files[self._next_scenario]
        logging.debug(f"Spawning {len(config.ego_vehicles)} ego actors.")
        ego_vehicles = CarlaDataProvider.request_new_actors(config.ego_vehicles, tick=False)
        scenario = OpenScenario(
            world=client.get_world(),
            ego_vehicles=ego_vehicles,
            config=config,
            config_file=config_file,
            timeout=self._timeout,
            debug_mode=False
        )
        return scenario
