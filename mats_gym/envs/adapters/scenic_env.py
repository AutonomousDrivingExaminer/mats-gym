from __future__ import annotations

import logging
import random
from typing import Any

import carla
import numpy as np
import scenic
from scenic.core.scenarios import Scene, Scenario
from scenic.syntax import veneer

from mats_gym.envs.base_env import BaseScenarioEnv
from mats_gym.envs.renderers import RenderConfig
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.scenarios import ScenicScenario
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from mats_gym.scenarios.scenic_scenario import (
    ScenicScenarioConfiguration,
)


class ScenicEnv(BaseScenarioEnvWrapper):
    def __init__(
        self,
        client: carla.Client = None,
        scenario_specification: str | list[str] = None,
        scenes_per_scenario: int = 1,
        scenes: list[Scene] | Scene = None,
        agent_name_prefixes: str | list[str] = None,
        seed: int | None = None,
        resample_scenes: bool = False,
        traffic_manager_port: int = 8000,
        max_time_steps: int | None = None,
        timestep: float = 0.05,
        render_mode: str = None,
        render_config: RenderConfig = RenderConfig(),
        debug_mode=False,
        params: dict | list[dict] = None,
        **kwargs,
    ) -> None:
        assert (
            scenes is not None or scenario_specification is not None
        ), "Must specify either scenes or scenario specification."

        if seed is not None:
            logging.debug(f"Setting seed to {seed}.")
            random.seed(seed)
            np.random.seed(seed)

        if agent_name_prefixes is None:
            logging.info("No agent name prefixes specified. Using 'ego'.")
            agent_name_prefixes = ["ego"]

        if isinstance(agent_name_prefixes, str):
            agent_name_prefixes = [agent_name_prefixes]

        if isinstance(scenario_specification, str):
            scenario_specification = [scenario_specification]

        self._agent_name_prefixes = agent_name_prefixes
        self._scenes_per_scenario = max(scenes_per_scenario, 1)
        self._client = client
        self._traffic_manager_port = traffic_manager_port
        self._max_time_steps = max_time_steps
        self._timestep = timestep
        self._debug_mode = debug_mode
        self._resample_scenes = resample_scenes
        self._scenarios = scenario_specification
        self._seed = seed
        self._current_config = 0

        if scenes is None:
            logging.debug(f"Sampling scenes from scenario specification.")
            self._configs = self._sample_configs(
                scenarios=self._scenarios,
                scenes_per_scenario=self._scenes_per_scenario,
                params=params,
            )
        else:
            if isinstance(scenes, Scene):
                scenes = [scenes]
            self._configs = [self._build_config(scene) for scene in scenes]

        env = BaseScenarioEnv(
            seed=seed,
            client=self._client,
            config=self._configs[0],
            scenario_fn=self._make_scenario,
            render_mode=render_mode,
            debug_mode=self._debug_mode,
            render_config=render_config,
            traffic_manager_port=self._traffic_manager_port,
            **kwargs,
        )
        logging.debug(f"ScenicEnv initialized.")
        super().__init__(env)

    def _sample_configs(
        self, scenarios: list[str], scenes_per_scenario: int, params: list[dict] = None
    ) -> list[ScenicScenarioConfiguration]:
        configs = []
        params = params or {}
        if isinstance(params, dict):
            params = [params] * len(scenarios)
        for s, param in zip(scenarios, params):
            logging.debug(
                f"Number of scenes to sample for scenario {s}: {scenes_per_scenario}."
            )
            scenario = scenic.scenarioFromFile(
                s, params=param, model="scenic.simulators.carla.model"
            )
            for i in range(scenes_per_scenario):
                scene = scenario.generate(maxIterations=10000)[0]
                logging.debug(f"Building scene {i + 1}/{scenes_per_scenario} for {s}.")
                config = self._build_config(scene)
                configs.append(config)
        logging.debug(f"Number of scenes sampled: {len(configs)}.")
        return configs

    def _build_config(self, scene: Scene) -> ScenicScenarioConfiguration:
        ego_names = self._get_ego_names(scene)
        ego_vehicles = []
        for object in scene.objects:
            if object.rolename in ego_names:
                actor_config = ActorConfiguration(
                    model=object.blueprint,
                    rolename=object.rolename,
                    transform=None,
                    route=object.route if hasattr(object, "route") else None,
                )
                ego_vehicles.append(actor_config)

        config = ScenicScenarioConfiguration(
            scene=scene,
            town=scene.params["carla_map"],
            ego_vehicles=ego_vehicles,
            seed=self._seed,
            timestep=self._timestep,
            max_time_steps=self._max_time_steps,
            traffic_manager_port=self._traffic_manager_port,
        )
        return config

    def _make_scenario(
        self, client: carla.Client, config: ScenicScenarioConfiguration
    ) -> ScenicScenario:
        logging.debug(f"Creating scenic scenario.")
        scenario = ScenicScenario(
            config=config,
            client=client,
            debug_mode=self._debug_mode,
            criteria_enable=True,
            terminate_on_failure=True,
        )
        logging.debug(f"Scenic scenario created.")
        return scenario

    def _get_ego_names(self, scene):
        ego_names = []
        for name in self._agent_name_prefixes:
            for obj in scene.objects:
                if obj.rolename and obj.rolename.startswith(name):
                    ego_names.append(obj.rolename)
        return set(ego_names)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        options = options or {}
        if seed is not None:
            logging.debug(f"Resetting seed to {seed}.")
            random.seed(seed)
            np.random.seed(seed)
            self._seed = seed

        if self.env.current_scenario is not None:
            self.env.current_scenario.terminate()

        if "traffic_manager_port" in options:
            self._traffic_manager_port = options["traffic_manager_port"]

        options = options or {}
        if "scene" in options:
            if isinstance(options["scene"], Scene):
                config = self._build_config(options["scene"])
            elif isinstance(options["scene"], dict):
                assert (
                    "code" in options["scene"] and "binary" in options["scene"]
                ), "Scene must contain code and binary."
                params = options["scene"].get("params", {})
                code = options["scene"]["code"]
                scenario: Scenario = scenic.scenarioFromString(code, params=params)
                scene = scenario.sceneFromBytes(options["scene"]["binary"])
                config = self._build_config(scene)
        else:
            config = self._configs[self._current_config]
            self._current_config = (self._current_config + 1) % len(self._configs)
            if self._current_config == 0 and self._resample_scenes:
                if (
                    self.env.current_scenario is not None
                    and veneer.currentSimulation is not None
                ):
                    logging.info(
                        f"Current scenic simulation still running. Terminating."
                    )
                    self.env.current_scenario.terminate()
                logging.info(f"Resampling new scenes.")
                self._configs = self._sample_configs(
                    scenarios=self._scenarios,
                    scenes_per_scenario=self._scenes_per_scenario,
                )

        options["scenario_config"] = config
        obs, info = super().reset(seed=seed, options=options)
        return obs, info
