from __future__ import annotations

from typing import Any, Callable

from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenarios.basic_scenario import BasicScenario

from mats_gym import BaseScenarioEnv
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.scenarios.scenario_wrapper import CriteriaScenarioWrapper


class CriteriaWrapper(BaseScenarioEnvWrapper):
    def __init__(
            self,
            env: BaseScenarioEnvWrapper,
            criteria_fns: list[Callable[[BasicScenario], Criterion]],
    ):
        super().__init__(env)
        self._criteria_fns = criteria_fns

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[
        dict, dict[Any, dict]]:
        wrapper = CriteriaScenarioWrapper(self._criteria_fns)
        options = options or {}
        options["scenario_wrappers"] = [wrapper]
        obs, info = super().reset(seed=seed, options=options)
        return obs, info
