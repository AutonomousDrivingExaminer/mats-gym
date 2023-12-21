from __future__ import annotations
from typing import Callable

from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenarios.basic_scenario import BasicScenario


class ScenarioWrapper:
    def wrap(self, scenario: BasicScenario):
        return scenario


class CriteriaScenarioWrapper(ScenarioWrapper):
    def __init__(
        self,
        criteria_fns: list[Callable[[BasicScenario], Criterion]],
    ):
        self._criteria_fns = criteria_fns

    def wrap(self, scenario: BasicScenario):
        criteria = scenario.criteria_tree.children
        for criterion_fn in self._criteria_fns:
            criterion = criterion_fn(scenario)
            criteria.append(criterion)
        return scenario
