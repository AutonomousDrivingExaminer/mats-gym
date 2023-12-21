from __future__ import annotations
from abc import ABC
import abc
from collections import defaultdict
from typing import Callable, Union


class Task(ABC):
    def __init__(self, agent: str):
        self.agent = agent

    @abc.abstractmethod
    def reward(self, obs: dict, action: dict, info: dict) -> float:
        pass

    @abc.abstractmethod
    def terminated(self, obs: dict, action: dict, info: dict) -> bool:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class TaskCombination(Task):
    def __init__(
        self,
        agent: str,
        tasks: list[Task],
        weights: list[float] = None,
        termination_fn: Callable[[list[bool]], bool] = None,
    ):
        super().__init__(agent)
        self._tasks = tasks
        self._weights = weights or [1.0 for _ in tasks]
        if termination_fn is None:
            termination_fn = all
        self._termination_fn = termination_fn

    def reset(self) -> None:
        for task in self._tasks:
            task.reset()

    def reward(self, obs: dict, action: dict, info: dict) -> float:
        return sum(
            w * task.reward(obs, action, info)
            for w, task in zip(self._weights, self._tasks)
        )

    def terminated(self, obs: dict, action: dict, info: dict) -> bool:
        terminated = [task.terminated(obs, action, info) for task in self._tasks]
        return self._termination_fn(terminated)
