from __future__ import annotations
from collections import defaultdict
import logging
from typing import Collection

from srunner.scenariomanager.traffic_events import TrafficEventType
from mats_gym.tasks.tasks import Task


class RouteFollowingTask(Task):
    """
    A task that rewards the agent for making progress along its route.
    Terminates when the agent reaches the end of its route. At each timestep,
    the reward is the progress (between 0 and 1) made along the route since the
    last timestep.
    Note that this task requires the scenario to monitor the route completion events
    and add them to the agent info.
    """

    def __init__(self, agent: str, extra_reward_on_completion: float = 0.0):
        """Constructor.

        Args:
            agent (str): Name of the agent to monitor.
        """
        super().__init__(agent)
        self._route_progress = 0.0
        self._extra_reward_on_completion = extra_reward_on_completion
        self._completed = False

    def reward(self, obs: dict, action: dict, info: dict) -> float:
        agent_info = info[self.agent]
        reward = 0.0
        for event in agent_info["events"]:
            if event["event"] == "ROUTE_COMPLETION":
                if "route_completed" in event:
                    progress = event["route_completed"]
                    reward = (progress - self._route_progress) / 100.0
                    self._route_progress = progress
                else:
                    reward = 0.0
            if event["event"] == "ROUTE_COMPLETED":
                if not self._completed:
                    reward = self._extra_reward_on_completion
                    self._completed = True
                break

        return reward

    def terminated(self, obs: dict, action: dict, info: dict) -> bool:
        for event in info[self.agent]["events"]:
            if event["event"] == "ROUTE_COMPLETED":
                return True
        return False

    def reset(self) -> None:
        self._route_progress = 0.0
        self._completed = False


class InfractionAvoidanceTask(Task):
    """
    A task that penalizes the agent for committing infractions.
    Terminates when an infraction occurs that is marked as a terminating infraction.
    Note that this task requires the scenario to monitor the corresponding TrafficEvent.
    This needs to be declared in the scenario specification.
    """

    def __init__(
        self,
        agent: str,
        infractions: Collection[str] = None,
        penalties: dict[str, float] | float = 1.0,
        terminate_on_infraction: set[str] | bool = False,
    ):
        """Constructor.

        Args:
            agent (str): Name of the agent to monitor.
            infractions (set[str], optional): Set of infraction names (as they appear in the event list in the info dict). Defaults to all available infractions.
            penalties (dict[str, float], optional): Dict that maps infractions to penalty weight. Defaults to 1.0 for each infraction.
            terminate_on_infraction (set[str], optional): Infractions for which the task should indicate termination. Empty by default.
        """
        super().__init__(agent)

        if infractions is None:
            infractions = [
                TrafficEventType.COLLISION_PEDESTRIAN.name,
                TrafficEventType.COLLISION_STATIC.name,
                TrafficEventType.COLLISION_VEHICLE.name,
                TrafficEventType.ON_SIDEWALK_INFRACTION.name,
                TrafficEventType.OUTSIDE_LANE_INFRACTION.name,
                TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION.name,
                TrafficEventType.ROUTE_DEVIATION.name,
                TrafficEventType.STOP_INFRACTION.name,
                TrafficEventType.TRAFFIC_LIGHT_INFRACTION.name,
                TrafficEventType.VEHICLE_BLOCKED.name,
                TrafficEventType.WRONG_WAY_INFRACTION.name,
            ]
        infractions = set(infractions)

        if penalties is None:
            penalties = {infraction: 1.0 for infraction in infractions}
        elif isinstance(penalties, float) or isinstance(penalties, int):
            penalties = {infraction: penalties for infraction in infractions}
        self._penalties = penalties
        if isinstance(terminate_on_infraction, bool) and terminate_on_infraction:
            terminate_on_infraction = set(infractions)
        self._terminate_on_infraction = terminate_on_infraction or set()
        self._infractions = {infr: 0 for infr in infractions}

    def reward(self, obs: dict, action: dict, info: dict) -> float:
        agent_info = info[self.agent]
        infraction_count = defaultdict(int)
        for event in agent_info["events"]:
            if event["event"] in self._infractions:
                infraction_count[event["event"]] += 1

        infraction_count = dict(infraction_count)
        reward = -sum(
            self._penalties[event]
            for event, count in infraction_count.items()
            if count > self._infractions[event]
        )
        self._infractions.update(infraction_count)
        return reward

    def terminated(self, obs: dict, action: dict, info: dict) -> bool:
        agent_info = info[self.agent]
        for event in agent_info["events"]:
            if event["event"] in self._terminate_on_infraction:
                return True
        return False

    def reset(self) -> None:
        self._infractions = {infr: 0 for infr in self._infractions}
