from __future__ import annotations

from collections import defaultdict
from typing import Any

from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType
from srunner.scenarios.basic_scenario import BasicScenario


class RouteEvaluator:

    def __init__(self, penalties: dict[str, float] = None):
        if not penalties:
            penalties = {
                TrafficEventType.COLLISION_PEDESTRIAN.name: 0.50,
                TrafficEventType.COLLISION_VEHICLE.name: 0.60,
                TrafficEventType.COLLISION_STATIC.name: 0.65,
                TrafficEventType.TRAFFIC_LIGHT_INFRACTION.name: 0.70,
                TrafficEventType.STOP_INFRACTION.name: 0.80
            }
        self._penalties = penalties

    def compute_score(self, events: list[TrafficEvent]) -> tuple[float, dict[str, Any]]:
        infractions = defaultdict(list)
        target_reached = False
        score_route = 1.0
        score_penalty = 1.0
        for event in events:
            event_type = event.get_type()
            if isinstance(event_type, TrafficEventType):
                event_type = event_type.name

            if event_type in self._penalties:
                score_penalty *= self._penalties[event_type]

            if event_type == TrafficEventType.ROUTE_COMPLETED.name:
                score_route = 100.0
                target_reached = True
            elif event_type == TrafficEventType.ROUTE_COMPLETION.name:
                if not target_reached and event.get_dict():
                    score_route = event.get_dict().get("route_completed", 0.0)
            else:
                infractions[event_type].append(event.get_message())

        scores = {
            "completion": score_route,
            "penalty": score_penalty,
            "total": max(score_route * score_penalty, 0.0),
            "infractions": dict(infractions),
        }
        return scores["total"], scores

