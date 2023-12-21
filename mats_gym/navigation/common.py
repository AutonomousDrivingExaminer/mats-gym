from enum import IntEnum
from typing import Tuple

import carla


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """

    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


WaypointWithRoadOption = Tuple[carla.Waypoint, RoadOption]
