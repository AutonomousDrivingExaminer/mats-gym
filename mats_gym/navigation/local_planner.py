# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """
from __future__ import annotations
from enum import IntEnum
from collections import deque
import random
from typing import Tuple, List

import carla

from .common import WaypointWithRoadOption, RoadOption
from .controller import VehiclePIDController

from mats_gym.navigation.misc import draw_waypoints, get_speed


class LocalPlanner:
    """
    LocalPlanner implements the basic behavior of following a
    trajectory of waypoints that is generated on-the-fly.

    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice,
    unless a given global plan has already been specified.
    """

    def __init__(
        self,
        vehicle: carla.Vehicle,
        parameters: dict = None,
        carla_map: carla.Map = None,
    ):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param parameters: dictionary of arguments with different parameters:
            dt: time between simulation steps
            target_speed: desired cruise speed in Km/h
            sampling_radius: distance between the waypoints part of the plan
            lateral_control_dict: values of the lateral PID controller
            longitudinal_control_dict: values of the longitudinal PID controller
            max_throttle: maximum throttle applied to the vehicle
            max_brake: maximum brake applied to the vehicle
            max_steering: maximum steering applied to the vehicle
            offset: distance between the route waypoints and the center of the lane
        :param carla_map: carla.Map instance to avoid the expensive call of getting it.
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = carla_map or self._world.get_map()

        self._vehicle_controller: VehiclePIDController = None
        self.target_waypoint: carla.Waypoint = None
        self.target_road_option: RoadOption = None
        self._waypoints_queue: deque[WaypointWithRoadOption] = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False

        # Base parameters
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = 2.0
        self._args_lateral_dict = {"K_P": 1.95, "K_I": 0.05, "K_D": 0.2, "dt": self._dt}
        self._args_longitudinal_dict = {
            "K_P": 1.0,
            "K_I": 0.05,
            "K_D": 0,
            "dt": self._dt,
        }
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0
        self._base_min_distance = 3.0
        self._distance_ratio = 0.0  # if 0.0 -> constant lookahead distance, if > 0.0 -> depends on vehicle speed
        self._follow_speed_limits = False

        # Overload parameters
        if parameters:
            if "dt" in parameters:
                self._dt = parameters["dt"]
            if "target_speed" in parameters:
                self._target_speed = parameters["target_speed"]
            if "sampling_radius" in parameters:
                self._sampling_radius = parameters["sampling_radius"]
            if "lateral_control_dict" in parameters:
                self._args_lateral_dict = parameters["lateral_control_dict"]
            if "longitudinal_control_dict" in parameters:
                self._args_longitudinal_dict = parameters["longitudinal_control_dict"]
            if "max_throttle" in parameters:
                self._max_throt = parameters["max_throttle"]
            if "max_brake" in parameters:
                self._max_brake = parameters["max_brake"]
            if "max_steering" in parameters:
                self._max_steer = parameters["max_steering"]
            if "offset" in parameters:
                self._offset = parameters["offset"]
            if "base_min_distance" in parameters:
                self._base_min_distance = parameters["base_min_distance"]
            if "distance_ratio" in parameters:
                self._distance_ratio = parameters["distance_ratio"]
            if "follow_speed_limits" in parameters:
                self._follow_speed_limits = parameters["follow_speed_limits"]

        # initializing controller
        self._plan_length = 0
        self._maneuver_plan_length = 0
        self._init_controller()

    def reset_vehicle(self):
        """Reset the ego-vehicle"""
        self._vehicle = None

    def _init_controller(self) -> None:
        """Controller initialization"""
        self._vehicle_controller = VehiclePIDController(
            self._vehicle,
            args_lateral=self._args_lateral_dict,
            args_longitudinal=self._args_longitudinal_dict,
            offset=self._offset,
            max_throttle=self._max_throt,
            max_brake=self._max_brake,
            max_steering=self._max_steer,
        )

        # Compute the current vehicle waypoint
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (
            current_waypoint,
            RoadOption.LANEFOLLOW,
        )
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def set_speed(self, speed: float) -> None:
        """
        Changes the target speed

        :param speed: new target speed in Km/h
        :return:
        """
        if self._follow_speed_limits:
            print(
                "WARNING: The max speed is currently set to follow the speed limits. "
                "Use 'follow_speed_limits' to deactivate this"
            )
        self._target_speed = speed

    def follow_speed_limits(self, value: bool = True) -> None:
        """
        Activates a flag that makes the max speed dynamically vary according to the spped limits

        :param value: bool
        :return:
        """
        self._follow_speed_limits = value

    def _compute_next_waypoints(self, k: int = 1) -> None:
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def current_plan_length(self):
        return self._plan_length

    def set_maneuver_plan(self, plan: List[WaypointWithRoadOption]):
        self.set_global_plan(plan, stop_waypoint_creation=False, clean_queue=True)
        self._maneuver_plan_length = len(plan)

    def set_global_plan(
        self,
        current_plan: List[WaypointWithRoadOption],
        stop_waypoint_creation: bool = True,
        clean_queue: bool = True,
    ) -> None:
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs
        The 'clean_queue` parameter erases the previous plan if True, otherwise, it adds it to the old one
        The 'stop_waypoint_creation' flag stops the automatic creation of random waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :param stop_waypoint_creation: bool
        :param clean_queue: bool
        :return:
        """
        self._plan_length = len(current_plan)
        if clean_queue:
            self._waypoints_queue.clear()

        # Remake the waypoints queue if the new plan has a higher length than the queue
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue

        for elem in current_plan:
            self._waypoints_queue.append(elem)

        self._stop_waypoint_creation = stop_waypoint_creation

    def run_step(self, debug: bool = False) -> carla.VehicleControl:
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """
        if self._follow_speed_limits:
            self._target_speed = self._vehicle.get_speed_limit()

        # Add more waypoints too few in the horizon
        if (
            not self._stop_waypoint_creation
            and len(self._waypoints_queue) < self._min_waypoint_queue_length
        ):
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle) / 3.6
        self._min_distance = (
            self._base_min_distance + self._distance_ratio * vehicle_speed
        )

        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            self._maneuver_plan_length = max(
                0, self._maneuver_plan_length - num_waypoint_removed
            )
            self._plan_length = max(0, self._plan_length - num_waypoint_removed)
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control = self._vehicle_controller.run_step(
                self._target_speed, self.target_waypoint
            )

        if debug:
            num_waypoints = min(len(self._waypoints_queue), 25)
            draw_waypoints(
                world=self._vehicle.get_world(),
                waypoints=[w[0] for w in list(self._waypoints_queue)[:num_waypoints]],
                life_time=0.3,
                size=0.1,
                color=carla.Color(255, 0, 255),
                z=0.05,
            )

        return control

    def get_incoming_waypoint_and_direction(
        self, steps: int = 3
    ) -> Tuple[carla.Waypoint | None, RoadOption]:
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        num_waypoints = len(self._waypoints_queue)
        if num_waypoints > 0:
            k = min(steps, num_waypoints - 1)
            return self._waypoints_queue[k]
        else:
            return None, RoadOption.VOID

    def get_plan(self) -> deque[WaypointWithRoadOption]:
        """Returns the current plan of the local planner"""
        return self._waypoints_queue

    def done(self) -> bool:
        """
        Returns whether the planner has finished
        :return: boolean
        """
        return len(self._waypoints_queue) == 0

    def maneuver_has_finished(self) -> bool:
        """
        Returns whether the current maneuver has finished
        :return: boolean
        """
        return self._maneuver_plan_length == 0


def _retrieve_options(
    list_waypoints: List[carla.Waypoint], current_waypoint: carla.Waypoint
) -> List[RoadOption]:
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(
    current_waypoint: carla.Waypoint, next_waypoint: carla.Waypoint, threshold=35
) -> RoadOption:
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :param threshold: angle threshold to classify a turn
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
