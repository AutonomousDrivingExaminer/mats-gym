# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""
import enum
import math
import random
from typing import List, Tuple, Optional, Dict

import carla
import numpy as np
from shapely.geometry import Polygon

from mats_gym.navigation.local_planner import LocalPlanner, RoadOption, WaypointWithRoadOption, _compute_connection
from mats_gym.navigation.global_route_planner import GlobalRoutePlanner
from mats_gym.navigation.misc import (get_speed, is_within_distance,
                                      get_trafficlight_trigger_location,
                                      compute_distance, draw_waypoints, get_surrounding_waypoints)


class Action(enum.IntEnum):
    ACCELERATE = 0
    DECELERATE = enum.auto()
    KEEP_LANE = enum.auto()
    LANE_CHANGE_LEFT = enum.auto()
    LANE_CHANGE_RIGHT = enum.auto()
    GO_STRAIGHT = enum.auto()
    TURN_LEFT = enum.auto()
    TURN_RIGHT = enum.auto()
    STOP = enum.auto()

class State(enum.IntEnum):
    IDLE = 0
    EXECUTING_MANEUVER = enum.auto()

class MetaActionsAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(
            self,
            vehicle: carla.Vehicle,
            target_speed: float = 20,
            opt_dict={},
            carla_map: carla.Map = None,
            route_planner: GlobalRoutePlanner = None
    ):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world: carla.World = self._vehicle.get_world()
        self._map = carla_map or self._world.get_map()
        self._topology = self._map.get_topology()
        self._last_traffic_light = None

        self._action = Action.KEEP_LANE
        self._action_wp = None
        self._intersection_paths = None
        self._state = State.IDLE
        self._maneuver_paths = []

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0
        self._update_frequency = 1 # Hz
        self._debug = False


        # Change parameters according to the dictionary
        opt_dict['target_speed'] = target_speed
        if "debug" in opt_dict:
            self._debug = opt_dict['debug']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']

        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, parameters=opt_dict, carla_map=self._map)
        self._global_planner = route_planner or GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

    def _generate_intersection_options(self, waypoint: carla.Waypoint) -> Dict[RoadOption, List[carla.Waypoint]]:
        next_waypoint = waypoint.next(self._sampling_resolution)[0]
        while not next_waypoint.is_intersection:
            waypoint = next_waypoint
            next_waypoint = waypoint.next(self._sampling_resolution)[0]
        successors = [
            succ
            for pred, succ
            in self._topology
            if pred.road_id == waypoint.road_id and pred.lane_id == waypoint.lane_id
        ]
        paths = [wp.next_until_lane_end(self._sampling_resolution) for wp in successors]
        labels = [_compute_connection(p[0], p[-1]) for p in paths]
        options = dict(zip(labels, paths))
        return options


    def get_available_actions(self) -> List[Action]:
        """
        Returns the actions available to the agent
        """
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        waypoint, option = self._local_planner.get_incoming_waypoint_and_direction(steps=2)


        if self._state == State.EXECUTING_MANEUVER:
            maneuver = list(self._local_planner.get_plan())[:self._local_planner._maneuver_plan_length]
            if self._debug:
                draw_waypoints(self._world, [p[0] for p in maneuver], z=0.05, size=0.15, color=carla.Color(0, 255, 0))

        actions = []
        if self._state == State.IDLE:
            if waypoint.is_intersection and not current_waypoint.is_intersection:
                intersection = waypoint.get_junction()
                options = self._generate_intersection_options(current_waypoint)
                if self._debug:
                    self._world.debug.draw_box(intersection.bounding_box, carla.Rotation())
                    for _, path in options.items():
                        draw_waypoints(self._world, path, z=0.05, size=0.15, color=carla.Color(255, 0, 0))
                        draw_waypoints(self._world, [path[0],path[-1]], z=0.05, size=0.15, color=carla.Color(0, 0, 255))
                for option in options:
                    if option == RoadOption.LEFT:
                        actions.append(Action.TURN_LEFT)
                    elif option == RoadOption.RIGHT:
                        actions.append(Action.TURN_RIGHT)
                    elif option == RoadOption.STRAIGHT:
                        actions.append(Action.GO_STRAIGHT)
                return actions
            elif current_waypoint.lane_type == carla.LaneType.Driving:
                actions = []
                if current_waypoint.lane_change == carla.LaneChange.Both:
                    actions.append(Action.LANE_CHANGE_LEFT)
                    actions.append(Action.LANE_CHANGE_RIGHT)
                elif current_waypoint.lane_change == carla.LaneChange.Left:
                    actions.append(Action.LANE_CHANGE_LEFT)
                elif current_waypoint.lane_change == carla.LaneChange.Right:
                    actions.append(Action.LANE_CHANGE_RIGHT)
                for action in actions.copy():
                    speed = get_speed(self._vehicle) / 3.6
                    dir = 'left' if action == Action.LANE_CHANGE_LEFT else 'right'
                    path = self._generate_lane_change_path(current_waypoint, direction=dir, distance_same_lane=0.5 * speed, distance_other_lane=0)
                    if path is not None:
                        if any([p.is_intersection for p, _ in path]):
                            actions.remove(action)
                actions.append(Action.KEEP_LANE)
                actions.append(Action.STOP)
                actions.append(Action.DECELERATE)
                actions.append(Action.ACCELERATE)
                return actions
        elif self._state == State.EXECUTING_MANEUVER:
            actions.append(Action.STOP)
            actions.append(Action.DECELERATE)
            actions.append(Action.ACCELERATE)
            actions.append(Action.KEEP_LANE)
            return actions
        else:
            raise ValueError('unknown state')


    def update_action(self, action: Action) -> None:
        if action == Action.LANE_CHANGE_LEFT:
            self.lane_change(direction='left', same_lane_time=0.5, other_lane_time=0)
        elif action == Action.LANE_CHANGE_RIGHT:
            self.lane_change(direction='right', same_lane_time=0.5, other_lane_time=0)
        elif action == Action.STOP:
            self._vehicle.set_light_state(carla.VehicleLightState.Brake)
            self._local_planner.set_speed(0.0)
        elif action == Action.DECELERATE:
            self.set_target_speed(max(0.0, self._target_speed - 1))
        elif action == Action.ACCELERATE:
            self.follow_speed_limits(False)
            self.set_target_speed(self._target_speed + 1)
        elif action == Action.KEEP_LANE:
            pass
        elif action == Action.GO_STRAIGHT:
            self.do_intersection_action(action=action)
        elif action == Action.TURN_LEFT:
            self.do_intersection_action(action=action)
        elif action == Action.TURN_RIGHT:
            self.do_intersection_action(action=action)


    def do_intersection_action(self, action: Action) -> None:
        waypoint = self._map.get_waypoint(self._vehicle.get_location())
        options = self._generate_intersection_options(waypoint)
        if action == Action.GO_STRAIGHT:
            option = RoadOption.STRAIGHT
        elif action == Action.TURN_LEFT:
            option = RoadOption.LEFT
        elif action == Action.TURN_RIGHT:
            option = RoadOption.RIGHT
        else:
            raise ValueError('unknown action')

        if option in options:
            path = [(wp, option) for wp in options[option]]
            last_wp = path[-1][0]
            while last_wp.is_intersection:
                last_wp = last_wp.next(self._sampling_resolution)[0]
                path.append((last_wp, RoadOption.LANEFOLLOW))
            if option == RoadOption.LEFT:
                self._vehicle.set_light_state(carla.VehicleLightState.LeftBlinker)
            elif option == RoadOption.RIGHT:
                self._vehicle.set_light_state(carla.VehicleLightState.RightBlinker)
            self._local_planner.set_maneuver_plan(path)
            self._state = State.EXECUTING_MANEUVER


    def add_emergency_stop(self, control: carla.VehicleControl) -> carla.VehicleControl:
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed: float) -> None:
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value: bool = True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits
            :param value (bool): whether to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self) -> LocalPlanner:
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self) -> GlobalRoutePlanner:
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location: carla.Location, start_location: carla.Location = None) -> None:
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)



    def trace_route(self, start_waypoint: carla.Waypoint, end_waypoint: carla.Waypoint) -> List[WaypointWithRoadOption]:
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self) -> carla.VehicleControl:
        """Execute one step of navigation."""
        hazard_detected = False

        # Retrieve all relevant actors
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step(debug=self._debug)

        if self._local_planner.maneuver_has_finished():
            self._vehicle.set_light_state(carla.VehicleLightState.NONE)
            self._state = State.IDLE

        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control


    def ignore_traffic_lights(self, active: bool = True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active: bool = True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active: bool = True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def lane_change(
            self,
            direction: str,
            same_lane_time: float = 0.0,
            other_lane_time: float = 1.0,
            lane_change_time: float = 2.0,
            check: bool = True
    ) -> None:
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        :param direction: 'left' or 'right'
        :param same_lane_time: time spent in the same lane before starting the maneuver
        :param other_lane_time: time spent in the other lane before returning to the original lane
        :param lane_change_time: time spent in the lane change maneuver
        """
        speed = get_speed(self._vehicle) / 3.6
        path = self._generate_lane_change_path(
            waypoint=self._map.get_waypoint(self._vehicle.get_location()),
            direction=direction,
            distance_same_lane=same_lane_time * speed,
            distance_other_lane=other_lane_time * speed,
            lane_change_distance=lane_change_time * speed,
            check=check,
            lane_changes=1,
            step_distance=self._sampling_resolution
        )

        if not path:
            print("WARNING: Ignoring the lane change as no path was found")
        else:
            blinker = carla.VehicleLightState.LeftBlinker if direction == "left" else carla.VehicleLightState.RightBlinker
            self._vehicle.set_light_state(blinker)
            self._local_planner.set_maneuver_plan(path)
            self._state = State.EXECUTING_MANEUVER

    def _affected_by_traffic_light(
            self,
            lights_list: List[carla.TrafficLight] = None,
            max_distance: float = None) -> Tuple[bool, Optional[carla.TrafficLight]]:
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected(
            self,
            vehicle_list: List[carla.Vehicle] = None,
            max_distance: float = None,
            up_angle_th: float = 90,
            low_angle_th: float = 0,
            lane_offset: float = 0
    ) -> Tuple[bool, Optional[carla.Vehicle], float]:
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """

        def get_route_polygon() -> Optional[Polygon]:
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    distance = compute_distance(target_vehicle.get_location(), ego_location)
                    return (True, target_vehicle, distance)

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance,
                                      [low_angle_th, up_angle_th]):
                    distance = compute_distance(target_transform.location, ego_transform.location)
                    return (True, target_vehicle, distance)

        return (False, None, -1)

    def _generate_lane_change_path(
            self,
            waypoint: carla.Waypoint,
            direction: str,
            distance_same_lane: float = 10,
            distance_other_lane: float = 25,
            lane_change_distance: float = 25,
            check: bool = True,
            lane_changes: float = 1,
            step_distance: float = 2
    ) -> List[WaypointWithRoadOption]:
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:

            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan

    def keep_lane(self, lane_time):
        pass


