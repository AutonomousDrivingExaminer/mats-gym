""" Scenario Description
Based on 2019 Carla Challenge Traffic Scenario 07.
Ego-vehicle is going straight at an intersection but a crossing vehicle
runs a red light, forcing the ego-vehicle to perform a collision avoidance maneuver.
"""
from __future__ import annotations
param map = localPath('../maps/Town04.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town04'
model scenic.simulators.carla.model

DELAY_TIME_1 = 1 # the delay time for ego
DELAY_TIME_2 = 40 # the delay time for the slow car
FOLLOWING_DISTANCE = 13 # normally 10, 40 when DELAY_TIME is 25, 50 to prevent collisions

DISTANCE_TO_INTERSECTION1 = Uniform(15, 20) * -1
DISTANCE_TO_INTERSECTION2 = Uniform(10, 15) * -1
SAFETY_DISTANCE = 20
BRAKE_INTENSITY = 1.0
NUM_ADVERSARIES = 2

class RouteFollowingCar(Car):
    route: list[Lane]

monitor TrafficLights:
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
            for adversary in adversaries:
                setClosestTrafficLightStatus(adversary, "red")
        wait

four_way_intersections = filter(lambda i: i.is4Way, network.intersections)
intersection = Uniform(*four_way_intersections)

start_lane = Uniform(*intersection.incomingLanes)
maneuver = Uniform(*start_lane.maneuvers)

route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]
ego = RouteFollowingCar following roadDirection from start_lane.centerline[-1] for DISTANCE_TO_INTERSECTION1,
    with route route,
    with rolename "sut",
    with color Color(0,1,0)

adversaries = []
for i in range(NUM_ADVERSARIES):
    adversary_maneuver = Uniform(*maneuver.conflictingManeuvers)
    adv_spawn_point = adversary_maneuver.startLane.centerline[-1]
    adversary = Car following roadDirection from adv_spawn_point for DISTANCE_TO_INTERSECTION2,
        with rolename f"adv_{i}",
        with color Color(1,0,0)
    adversaries.append(adversary)

terminate when (distance to route[-1].centerline.end) < 3.0
terminate after 120 seconds