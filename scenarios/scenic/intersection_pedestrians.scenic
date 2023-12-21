""" Scenario Description
Traffic Scenario 08.
Unprotected left turn at intersection with oncoming traffic.
The ego-vehicle is performing an unprotected left turn at an intersection, yielding to oncoming
traffic.
"""
import carla
from agents.navigation.basic_agent import BasicAgent
from scenic.simulators.carla.utils import utils


## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('../maps/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model

from scenic.simulators.carla.behaviors import AutopilotBehavior

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz_2017"

EGO_SPEED = 10
SAFETY_DISTANCE = 20
BRAKE_INTENSITY = 1.0
NUM_VEHICLES = 2
NUM_PEDESTRIANS = 2

EGO_COLOR = Color(0, 1, 0)
ADVS_COLOR = Color(1, 0, 0)
ADV2_COLOR = Color(1,0,0)

## MONITORS
monitor TrafficLights:
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        if False and withinDistanceToTrafficLight(adversary, 100):
            setClosestTrafficLightStatus(adversary, "green")
        wait


fourWayIntersection = filter(lambda i: i.is4Way and i.isSignalized, network.intersections)
intersec = Uniform(*fourWayIntersection)
ego_start_lane = Uniform(*intersec.incomingLanes)

# Get the ego manuever
ego_maneuvers = filter(lambda i: i.type == ManeuverType.LEFT_TURN, ego_start_lane.maneuvers)
ego_maneuver = Uniform(*ego_maneuvers)
ego_start_section = ego_maneuver.startLane.sections[-1]

# Get the adversary maneuver
adv_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, ego_maneuver.conflictingManeuvers)
adv_maneuver = Uniform(*adv_maneuvers)
adv_trajectory = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]

adv_start_lane = adv_maneuver.startLane
adv_end_section = adv_maneuver.endLane.sections[0]

## OBJECT PLACEMENT
ego_spawn_pt = OrientedPoint in ego_maneuver.startLane.centerline
ego_target_pt = OrientedPoint in ego_maneuver.endLane.centerline
adv_spawn_pt = OrientedPoint in adv_maneuver.startLane.centerline
adv_spawn_pt2 = OrientedPoint in adv_maneuver.startLane.centerline

ego = Car at ego_spawn_pt,
    with rolename "vehicle_sut"


for i in range(NUM_PEDESTRIANS):
    sidewalk = Uniform(*[ego.oppositeLaneGroup.sidewalk, ego.laneGroup.sidewalk])
    Pedestrian on visible sidewalk,
        with rolename f"ped_{i}"


for i in range(NUM_VEHICLES):
    lane = Uniform(*intersec.incomingLanes, *intersec.outgoingLanes)
    Car at (OrientedPoint in lane.centerline),
        with rolename "vehicle_" + str(i)

require (ego_start_section.laneToLeft == adv_end_section)  # make sure the ego and adversary are spawned in opposite lanes
require 25 <= (distance to intersec) <= 30

terminate when (distance to ego_target_pt.position) < 1
terminate after 10 seconds
