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
NUM_NPCS = 0

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

## DEFINING BEHAVIORS
behavior AdversaryBehavior(trajectory):
    do FollowTrajectoryBehavior(trajectory=trajectory)

behavior SUTBehavior(target_location):
    vehicle = self.carlaActor
    controller = BasicAgent(vehicle, target_speed=30)
    target_location = utils.scenicToCarlaLocation(target_location.position, z=0)
    controller.set_destination(target_location)
    while True:
        control = controller.run_step()
        take SetThrottleAction(control.throttle), SetSteerAction(control.steer), SetBrakeAction(control.brake)

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
    with rolename "sut",
    with name "sut",
    with color EGO_COLOR,
    with blueprint EGO_MODEL,
    with behavior SUTBehavior(ego_target_pt)

adversary = Car at adv_spawn_pt,
    with rolename "adv_1",
    with name "adv_1",
    with color ADVS_COLOR

adversary2 = Car at adv_spawn_pt2,
    with rolename "adv_2",
    with name "adv_2",
    with color ADV2_COLOR

for i in range(NUM_NPCS):
    lane = Uniform(*intersec.incomingLanes, *intersec.outgoingLanes)
    npc = Car at (OrientedPoint in lane.centerline),
        with rolename "npc_" + str(i),
        with name "npc_" + str(i),
        with color EGO_COLOR,
        with blueprint EGO_MODEL,
        with behavior AutopilotBehavior()

require (ego_start_section.laneToLeft == adv_end_section)  # make sure the ego and adversary are spawned in opposite lanes
require 25 <= (distance to intersec) <= 30
#require 15 <= (distance from adversary to intersec) <= 20

terminate when (distance to ego_target_pt.position) < 1
terminate after 10 seconds
