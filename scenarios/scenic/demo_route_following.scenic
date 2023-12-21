from __future__ import annotations
param map = localPath('maps/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model
from training.scenarios.behaviors import BasicAgentBehavior

param MAX_DISTANCE_TO_INTERSECTION = 35
param MANEUVER_TYPE = ManeuverType.STRAIGHT
param NPC_MANEUVER_CONFLICT_ONLY = False
param NPC_PARAMS = {}
param NUM_NPCS = 2

class RouteFollowingCar(Car):
    route: list[Lane]

mtype = ManeuverType(globalParameters.MANEUVER_TYPE)
if isinstance(globalParameters.NPC_PARAMS, dict):
    npc_params = [globalParameters.NPC_PARAMS] * globalParameters.NUM_NPCS
else:
    npc_params = globalParameters.NPC_PARAMS

assert len(npc_params) == globalParameters.NUM_NPCS
four_way_intersections = filter(lambda i: i.is4Way and i.isSignalized, network.intersections)
intersection = Uniform(*four_way_intersections)

startLane = Uniform(*intersection.incomingLanes)
start_maneuvers = filter(lambda i: i.type == mtype, startLane.maneuvers)
start_maneuver = Uniform(*start_maneuvers)
start_trajectory = [start_maneuver.startLane, start_maneuver.connectingLane, start_maneuver.endLane]
distance = Range(-globalParameters.MAX_DISTANCE_TO_INTERSECTION, -5)

ego = RouteFollowingCar following roadDirection from start_maneuver.startLane.centerline[-1] for distance,
    with route start_trajectory,
    with rolename "student",
    with color Color(0,1,0),
    with blueprint "vehicle.tesla.model3",
    with behavior BasicAgentBehavior()

npcs = []
adv_maneuvers = start_maneuver.conflictingManeuvers
for i in range(globalParameters.NUM_NPCS):
    distance = Range(-globalParameters.MAX_DISTANCE_TO_INTERSECTION, -5)
    if globalParameters.NPC_MANEUVER_CONFLICT_ONLY:
        maneuver = Uniform(*adv_maneuvers)
    else:
        maneuver = Uniform(*intersection.maneuvers)

    route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]
    npc = RouteFollowingCar following roadDirection from maneuver.startLane.centerline[-1] for distance,
        with route route,
        with rolename f"npc_{i}",
        with color Color(1,0,0),
        with blueprint "vehicle.tesla.model3",
        with behavior BasicAgentBehavior(opt_dict=npc_params[i])
    npcs.append(npc)

monitor TrafficLights:
    freezeTrafficLights()
    while True:
        setClosestTrafficLightStatus(ego, "green")
        wait

terminate after 20 seconds