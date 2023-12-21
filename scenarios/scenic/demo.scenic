from __future__ import annotations
param map = localPath('../maps/Town04.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town04'
param weather = 'ClearSunset'
model scenic.simulators.carla.model
from scenarios.scenic.cars import RouteFollowingCar
from training.scenarios.behaviors import BasicAgentBehavior
def get_route(maneuver):
    route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]
    return route, maneuver.startLane.centerline[-1]
param MAX_DISTANCE_TO_INTERSECTION = 15
param MANEUVER_TYPE = ManeuverType.LEFT_TURN
param NPC_MANEUVER_CONFLICT_ONLY = False
param NPC_PARAMS = {}
NUM_PEDESTRIANS = 3
param NUM_NPCS = 4
mtype = globalParameters.MANEUVER_TYPE
intersection = Uniform(*network.intersections)
startLane = Uniform(*intersection.incomingLanes)
start_maneuver = Uniform(*filter(lambda i: i.type == mtype, startLane.maneuvers))
route, start = get_route(start_maneuver)
distance = Range(-globalParameters.MAX_DISTANCE_TO_INTERSECTION, -5)
print([i for i in network.intersections if len(i.crossings) > 0])

ego = RouteFollowingCar following roadDirection from start for distance,
    with route route,
    with rolename "student",
    with color Color(0,1,0),
    with blueprint "vehicle.tesla.model3"

terminate after 20 seconds