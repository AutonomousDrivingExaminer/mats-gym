from __future__ import annotations
param map = localPath('../maps/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model

DISTANCE_TO_INTERSECTION = Uniform(15, 20) * -1
NUM_VEHICLES = 1

class RouteFollowingCar(Car):
    route: list[Lane]

def is_4way_intersection(inter) -> bool:
    left_turns = filter(lambda i: i.type == ManeuverType.LEFT_TURN, inter.maneuvers)
    all_single_lane = all(len(lane.adjacentLanes) == 1 for lane in inter.incomingLanes)
    return len(left_turns) >=4 and inter.is4Way


vehicles = []
four_way_intersections = filter(is_4way_intersection, network.intersections)
intersection = Uniform(*four_way_intersections)
maneuvers = intersection.maneuvers
for i in range(NUM_VEHICLES):
    maneuver = Uniform(*maneuvers)
    maneuvers.remove(maneuver)
    route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]
    vehicle = RouteFollowingCar following roadDirection from maneuver.startLane.centerline[-1] for DISTANCE_TO_INTERSECTION,
        with route route,
        with rolename f"vehicle_{i}",
        with color Color(0,1,0)
    vehicles.append(vehicle)

ego = vehicles[0]

terminate after 15 seconds