from __future__ import annotations
param map = localPath('../maps/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model

DISTANCE_TO_INTERSECTION = Uniform(5, 25) * -1
NUM_VEHICLES = 16

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
    route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]
    vehicle = RouteFollowingCar at (OrientedPoint in maneuver.startLane.centerline),
        with rolename "agent_" + str(i),
        with name "agent_" + str(i),
        with route route
    print(f"Vehicle ", vehicle, vehicle.rolename)
    vehicles.append(vehicle)

ego = vehicles[0]

terminate after 30 seconds