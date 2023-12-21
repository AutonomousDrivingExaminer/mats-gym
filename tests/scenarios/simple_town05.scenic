param map = localPath('./maps/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model

NUM_AGENTS = 2

intersection = Uniform(*network.intersections)
vehicles = []
for i in range(NUM_AGENTS):
    lane = Uniform(*intersection.incomingLanes)
    spawn_point = OrientedPoint in lane.centerline
    car = Car at spawn_point,
        with rolename f"car_{i}"
    vehicles.append(car)

ego = vehicles[0]

terminate after 10 seconds
