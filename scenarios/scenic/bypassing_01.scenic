"""
TITLE: Bypassing 01
AUTHOR: Francis Indaheng, findaheng@berkeley.edu
DESCRIPTION: Ego vehicle performs a lane change to bypass a slow
adversary vehicle before returning to its original lane.
SOURCE: NHSTA, #16
"""

#################################
# MAP AND MODEL                 #
#################################

param map = localPath('../maps/Town03.xodr')
param carla_map = 'Town03'
model scenic.simulators.carla.model

#################################
# CONSTANTS                     #
#################################

MODEL = 'vehicle.tesla.model3'

param EGO_SPEED = Range(7, 10)

param ADV_DIST = Range(10, 25)
param ADV_SPEED = Range(2, 4)

BYPASS_DIST = [15, 10]
INIT_DIST = 50
TERM_TIME = 30

#################################
# AGENT BEHAVIORS               #
#################################

from scenic.simulators.carla.behaviors import AutopilotBehavior


behavior SUTBehavior(target_location):
    print(target_location)
    vehicle = self.carlaActor
    controller = BasicAgent(vehicle, target_speed=30)
    target_location = utils.scenicToCarlaLocation(target_location.position, z=0)
    controller.set_destination(target_location)
    while True:
        control = controller.run_step()
        take SetThrottleAction(0), SetSteerAction(control.steer), SetBrakeAction(control.brake)





#################################
# SPATIAL RELATIONS             #
#################################

initLane = Uniform(*network.lanes)
egoSpawnPt = OrientedPoint in initLane.centerline

#################################
# SCENARIO SPECIFICATION        #
#################################

ego = Car at egoSpawnPt,
    with name "sut",
    with rolename "sut",
	with blueprint MODEL,
	with behavior AutopilotBehavior()

adversary = Car following roadDirection for globalParameters.ADV_DIST,
    with name "adv_1",
    with rolename "adv_1",
	with blueprint MODEL

require (distance to intersection) > INIT_DIST
require (distance from adversary to intersection) > INIT_DIST