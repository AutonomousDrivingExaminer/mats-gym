"""
TITLE: Bypassing 02
AUTHOR: Francis Indaheng, findaheng@berkeley.edu
DESCRIPTION: Adversary vehicle performs a lane change to bypass the
slow ego vehicle before returning to its original lane.
SOURCE: NHSTA, #16
"""
import carla
from agents.navigation.basic_agent import BasicAgent
from scenic.simulators.carla.utils import utils

#################################
# MAP AND MODEL                 #
#################################

param map = localPath('../maps/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model

#################################
# CONSTANTS                     #
#################################

MODEL = 'vehicle.lincoln.mkz_2017'

param EGO_SPEED = VerifaiRange(2, 4)

param ADV_DIST = VerifaiRange(-25, -10)
param ADV_SPEED = VerifaiRange(7, 10)

BYPASS_DIST = [15, 10]
INIT_DIST = 50
TERM_TIME = 15
EGO_COLOR = Color(0, 1, 0)
ADVS_COLOR = Color(1, 0, 0)
ADV2_COLOR = Color(0,0,1)
EGO_MODEL = "vehicle.lincoln.mkz_2017"

#################################
# AGENT BEHAVIORS               #
#################################

behavior SUTBehavior(target_location):
    vehicle = self.carlaActor
    controller = BasicAgent(vehicle, target_speed=30)
    target_location = utils.scenicToCarlaLocation(target_location.position, z=0)
    controller.set_destination(target_location)
    while True:
        control = controller.run_step()
        take SetThrottleAction(control.throttle), SetSteerAction(control.steer), SetBrakeAction(control.brake)

#################################
# SPATIAL RELATIONS             #
#################################


laneSecsWithRightLane = []
for lane in network.lanes:
    for laneSec in lane.sections:
        if laneSec._laneToRight != None:
            laneSecsWithRightLane.append(laneSec)

assert len(laneSecsWithRightLane) > 0, \
    'No lane sections with adjacent left lane in network.'

initLaneSec = Uniform(*laneSecsWithRightLane)
rightLane = initLaneSec._laneToRight


adv_spawn_pt = OrientedPoint in initLaneSec.centerline
ego_spawn_pt = OrientedPoint in rightLane.centerline
ego_spawn_pt2 = OrientedPoint ahead of ego_spawn_pt by 5
ego_target_pt = OrientedPoint ahead of adv_spawn_pt by 50

#################################
# SCENARIO SPECIFICATION        #
#################################

adversary = Car at adv_spawn_pt,
    with rolename "adv_1",
    with name "adv_1",
    with color ADVS_COLOR

ego = Car at ego_spawn_pt2,
    with rolename "sut",
    with name "sut",
    with color EGO_COLOR,
    with blueprint EGO_MODEL,
    with behavior SUTBehavior(ego_target_pt)

require (distance to intersection) > INIT_DIST
require (distance from adversary to intersection) > INIT_DIST
require (distance from ego to adversary) < 8
terminate after TERM_TIME seconds