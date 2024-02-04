from __future__ import annotations

import math
import os
import pickle
import random
import tempfile
from copy import deepcopy
from enum import Enum
from typing import Any

import carla
import gymnasium
import gymnasium.spaces
import numpy as np
import optree
import scenic.domains.driving.roads
import torch
from scenic.core.object_types import Point, OrientedPoint
from scenic.core.vectors import Vector
from scenic.domains.driving.roads import Network, Lane, VehicleType, LaneGroup, Road
from scenic.formats.opendrive import xodr_parser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from scenic.simulators.carla.utils.utils import scenicToCarlaLocation, carlaToScenicPosition

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from trafficgen.trafficgen.traffic_generator.traffic_generator import TrafficGen
from trafficgen.trafficgen.traffic_generator.utils.data_utils import process_data_to_internal_format
from trafficgen.trafficgen.utils.config import load_config_init
from trafficgen.trafficgen.utils.utils import rotate


class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    @staticmethod
    def is_road_line(line):
        return True if line.__class__ == RoadLineType else False

    @staticmethod
    def is_yellow(line):
        return True if line in [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        return True if line in [
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_SINGLE_WHITE
        ] else False


class RoadGraphObservationWrapper(BaseScenarioEnvWrapper):
    """
    Wrapper to add road information to the observation.
    Road information includes:
    - Identification of the current lane (road, section, lane)
    - Lane type and width
    - Lane change possibility
    """

    def __init__(
            self,
            env: BaseScenarioEnvWrapper,
            max_samples: int = 20000,
            max_radius: float = 100.0,
            line_resolution: float = 5.0,
            num_time_steps: int = 190,
            time_step: float = 0.1,
    ):
        """
        @param env: The environment to wrap.
        """
        self._max_samples = max_samples
        self._max_radius = max_radius
        self._line_resolution = line_resolution
        self._num_time_steps = num_time_steps
        self._time_step = time_step
        self._network: Network = None
        self._lane_ids: list[str] = []
        self._object_id = 0
        self._debug = True
        super().__init__(env)

    def observation_space(self, agent: str) -> gymnasium.spaces.Dict:
        obs_space: gymnasium.spaces.Dict = self.env.observation_space(agent)
        return obs_space
        N = self._max_samples
        max_int = np.iinfo(np.int32).max
        roadgraph_format = {
            "xyz": gymnasium.spaces.Box(-np.inf, np.inf, shape=(N, 3), dtype=np.float32),
            "dir": gymnasium.spaces.Box(-1, 1, shape=(N, 3), dtype=np.float32),
            "type": gymnasium.spaces.Box(0, 1, shape=(N, 19), dtype=np.int32),
            "valid": gymnasium.spaces.Box(0, 1, shape=(N,), dtype=np.int32),
            "id": gymnasium.spaces.Box(0, np.iinfo(np.int32).max, shape=(N,), dtype=np.int32),
        }
        obs_space["roadgraph"] = gymnasium.spaces.Dict(roadgraph_format)
        return obs_space

    def _get_roadgraph_format(self, xyz: np.ndarray, dir: np.ndarray, type: np.ndarray, valid: np.ndarray) -> dict:
        return {
            "xyz": xyz,
            "dir": dir,
            "type": type,
            "valid": valid,
        }

    def _get_type(self, actor: carla.Actor) -> int:
        type_id = actor.type_id
        if type_id.startswith("vehicle"):
            if actor.attributes["base_type"] == "bicycle":
                return 3
            else:
                return 1
        elif type_id.startswith("walker"):
            return 2
        return 4

    def _extract_agents(self, ego: str):
        world = CarlaDataProvider.get_world()
        actors = [self.actors[ego]]
        actors.extend([v for v in world.get_actors().filter("vehicle.*") if v.id != self.actors[ego].id])
        actors.extend([w for w in world.get_actors().filter("walker.*") if w.id != self.actors[ego].id])

        tracks = np.zeros((len(actors), 9), dtype=np.float32)
        for i, actor in enumerate(actors):
            loc = actor.get_location()
            vel = actor.get_velocity()
            heading = math.radians(actor.get_transform().rotation.yaw)
            bbox: carla.BoundingBox = actor.bounding_box.extent
            length, width = bbox.x * 2, bbox.y * 2
            type = self._get_type(actor)
            tracks[i] = np.array([
                loc.x, loc.y,
                vel.x, vel.y,
                heading,
                length, width,
                type,
                True
            ], dtype=np.float32)
        return tracks

    def _extract_traffic_lights(self, info: dict):
        world: carla.World = CarlaDataProvider.get_world()
        traffic_lights = world.get_actors().filter("traffic.traffic_light")

        dynamic_states = []
        for t in range(self._num_time_steps):
            current_time = t * self._time_step
            traffic_light_states = []

        for i, tl in enumerate(traffic_lights):
            tl.freeze(True)
            stop_points: list[carla.Waypoint] = tl.get_stop_waypoints()
            for stop_wp in stop_points:
                stop_loc = stop_wp.transform.location
                lane: Lane = self._network.laneAt(carlaToScenicPosition(stop_loc))
                idx = info[lane.id]["object_id"]
                if tl.state == carla.TrafficLightState.Red:
                    state = 1
                elif tl.state == carla.TrafficLightState.Yellow:
                    state = 2
                elif tl.state == carla.TrafficLightState.Green:
                    state = 3
                else:
                    state = 0
                data = np.array([
                    idx,
                    stop_loc.x, stop_loc.y, 0,
                    state,
                    1,

                ], dtype=np.float32)
                dynamic_states.append(data)
        return [dynamic_states]

    def _extract_center(self):
        world = CarlaDataProvider.get_world()
        map: carla.Map = CarlaDataProvider.get_map()
        lanes, infos = [], {}
        for lane in self._network.lanes:
            center_line = lane.centerline.pointsSeparatedBy(self._line_resolution)
            center_line = [scenicToCarlaLocation(p, world=world) for p in center_line]
            if self._debug:
                for s, e in zip(center_line[:-1], center_line[1:]):
                    s = carla.Location(s.x, s.y, 0.1)
                    e = carla.Location(e.x, e.y, 0.1)
                    #world.debug.draw_line(s, e, thickness=0.1, color=carla.Color(0, 5, 0), life_time=0)
            # 0 UNDEFINED
            # 1 FREEWAY
            # 2 SURFACE_STREET
            # 3 BIKE_LANE
            if VehicleType.BICYCLE in lane.vehicleTypes and VehicleType.CAR not in lane.vehicleTypes:
                type = 3
            elif VehicleType.CAR in lane.vehicleTypes:
                type = 2
            else:
                type = 0
            id = self._get_next_object_id()
            lanes.append(np.array([
                [p.x, p.y, type, id]
                for p in center_line
            ], dtype=np.float32))
            lane_widths = np.array([
                map.get_waypoint(p).lane_width
                for p in center_line
            ])
            infos[lane.id] = {
                "object_id": id,
                "width": np.array([lane_widths / 2, lane_widths / 2], dtype=np.float32),
            }
        return np.concatenate(lanes, axis=0), infos

    def _extract_edges(self):
        world = CarlaDataProvider.get_world()

        def get_line(polyline, type, id):
            line = []
            for point in polyline.pointsSeparatedBy(self._line_resolution):
                loc = scenicToCarlaLocation(point, world=world)
                line.append([loc.x, loc.y, type, id])
            if self._debug:
                for s, e in zip(line[:-1], line[1:]):
                    s = carla.Location(s[0], s[1], 0.1)
                    e = carla.Location(e[0], e[1], 0.1)
                    #world.debug.draw_line(s, e, thickness=0.1, color=carla.Color(5, 0, 0), life_time=0)
            return np.array(line, dtype=np.float32)

        edges = []
        for road in self._network.roads:
            left_edge = get_line(road.leftEdge, type=15, id=self._get_next_object_id())
            right_edge = get_line(road.rightEdge, type=15, id=self._get_next_object_id())
            center_line = get_line(road.centerline, type=16, id=self._get_next_object_id())
            edges.extend([left_edge, right_edge, center_line])
        return np.concatenate(edges, axis=0)

    def _extract_lines(self):

        def _get_line_type(type: carla.LaneMarkingType, color: carla.LaneMarkingColor):
            if color == carla.LaneMarkingColor.White:
                match type:
                    case carla.LaneMarkingType.Broken | carla.LaneMarkingType.BrokenBroken | carla.LaneMarkingType.BrokenSolid | carla.LaneMarkingType.NONE:
                        return RoadLineType.BROKEN_SINGLE_WHITE
                    case carla.LaneMarkingType.Solid:
                        return RoadLineType.SOLID_SINGLE_WHITE
                    case carla.LaneMarkingType.SolidSolid:
                        return RoadLineType.SOLID_DOUBLE_WHITE
                    case _:
                        return RoadLineType.UNKNOWN
            elif color == carla.LaneMarkingColor.Yellow:
                match type:
                    case carla.LaneMarkingType.Broken:
                        return RoadLineType.BROKEN_SINGLE_YELLOW
                    case carla.LaneMarkingType.BrokenBroken:
                        return RoadLineType.BROKEN_DOUBLE_YELLOW
                    case carla.LaneMarkingType.Solid:
                        return RoadLineType.SOLID_SINGLE_YELLOW
                    case carla.LaneMarkingType.SolidSolid:
                        return RoadLineType.SOLID_DOUBLE_YELLOW
                    case carla.LaneMarkingType.SolidBroken | carla.LaneMarkingType.BrokenSolid:
                        return RoadLineType.PASSING_DOUBLE_YELLOW
                    case _:
                        return RoadLineType.UNKNOWN
            else:
                return RoadLineType.UNKNOWN

        lines = []
        world = CarlaDataProvider.get_world()
        map: carla.Map = CarlaDataProvider.get_map()

        for lane in self._network.lanes:
            left, right = [], []
            left_id, right_id = self._get_next_object_id(), self._get_next_object_id()
            for point in lane.leftEdge.pointsSeparatedBy(self._line_resolution):
                loc = scenicToCarlaLocation(point, world=world)
                wp = map.get_waypoint(loc)
                line_type = _get_line_type(wp.left_lane_marking.type, wp.left_lane_marking.color)
                if len(left) != 0 and line_type.value != left[-1][-2]:
                    if len(left) > 1:
                        lines.append(np.stack(left, dtype=np.float32))
                    left_id = self._get_next_object_id()
                    left = []
                left.append([loc.x, loc.y, line_type.value, left_id])
            if len(left) > 1:
                lines.append(np.stack(left, dtype=np.float32))
                if self._debug:
                    for s, e in zip(left[:-1], left[1:]):
                        s = carla.Location(s[0], s[1], 0.1)
                        e = carla.Location(e[0], e[1], 0.1)
                        #world.debug.draw_line(s, e, thickness=0.1, color=carla.Color(0, 0, 5), life_time=0)
            for point in lane.rightEdge.pointsSeparatedBy(self._line_resolution):
                loc = scenicToCarlaLocation(point, world=world)
                wp = map.get_waypoint(loc)
                line_type = _get_line_type(wp.right_lane_marking.type, wp.right_lane_marking.color)
                if len(right) != 0 and line_type.value != right[-1][-2]:
                    if len(right) > 1:
                        lines.append(np.stack(right, dtype=np.float32))
                        if self._debug:
                            for s, e in zip(right[:-1], right[1:]):
                                s = carla.Location(s[0], s[1], 0.1)
                                e = carla.Location(e[0], e[1], 0.1)
                                #world.debug.draw_line(s, e, thickness=0.1, color=carla.Color(5, 0, 5), life_time=0)
                    right_id = self._get_next_object_id()
                    right = []
                right.append([loc.x, loc.y, line_type.value, right_id])
            if len(right) > 1:
                lines.append(np.stack(right, dtype=np.float32))

        lines = np.concatenate(lines, axis=0)
        lines[:, 2] += 5
        return lines

    def _extract_map(self):
        center_line, info = self._extract_center()
        map = [
            center_line,
            self._extract_edges(),
            # self._extract_crosswalks(),
            self._extract_lines()
        ]
        return np.concatenate(map, axis=0), info

    def from_list_to_array(self, inp_list, max_agent=32):
        agent = np.concatenate([x.get_inp(act=True) for x in inp_list], axis=0)
        agent = agent[:max_agent]
        agent_num = agent.shape[0]
        agent = np.pad(agent, ([0, max_agent - agent_num], [0, 0]))
        agent_mask = np.zeros([agent_num])
        agent_mask = np.pad(agent_mask, ([0, max_agent - agent_num]))
        agent_mask[:agent_num] = 1
        agent_mask = agent_mask.astype(bool)
        return agent, agent_mask

    def init_map(self):
        lane, center_info = self._extract_map()
        agent = self.agents[0]
        scene = {}
        agent_info = self._extract_agents(agent)
        traffic_lights = self._extract_traffic_lights(center_info)[0]

        scene = process_data_to_internal_format({
            "lane": lane,
            "all_agent": np.expand_dims(agent_info, axis=0),
            "traffic_light": np.expand_dims(traffic_lights, axis=0),
            "unsampled_lane": lane
        })[0]
        scene = optree.tree_map(lambda x: np.expand_dims(x, axis=0), scene)
        cfg = load_config_init("local")
        print('loading checkpoint...')
        trafficgen = TrafficGen(cfg)

        context_num = cfg['context_num']

        init_vis_dir = 'trafficgen/trafficgen/traffic_generator/output/vis/scene_initialized'
        if not os.path.exists(init_vis_dir):
            os.makedirs(init_vis_dir)
        tmp_pth = 'trafficgen/trafficgen/traffic_generator/output/initialized_tmp'
        if not os.path.exists(tmp_pth):
            os.makedirs(tmp_pth)

        trafficgen.init_model.eval()
        with torch.no_grad():
            scene["agent_mask"][..., :18] = 1
            data = deepcopy(scene)
            model_output = trafficgen.place_vehicles_for_single_scenario(data, 0, True, init_vis_dir, context_num)
            agent, agent_mask = self.from_list_to_array(model_output['agent'])
            data = optree.tree_map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, data)
            output = {}
            output['context_num'] = context_num
            output['all_agent'] = agent
            output['agent_mask'] = agent_mask
            output['lane'] = data['other']['lane'][0]
            output['unsampled_lane'] = data['other']['unsampled_lane'][0]
            output['traf'] = np.repeat(data['other']['traf'][0], 190, axis=0)
            output['gt_agent'] = data['other']['gt_agent'][0]
            output['gt_agent_mask'] = data['other']['gt_agent_mask'][0]

            pred_i = trafficgen.inference_control(output)

        bp_lib = CarlaDataProvider.get_world().get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.audi.a2")

        ego_loc = self.actors[self.agents[0]].get_location()
        ego_heading = math.radians(self.actors[self.agents[0]].get_transform().rotation.yaw)

        traj = []
        tm: carla.TrafficManager = self.client.get_trafficmanager(CarlaDataProvider.get_traffic_manager_port())

        def transform(position, heading):
            x, y = position[0], position[1]
            x, y = rotate(x, y, ego_heading)
            position = np.array([ego_loc.x + x, ego_loc.y + y])
            heading += ego_heading
            tf = carla.Transform(
                location=carla.Location(x=position[0], y=position[1], z=1.0),
                rotation=carla.Rotation(yaw=math.degrees(agent.heading))
            )
            return tf
        world = CarlaDataProvider.get_world()
        for i in range(1, pred_i.shape[1]):
            agent = model_output['agent'][i]
            tf = transform(agent.position[0], agent.heading[0])
            npc = CarlaDataProvider.request_new_actor("vehicle.audi.a2", tf)
            if npc is not None:
                agent_traj = []
                for t in range(pred_i.shape[0]):
                    tf = transform(pred_i[t, i, :2], pred_i[t, i, 2])
                    agent_traj.append(tf)
                npc.set_autopilot(True)
                if self._debug:
                    vecs = list(zip(agent_traj[:-1], agent_traj[1:]))
                    for i in range(0, len(vecs), 10):
                        start, end = vecs[i]
                        color = npc.attributes["color"]
                        world.debug.draw_arrow(start.location, end.location, thickness=0.1, color=carla.Color(0, 5, 0), life_time=0)
                tm.set_path(npc, [tf.location for tf in agent_traj])

    # self.place_vehicles_for_single_scenario(batch, idx, vis, init_vis_dir, context_num)

    def _get_next_object_id(self) -> int:
        self._object_id += 1
        return self._object_id

    def reset(
            self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        obs, info = super().reset(seed, options)
        map: carla.Map = CarlaDataProvider.get_map()
        tmp = tempfile.NamedTemporaryFile()
        self._network = Network.fromFile("scenarios/maps/Town05.xodr", useCache=True)
        self._lane_ids = [l.id for l in self._network.lanes]
        self._topology = map.get_topology()
        self._object_id = 0
        self.init_map()
        CarlaDataProvider.get_world().tick()
        world = CarlaDataProvider.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False
        #world.apply_settings(settings)
        return obs, info

    def step(
            self, actions: dict
    ) -> tuple[
        dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]
    ]:
        obs, *rest = super().step(actions)
        return obs, *rest
