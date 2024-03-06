from __future__ import annotations

import copy
import enum
from typing import Any

import carla
import gymnasium
import gymnasium.spaces
import numpy as np
from scenic.domains.driving.roads import Network
from scenic.simulators.carla.utils.utils import scenicToCarlaLocation
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from mats_gym import BaseScenarioEnv
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper


class RoadGraphTypes(enum.Enum):
    UNKNOWN = 0
    LANE_SURFACE_STREET = 1
    LANE_BIKE_LANE = 2
    ROAD_LINE_BROKEN_SINGLE_WHITE = 3
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 4
    ROAD_LINE_SOLID_SINGLE_WHITE = 5
    ROAD_LINE_SOLID_SINGLE_YELLOW = 6
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 7
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 9
    ROAD_EDGE_BOUNDARY = 10
    ROAD_EDGE_MEDIAN = 11
    CROSSWALK = 12


class RoadGraphObservationWrapper(BaseScenarioEnvWrapper):
    """
    A wrapper that adds a roadgraph observation to the environment's observation space.
    A roadgraph is a set of lines that represent the road network relative to the agent's position and
    orientation. The roadgraph is a dictionary with the following keys:
    - "xyz": The (x, y, z) coordinates of the points in the roadgraph.
    - "dir": The (x, y, z) directions of the points in the roadgraph.
    - "type": The type of each point in the roadgraph.
    - "valid": A boolean indicating whether each point in the roadgraph is valid.
    - "id": The id of each line in the roadgraph. Points with the same id belong to the same line.
    - "lane_ids": The ids of the lanes in the roadgraph.
    @param env: The environment to wrap.
    @param max_samples: The maximum number of samples to include in the roadgraph.
    @param sampling_resolution: The distance between samples in the roadgraph.
    """

    def __init__(self, env: BaseScenarioEnv | BaseScenarioEnvWrapper, max_samples: int, sampling_resolution: float):
        """
        @param env: The environment to wrap.
        """
        super().__init__(env)
        self._max_samples = max_samples
        self._sampling_resolution = sampling_resolution
        self._road_graphs = {}

    def observation_space(self, agent: str) -> gymnasium.spaces.Dict:
        obs_space: gymnasium.spaces.Dict = self.env.observation_space(agent)
        obs_space["roadgraph"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._max_samples, 4)
        )
        return obs_space

    def observation(self, observation: dict) -> dict:
        for agent in observation:
            tf = self.env.actors[agent].get_transform()
            obs = self._get_roadgraph_samples(tf, self._max_samples)
            observation[agent]["roadgraph"] = obs
        return observation

    def _get_roadgraph_samples(self, tf: carla.Transform, max_samples: int, radius: float = None) -> np.ndarray:
        map = CarlaDataProvider.get_map()
        road_graph = copy.deepcopy(self._road_graphs[map.name])
        lane_ids = road_graph.pop("lane_ids")
        ids = road_graph["id"]

        pos = np.array([tf.location.x, tf.location.y, tf.location.z])
        headings = np.deg2rad(tf.rotation.yaw)

        if radius is not None:
            indices = np.linalg.norm(road_graph["xyz"] - tf.location, axis=1) < radius
            road_graph = {k: v[indices] for k, v in road_graph.items()}

        xyz = road_graph["xyz"]
        if xyz.shape[0] > max_samples:
            dists = np.linalg.norm(xyz - pos, axis=1)
            idxs = np.argsort(dists)[:max_samples]
            idxs.sort()
            for k, feats in road_graph.items():
                road_graph[k] = feats[idxs]

        new_lane_ids = []
        for new_id, old_id in enumerate(sorted(np.unique(ids))):
            road_graph["id"][road_graph["id"] == old_id] = new_id
            if old_id < len(lane_ids):
                new_lane_ids.append(lane_ids[old_id])

        road_graph["lane_ids"] = new_lane_ids
        xyz = road_graph["xyz"] - pos
        dir = road_graph["dir"]
        x = xyz[:, 0] * np.cos(headings) - xyz[:, 1] * np.sin(headings)
        y = xyz[:, 0] * np.sin(headings) + xyz[:, 1] * np.cos(headings)
        z = xyz[:, 2]
        dx = dir[:, 0] * np.cos(headings) - dir[:, 1] * np.sin(headings)
        dy = dir[:, 0] * np.sin(headings) + dir[:, 1] * np.cos(headings)
        dz = dir[:, 2]
        xyz = np.stack([x, y, z], axis=1)
        dir = np.stack([dx, dy, dz], axis=1)
        road_graph["xyz"] = xyz
        road_graph["dir"] = dir
        return road_graph


    def reset(
            self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        map = CarlaDataProvider.get_map()
        if map.name not in self._road_graphs:
            self._road_graphs[map.name] = self._parse_roadgraph(map, resolution=self._sampling_resolution)

        return self.observation(obs), info

    def step(
            self, actions: dict
    ) -> tuple[
        dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]
    ]:
        obs, *rest = super().step(actions)
        return self.observation(obs), *rest

    def _parse_roadgraph(self, map: carla.Map, resolution: float):
        world = CarlaDataProvider.get_world()
        topology = map.get_topology()
        town = map.name.split("/")[-1]
        dir, id, type, valid, xyz = [], [], [], [], []
        num_features = 0
        lanes = self._get_centerlines(topology=topology, resolution=resolution)
        lane_ids = list(sorted(lanes.keys()))

        elements, ids = [], []
        elements.extend(lanes[id] for id in lane_ids)
        elements.extend(self._get_road_edges(
            map=map,
            world=world,
            resolution=resolution
        ))
        elements.extend(self._get_crosswalks(map=map))
        elements.extend(self._get_road_markings(topology=topology, resolution=resolution))

        for line in elements:
            assert line["xyz"].shape[0] == line["dir"].shape[0]
            xyz.append(line["xyz"])
            type.append(line["type"].reshape(-1, 1))
            valid.append(line["valid"].reshape(-1, 1))
            dir.append(line["dir"])
            length = line["type"].shape[0]
            id.append(np.full([length, 1], num_features, dtype=np.int64))
            num_features += 1

        road_graph = {
            "dir": np.concatenate(dir, axis=0),
            "id": np.concatenate(id, axis=0),
            "type": np.concatenate(type, axis=0),
            "valid": np.concatenate(valid, axis=0),
            "xyz": np.concatenate(xyz, axis=0),
            "lane_ids": lane_ids
        }
        road_graph["lane_ids"] = lane_ids
        return road_graph

    def _get_centerlines(self, topology, resolution: float) -> dict[tuple, dict[str, np.ndarray]]:
        centerlines = {}
        for start, _ in topology:
            types, xyz = [], []
            prev = None
            dist = resolution
            previous_wps = start.previous(dist)
            while len(previous_wps) > 0 and dist > 0:
                previous_wps = start.previous(dist)
                prev = min(previous_wps,
                           key=lambda wp: wp.transform.location.distance(start.transform.location))
                dist -= 0.5

            wps = start.next_until_lane_end(resolution)
            if prev is not None:
                wps = [prev, *wps]
            for wp in wps:
                if wp.lane_type == carla.LaneType.Driving:
                    type = RoadGraphTypes.LANE_SURFACE_STREET
                elif wp.lane_type == carla.LaneType.Biking:
                    type = RoadGraphTypes.LANE_BIKE_LANE
                else:
                    type = RoadGraphTypes.UNKNOWN
                location = wp.transform.location
                xyz.append([location.x, location.y, location.z])
                types.append(type.value)
            xyz = np.array(xyz, dtype=np.float32)
            types = np.array(types, dtype=np.int64)
            valid = np.ones_like(types, dtype=np.int64)
            dir = (xyz[1:] - xyz[:-1]) / (
                    np.linalg.norm(xyz[1:] - xyz[:-1], axis=1, keepdims=True) + 1e-6)

            dir = np.concatenate([dir, dir[-1:]], axis=0)
            id = 100 * start.road_id + start.lane_id
            centerlines[id] = {
                "dir": dir,
                "type": types,
                "valid": valid,
                "xyz": xyz
            }
        return centerlines

    def _get_road_markings(self, topology, resolution: float) -> list[dict[str, np.ndarray]]:
        def get_type(type, color) -> RoadGraphTypes:
            rg_type = RoadGraphTypes.UNKNOWN
            if type == carla.LaneMarkingType.Broken:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_SINGLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_SINGLE_YELLOW
            elif type == carla.LaneMarkingType.Solid:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_SINGLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_SINGLE_YELLOW
            elif type == carla.LaneMarkingType.BrokenBroken:
                if color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_DOUBLE_YELLOW
            elif type == carla.LaneMarkingType.SolidSolid:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_DOUBLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_DOUBLE_YELLOW
            return rg_type

        def extract_line(line: list[carla.Waypoint]):
            return np.stack([
                [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                for wp in line
            ])

        def parallel_line(line: list[carla.Waypoint], left: bool):
            widths = np.array([wp.lane_width for wp in line])
            line = extract_line(line)
            dir = (line[1:] - line[:-1]) / np.linalg.norm(line[1:] - line[:-1], axis=1,
                                                          keepdims=True)
            dir = np.concatenate([dir, dir[-1:]], axis=0)
            angle = np.arctan2(dir[:, 1], dir[:, 0])
            offset = widths * (1 if left else -1) * 0.5
            shift = np.stack([
                offset * np.sin(angle),
                offset * -np.cos(angle),
                np.zeros_like(angle)
            ], axis=1)
            return line + shift, dir

        markings = []
        lanes = []
        # for road in self._network.roads:
        #     lanes.extend(road.lanes)
        # for intersection in self._network.intersections:
        #     for maneuver in filter(lambda m: m.type == ManeuverType.STRAIGHT, intersection.maneuvers):
        #         if maneuver.connectingLane is not None:
        #             lanes.append(maneuver.connectingLane)
        # lanes = [
        #    [self._map.get_waypoint(scenicToCarlaLocation(p, world=world)) for p in lane]
        #    for lane in lanes
        # ]

        for start, _ in topology:
            lanes.append([
                wp for wp in start.next_until_lane_end(resolution)
                if not wp.is_intersection
            ])

        for lane in lanes:
            centerline = lane
            if len(centerline) < 2:
                continue

            left = [
                wp for wp in centerline
                if wp.left_lane_marking.type != carla.LaneMarkingType.NONE
            ]

            right = [
                wp for wp in centerline
                if wp.right_lane_marking.type != carla.LaneMarkingType.NONE
            ]

            if len(left) > 1:
                xyz, dir = parallel_line(left, True)
                types = np.array([
                    get_type(wp.left_lane_marking.type, wp.left_lane_marking.color).value
                    for wp in left
                ], dtype=np.int64)
                markings.append({
                    "dir": dir,
                    "type": types,
                    "valid": np.ones_like(types, dtype=np.int64),
                    "xyz": xyz
                })

            if len(right) > 1:
                xyz, dir = parallel_line(right, False)
                types = np.array([
                    get_type(wp.right_lane_marking.type, wp.right_lane_marking.color).value
                    for wp in right
                ], dtype=np.int64)
                markings.append({
                    "dir": dir,
                    "type": types,
                    "valid": np.ones_like(types, dtype=np.int64),
                    "xyz": xyz
                })
        return markings

    def _get_crosswalks(self, map: carla.Map) -> list[dict[str, np.ndarray]]:
        crosswalks = map.get_crosswalks()
        crosswalk_encodings = []
        idx = 0
        while idx < len(crosswalks) - 4:
            start = idx
            idx += 1
            while crosswalks[idx] != crosswalks[start]:
                idx += 1
            end = idx
            idx += 1
            # crosswalks are 5 points, first point is repeated at the end
            crosswalk = crosswalks[start:end + 1]
            xyz = np.array([
                [loc.x, loc.y, loc.z + 0.1]
                for loc in crosswalk
            ], dtype=np.float32)
            dir = (xyz[1:] - xyz[:-1]) / np.linalg.norm(xyz[1:] - xyz[:-1], axis=1, keepdims=True)
            xyz = xyz[:-1]
            type = np.full((xyz.shape[0],), RoadGraphTypes.CROSSWALK.value, dtype=np.int64)
            valid = np.ones_like(type, dtype=np.int64)
            crosswalk_encodings.append({
                "dir": dir,
                "type": type,
                "valid": valid,
                "xyz": xyz
            })

        return crosswalk_encodings

    def _get_road_edges(self, map: carla.Map, world: carla.World, resolution: float) -> list[dict[str, np.ndarray]]:
        lines = []
        network = Network.fromFile(f"scenarios/maps/{map.name.split('/')[-1]}.xodr", useCache=True)
        for road in network.roads:
            edges = [
                (road.leftEdge, RoadGraphTypes.ROAD_EDGE_BOUNDARY),
                (road.rightEdge, RoadGraphTypes.ROAD_EDGE_BOUNDARY),
                (road.centerline, RoadGraphTypes.ROAD_EDGE_MEDIAN)
            ]
            for edge, type in edges:
                line = []
                for point in edge.pointsSeparatedBy(resolution):
                    loc = scenicToCarlaLocation(point, world=world)
                    if len(line) > 0 and abs(line[-1][2] - loc.z) > 0.5:
                        loc.z = line[-1][2]
                    line.append([loc.x, loc.y, loc.z])
                if len(line) < 2:
                    continue
                xyz = np.array(line, dtype=np.float32)
                type = np.full([xyz.shape[0], 1], type.value, dtype=np.int64)
                valid = np.ones_like(type, dtype=np.int64)
                dir = (xyz[1:] - xyz[:-1]) / np.linalg.norm(xyz[1:] - xyz[:-1], axis=1,
                                                            keepdims=True)
                dir = np.concatenate([dir, dir[-1:]], axis=0)
                lines.append({
                    "dir": dir,
                    "type": type,
                    "valid": valid,
                    "xyz": xyz
                })
        return lines
