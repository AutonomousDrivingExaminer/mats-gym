from __future__ import annotations

import tempfile
from typing import Any

import carla
import gymnasium
import gymnasium.spaces
import numpy as np
from scenic.domains.driving.roads import Network
from scenic.formats.opendrive import xodr_parser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper


class RoadGraphObservationWrapper(BaseScenarioEnvWrapper):
    """
    Wrapper to add road information to the observation.
    Road information includes:
    - Identification of the current lane (road, section, lane)
    - Lane type and width
    - Lane change possibility
    """

    def __init__(self, env: BaseScenarioEnvWrapper, max_samples: int = 20000, max_radius: float = 100.0):
        """
        @param env: The environment to wrap.
        """
        self._max_samples = max_samples
        self._max_radius = max_radius
        self._network: Network = None
        super().__init__(env)


    def observation_space(self, agent: str) -> gymnasium.spaces.Dict:
        obs_space: gymnasium.spaces.Dict = self.env.observation_space(agent)
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

    def _get_center_lines(self, location: carla.Location, map: carla.Map) -> tuple[np.ndarray, np.ndarray]:
        center_lines, types = [], []
        topology = map.get_topology()
        for start, end in topology:
            start_loc = start.transform.location
            end_loc = end.transform.location
            if location.distance(start_loc) < self._max_radius or location.distance(end_loc) < self._max_radius:
                waypoints = start.next_until_lane_end(1.0)
                xyz, type = [], []
                for waypoint in filter(lambda w: location.distance(w.transform.location) < self._max_radius, waypoints):
                    xyz = np.array([
                        waypoint.transform.location.x,
                        waypoint.transform.location.y,
                        waypoint.transform.location.z
                    ])
                    wp: carla.Waypoint = waypoint
                    landmarks = wp.get_landmarks(100.0)
                    for landmark in filter(lambda l: l.type == carla.LandmarkType.MaximumSpeed, landmarks):
                        for r in landmark.get_lane_validities():
                            if r[0] <= wp.lane_id <= r[1]:
                                speed_limit = landmark.value
                                return speed_limit




                xyz = np.array([
                    [w.transform.location.x, w.transform.location.y, w.transform.location.z]
                    for w in waypoints
                    if location.distance(w.transform.location) < self._max_radius
                ])
                dir = (xyz[1:] - xyz[:-1])
                dir = dir / np.linalg.norm(dir, axis=1, keepdims=True)
                type = np.zeros((dir.shape[0], 19), dtype=np.int32)

                match waypoints[0].type:
                    case carla.LaneType.Driving | carla.LaneType.Bidirectional:
                        type[:, 1] = 1
                    case carla.LaneType.Entry | carla.LaneType.Exit | carla.LaneType.OffRamp | carla.LaneType.OnRamp:
                        type[:, 2] = 1
                center_lines.append(center_line)
                wp: carla.Waypoint = waypoints[-1]
        return np.concatenate(center_lines, axis=0)

    def _get_crosswalks(self, location: carla.Location, map: carla.Map) -> np.ndarray:
        crosswalks = []
        crosswalk_points = map.get_crosswalks()
        for idx in range(len(crosswalk_points), 5):
            crosswalk = crosswalk_points[idx:idx+4]
            if all(location.distance(l) < self._max_radius for l in crosswalk):
                polygon = np.array([
                    [l.x, l.y, l.z]
                    for l in crosswalk
                ])
                crosswalks.append(polygon)
        return np.concatenate(crosswalks, axis=0)


    def observation(self, observation: dict) -> dict:
        map = CarlaDataProvider.get_map()
        for agent in observation:
            location = self.env.actors[agent].get_location()
            for lane in self._network.lanes:
                lane.distanceTo
            roadgraph = {
                "xyz": np.zeros((self._max_samples, 3), dtype=np.float32),
                "dir": np.zeros((self._max_samples, 3), dtype=np.float32),
                "type": np.zeros((self._max_samples, 19), dtype=np.int32),
                "valid": np.zeros((self._max_samples,), dtype=np.int32),
                "id": np.zeros((self._max_samples,), dtype=np.int32) - 1,
            }
            center_lines = self._get_center_lines(location, map)
            roadgraph["xyz"][:center_lines.shape[0]] = center_lines
            roadgraph["valid"][:center_lines.shape[0]] = 1
            roadgraph["type"][:center_lines.shape[0], 0] = 1


            waypoint: carla.Waypoint = map.get_waypoint(location)
            observation[agent]["lane_id"] = waypoint.lane_id
            observation[agent]["road_id"] = waypoint.road_id
            observation[agent]["section_id"] = waypoint.section_id
            observation[agent]["on_junction"] = waypoint.is_junction
            observation[agent]["lane_width"] = waypoint.lane_width
            observation[agent]["lane_type"] = self._lane_type_values[waypoint.lane_type]
            observation[agent]["lane_change"] = self._lane_change[waypoint.lane_change]
        return observation

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        obs, info = super().reset(seed, options)
        map: carla.Map = CarlaDataProvider.get_map()
        network: xodr_parser.RoadMap = xodr_parser.RoadMap()
        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, "w") as f:
            f.write(map.to_opendrive())
        network.parse(tmp.name)
        network.calculate_geometry(num=20)
        return self.observation(obs), info

    def step(
        self, actions: dict
    ) -> tuple[
        dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]
    ]:
        obs, *rest = super().step(actions)
        return self.observation(obs), *rest