from __future__ import annotations

import abc
import copy
from abc import ABC
from typing import Protocol, Any, Generic, TypeVar

import carla
import numpy as np
from leaderboard.envs.sensor_interface import GenericMeasurement, SensorInterface

T = TypeVar("T")


class Sensor(Protocol):

    def listen(self, callback: SensorCallBack) -> None:
        pass

    def destroy(self) -> None:
        pass

    def stop(self) -> None:
        pass


class SensorCallBack(ABC, Generic[T]):

    def __init__(self, tag: str, type: str, sensor: Sensor, sensor_interface: SensorInterface) -> None:
        self._tag = tag
        self._sensor_interface = sensor_interface
        self._sensor_interface.register_sensor(tag, type, sensor)

    def update(self, data: Any, frame: int) -> None:
        self._sensor_interface.update_sensor(self._tag, data, frame)

    @abc.abstractmethod
    def __call__(self, data: T) -> None:
        ...


class CameraCallback(SensorCallBack[carla.Image]):

    def __call__(self, data: carla.Image) -> None:
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (data.height, data.width, 4))
        self.update(array, data.frame)


class LidarCallback(SensorCallBack[carla.LidarMeasurement]):

    def __call__(self, data: carla.LidarMeasurement) -> None:
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.update(points, data.frame)


class RadarCallback(SensorCallBack[carla.RadarMeasurement]):

    def __call__(self, data: carla.RadarMeasurement) -> None:
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self.update(points, data.frame)


class GnssCallback(SensorCallBack[carla.GnssMeasurement]):

    def __call__(self, data: carla.GnssMeasurement) -> None:
        array = np.array([data.latitude, data.longitude, data.altitude], dtype=np.float64)
        self.update(array, data.frame)


class ImuCallback(SensorCallBack[carla.IMUMeasurement]):

    def __call__(self, data: carla.IMUMeasurement) -> None:
        array = np.array([
            data.accelerometer.x,
            data.accelerometer.y,
            data.accelerometer.z,
            data.gyroscope.x,
            data.gyroscope.y,
            data.gyroscope.z,
            data.compass
        ], dtype=np.float64)
        self.update(array, data.frame)


class CollisionCallback(SensorCallBack[carla.CollisionEvent]):

    def __call__(self, data: carla.CollisionEvent) -> None:
        other_actor_id = data.other_actor.id
        other_location = data.other_actor.get_location()
        self.update({
            "other_actor_id": other_actor_id,
            "impulse": np.array([data.normal_impulse.x, data.normal_impulse.y, data.normal_impulse.z],
                                dtype=np.float32),
            "other_location": np.array([other_location.x, other_location.y, other_location.z], dtype=np.float32)
        }, data.frame)


class LaneInvasionCallback(SensorCallBack[carla.LaneInvasionEvent]):
    def __call__(self, data: carla.LaneInvasionEvent) -> None:
        crossings = []
        for crossing in data.crossed_lane_markings:
            crossings.append({
                "type": carla.LaneMarkingType.values[crossing.type],
                "location": np.array(
                    [crossing.transform.location.x, crossing.transform.location.y, crossing.transform.location.z],
                    dtype=np.float32)
            })
        self.update(crossings, data.frame)


class ObstacleDetectionCallback(SensorCallBack[carla.ObstacleDetectionEvent]):
    def __call__(self, data: carla.ObstacleDetectionEvent) -> None:
        obstacle_location = data.other_actor.get_location()
        self.update({
            "obstacle_id": data.other_actor.id,
            "obstacle_location": np.array([obstacle_location.x, obstacle_location.y, obstacle_location.z], dtype=np.float32),
            "distance": data.distance
        }, data.frame)


class PseudoSensorCallback(SensorCallBack[GenericMeasurement]):

    def __call__(self, data: GenericMeasurement) -> None:
        self.update(data.data, data.frame)
