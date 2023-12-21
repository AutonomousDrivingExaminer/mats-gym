from __future__ import annotations

import abc
from abc import ABC
from typing import Any

import carla
import gymnasium
import numpy as np
from leaderboard.envs.sensor_interface import SpeedometerReader, SensorInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from mats_gym.sensors.callbacks import Sensor, CameraCallback, LidarCallback, GnssCallback, ImuCallback, \
    PseudoSensorCallback


class SensorBuilder(ABC):
    """
    A sensor is responsible for determining the observation space and configuring the blueprint.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    @property
    def transform(self) -> carla.Transform:
        """
        Computes the relative transform to the parent actor.
        """
        return carla.Transform(
            location=carla.Location(
                x=self.spec.get("x", 0),
                y=self.spec.get("y", 0),
                z=self.spec.get("z", 0),
            ),
            rotation=carla.Rotation(
                pitch=self.spec.get("pitch", 0),
                yaw=self.spec.get("yaw", 0),
                roll=self.spec.get("roll", 0),
            ),
        )

    @abc.abstractmethod
    def observation_space(self) -> gymnasium.spaces.Space:
        """
        The observation space of the sensor.
        """
        ...

    @abc.abstractmethod
    def configure_sensor(self, vehicle: carla.Vehicle, sensor_interface: SensorInterface) -> Sensor:
        """
        Creates and configures sensor.
        """
        ...

class CameraSensorBuilder(SensorBuilder):
    def __init__(self, spec: dict[str, Any]) -> None:
        assert spec["type"].startswith("sensor.camera")
        self.width = spec.get("width", 800)
        self.height = spec.get("height", 600)
        self.fov = spec.get("fov", 100)
        super().__init__(spec)

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 4),
            dtype=np.uint8,
        )

    def configure_sensor(self, vehicle: carla.Vehicle, sensor_interface: SensorInterface) -> Sensor:
        bp_lib = CarlaDataProvider.get_world().get_blueprint_library()
        sensor_spec = self.spec
        bp = bp_lib.find(sensor_spec["type"])
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov", str(self.fov))
        sensor = CarlaDataProvider.get_world().spawn_actor(bp, self.transform, vehicle)
        sensor.listen(CameraCallback(sensor_spec['id'], self.spec["type"], sensor, sensor_interface))
        return sensor


class LidarSensorBuilder(SensorBuilder):

    def __init__(self, spec: dict[str, Any]) -> None:
        assert spec["type"].startswith("sensor.lidar")
        assert "channels" in spec and "range" in spec
        super().__init__(spec)

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=0,
            high=self.spec["range"],
            shape=(int(self.spec["channels"] / 4), 4),
            dtype=np.dtype("f4"),
        )

    def configure_sensor(self, vehicle: carla.Vehicle, sensor_interface: SensorInterface) -> Sensor:
        bp_lib = CarlaDataProvider.get_world().get_blueprint_library()
        sensor_spec = self.spec
        bp = bp_lib.find(sensor_spec["type"])
        if "channels" in sensor_spec:
            bp.set_attribute("channels", str(sensor_spec["channels"]))
        if "range" in sensor_spec:
            bp.set_attribute("range", str(sensor_spec["range"]))
        if "points_per_second" in sensor_spec:
            bp.set_attribute("points_per_second", str(sensor_spec["points_per_second"]))
        if "rotation_frequency" in sensor_spec:
            bp.set_attribute(
                "rotation_frequency", str(sensor_spec["rotation_frequency"])
            )
        if "upper_fov" in sensor_spec:
            bp.set_attribute("upper_fov", str(sensor_spec["upper_fov"]))
        if "lower_fov" in sensor_spec:
            bp.set_attribute("lower_fov", str(sensor_spec["lower_fov"]))
        sensor = CarlaDataProvider.get_world().spawn_actor(bp, self.transform, vehicle)
        sensor.listen(LidarCallback(sensor_spec['id'], self.spec["type"], sensor, sensor_interface))
        return sensor

class GnssSensorBuilder(SensorBuilder):
    def __init__(self, spec: dict[str, Any]) -> None:
        assert spec["type"].startswith("sensor.other.gnss")
        super().__init__(spec)

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
        )

    def configure_sensor(self, vehicle: carla.Vehicle, sensor_interface: SensorInterface) -> Sensor:
        bp_lib = CarlaDataProvider.get_world().get_blueprint_library()
        sensor_spec = self.spec
        bp = bp_lib.find(sensor_spec["type"])
        sensor = CarlaDataProvider.get_world().spawn_actor(bp, self.transform, vehicle)
        sensor.listen(GnssCallback(sensor_spec['id'], self.spec["type"], sensor, sensor_interface))
        return sensor

class IMUSensorBuilder(SensorBuilder):

    def __init__(self, spec: dict[str, Any]) -> None:
        assert spec["type"].startswith("sensor.other.imu")
        super().__init__(spec)

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)

    def configure_sensor(self, vehicle: carla.Vehicle, sensor_interface: SensorInterface) -> Sensor:
        bp_lib = CarlaDataProvider.get_world().get_blueprint_library()
        bp = bp_lib.find(self.spec["type"])
        bp.set_attribute("noise_accel_stddev_x", str(0.001))
        bp.set_attribute("noise_accel_stddev_y", str(0.001))
        bp.set_attribute("noise_accel_stddev_z", str(0.015))
        bp.set_attribute("noise_gyro_stddev_x", str(0.001))
        bp.set_attribute("noise_gyro_stddev_y", str(0.001))
        bp.set_attribute("noise_gyro_stddev_z", str(0.001))
        sensor = CarlaDataProvider.get_world().spawn_actor(bp, self.transform, vehicle)
        sensor.listen(ImuCallback(self.spec['id'], self.spec["type"],  sensor, sensor_interface))
        return sensor

class SpeedoMeterBuilder(SensorBuilder):

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__(spec)

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)

    def configure_sensor(self, vehicle: carla.Vehicle, sensor_interface: SensorInterface) -> Sensor:
        settings: carla.WorldSettings = CarlaDataProvider.get_world().get_settings()
        dt = 1.0 / settings.fixed_delta_seconds
        sensor = SpeedometerReader(vehicle, dt)
        sensor.listen(PseudoSensorCallback(self.spec['id'], self.spec["type"], sensor, sensor_interface))
        return sensor