from __future__ import annotations

import logging
from typing import Any

import carla
from gymnasium import spaces
from leaderboard.envs.sensor_interface import SensorInterface


from mats_gym.sensors.builders import (
    SpeedoMeterBuilder,
    CameraSensorBuilder,
    LidarSensorBuilder,
    GnssSensorBuilder,
    IMUSensorBuilder,
    SensorBuilder,
)
from mats_gym.sensors.callbacks import SensorCallBack


class SensorSuite:
    """
    A collection of sensors. Provides the observation space and spawns the sensors.
    """

    def __init__(self, sensor_specs: list[dict[str, Any]]):
        self._vehicle = None
        self._specs = sensor_specs
        self._sensor_builders = {
            spec["id"]: self._make_sensor_builder(spec) for spec in sensor_specs
        }
        self._carla_sensors = []
        self._sensor_interface = SensorInterface()

    def _make_sensor_builder(self, spec: dict[str, Any]) -> SensorBuilder:
        if spec["type"].startswith("sensor.camera"):
            return CameraSensorBuilder(spec)
        elif spec["type"].startswith("sensor.lidar"):
            return LidarSensorBuilder(spec)
        elif spec["type"].startswith("sensor.other.gnss"):
            return GnssSensorBuilder(spec)
        elif spec["type"].startswith("sensor.other.imu"):
            return IMUSensorBuilder(spec)
        elif spec["type"].startswith("sensor.speedometer"):
            return SpeedoMeterBuilder(spec)
        else:
            raise NotImplementedError(f"Sensor type {spec['type']} not implemented")

    def get_observations(self) -> dict[str, Any]:
        """
        Returns the observations of all sensors combined in a dictionary.
        :
        """
        return self._sensor_interface.get_data()

    @property
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                id: sensor.observation_space
                for id, sensor in self._sensor_builders.items()
            }
        )

    def _make_callback(self, type: str) -> SensorCallBack:
        pass

    def setup_sensors(self, vehicle: carla.Actor):
        """
        Spawns the sensors based on the blueprints and attaches them to the vehicle.
        :param vehicle: The vehicle to attach the sensors to.
        """
        for id, builder in self._sensor_builders.items():
            logging.debug(f"Spawning sensor {id} of type {builder.spec['type']}.")
            sensor = builder.configure_sensor(
                vehicle=vehicle, sensor_interface=self._sensor_interface
            )
            self._carla_sensors.append(sensor)

    def cleanup(self):
        """
        Remove and destroy all sensors.
        """
        for sensor in self._carla_sensors:
            sensor.stop()
            sensor.destroy()
        self._carla_sensors.clear()
