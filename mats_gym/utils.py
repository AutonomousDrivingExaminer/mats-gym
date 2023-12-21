# Useful types
import typing
from typing import Union, Tuple, List

import carla
import numpy as np

Interval = Union[np.ndarray,
                 Tuple[float, float],
                 List[float]]


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def get_vehicles_by_prefix(prefix: str, world: carla.World) -> typing.List[carla.Vehicle]:
    actor_list = world.get_actors()
    vehicles = []
    for vehicle in actor_list.filter('vehicle.*'):
        if vehicle.attributes.get('role_name').startswith(prefix):
            vehicles.append(vehicle)
    return vehicles


def get_vehicle_by_rolename(rolename: str, world: carla.World) -> typing.Optional[carla.Vehicle]:
    actor_list = world.get_actors()
    for vehicle in actor_list.filter('vehicle.*'):
        if vehicle.attributes.get('role_name') == rolename:
            return vehicle
    return None


def rotation_matrix_to_euler(matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    computes a carla rotation from a rotation matrix
    """
    if matrix[0, 2] == 1 or matrix[0, 2] == -1:
        E2 = 0
        delta = np.arctan2(matrix[0, 1], matrix[0, 2])
        if matrix[0, 2] == -1:
            E1 = np.pi / 2
            E3 = E2 + delta
        else:
            E1 = -np.pi / 2
            E3 = -E2 + delta
    else:
        E1 = -np.arcsin(matrix[0, 2])
        E2 = np.arctan2(matrix[1, 2] / np.cos(E1), matrix[2, 2] / np.cos(E1))
        E3 = -np.arctan2(matrix[0, 1] / np.cos(E1), matrix[0, 0] / np.cos(E1))
    return E1, E2, E3


def get_relative_transform(source: carla.Transform, target: carla.Transform) -> carla.Transform:
    source_location = source.location
    source_rotation = source.rotation
    target_location = target.location
    target_rotation: carla.Rotation = target.rotation
    relative_position = np.array(source.get_inverse_matrix()) @ np.array(
        [target_location.x, target_location.y, target_location.z, 1])
    relative_rotation = carla.Rotation(
        pitch=target_rotation.pitch - source_rotation.pitch,
        yaw=target_rotation.yaw - source_rotation.yaw,
        roll=target_rotation.roll - source_rotation.roll
    )
    relative_position = carla.Location(
        x=relative_position[0],
        y=relative_position[1],
        z=relative_position[2]
    )
    return carla.Transform(
        location=relative_position,
        rotation=relative_rotation
    )


def transform_to_vector(transform: carla.Transform) -> np.ndarray:
    return np.array([transform.location.x, transform.location.y, transform.rotation.yaw])


def velocity_to_vector(velocity: carla.Vector3D) -> np.ndarray:
    return np.array([velocity.x, velocity.y, velocity.z])