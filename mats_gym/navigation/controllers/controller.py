from typing import Protocol, TypeVar

import carla

T = TypeVar('T', contravariant=True)

class Controller(Protocol[T]):

    def update(self, dt: float) -> T:
        ...


class LongitudinalController(Controller[float]):

    def set_target_speed(self, target_speed: float):
        ...

class LatitudinalController(Controller[float]):

    def set_target_waypoint(self, target_waypoint: carla.Waypoint):
        ...

class VehiclePIDController(Controller[carla.VehicleControl]):
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """


    def __init__(
            self,
            vehicle: carla.Vehicle,
            longitudinal_controller: LongitudinalController,
            latitudinal_controller: LatitudinalController,
            offset: float = 0,
            max_throttle: float = 0.75,
            max_brake: float = 0.3,
            max_steering: float = 0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param longitudinal_controller: longitudinal controller
        :param latitudinal_controller: latitudinal controller
        :param offset: If different from zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        :param max_throttle: maximum throttle applied to the vehicle (0 to 1)
        :param max_brake: maximum brake applied to the vehicle (0 to 1)
        :param max_steering: maximum steering applied to the vehicle (0 to 1)
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering
        self.offset = offset
        self.target_speed = 0.0
        self.target_waypoint = None

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = longitudinal_controller
        self._lat_controller = latitudinal_controller

    def set_target_speed(self, target_speed: float):
        self._lon_controller.set_target_speed(target_speed)

    def set_target_waypoint(self, target_waypoint: carla.Waypoint):
        self._lat_controller.set_target_waypoint(target_waypoint)

    def update(self, dt: float) -> carla.VehicleControl:
        return self.run_step(self.target_speed, self.target_waypoint)

    def run_step(self, target_speed: float, waypoint: carla.Waypoint) -> carla.VehicleControl:
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """
        self._lon_controller.set_target_speed(target_speed)
        self._lat_controller.set_target_waypoint(waypoint)

        acceleration = self._lon_controller.update(self.dt)
        current_steering = self._lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_PID(self, args_lateral):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_lateral)