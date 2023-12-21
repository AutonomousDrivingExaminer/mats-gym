from collections import deque

import carla
import numpy as np

from mats_gym.navigation.controllers.controller import Controller, T
from mats_gym.navigation.misc import get_speed

class PIDController(Controller):

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_params(self, kp: float = None, ki: float = None, kd: float = None):
        self.kp = kp or self.kp
        self.ki = ki or self.ki
        self.kd = kd or self.kd

class PIDLongitudinalController(PIDController[float]):
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(
            self,
            vehicle: carla.Vehicle,
            kp: float = 1.0,
            ki: float = 0.0,
            kd: float = 0.0,
            dt: float = 0.03
    ):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        super().__init__(kp, ki, kd)
        self._vehicle = vehicle
        self._target_speed = 0.0
        self._error_buffer = deque(maxlen=10)

    def set_target_speed(self, target_speed: float):
        self._target_speed = target_speed

    def update(self, dt: float) -> float:
        current_speed = get_speed(self._vehicle)
        error = self._target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / dt
            _ie = sum(self._error_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self.kp * error) + (self.kd * _de) + (self.ki * _ie), -1.0, 1.0).item()

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.
            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        self.set_target_speed(target_speed)

        return self.update()
