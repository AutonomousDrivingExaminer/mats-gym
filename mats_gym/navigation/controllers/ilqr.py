from dataclasses import dataclass

import carla
import numpy as np

from mats_gym.navigation.controllers.controller import Controller, T
from mats_gym.navigation.controllers.pid import PIDController
from mats_gym.navigation.misc import get_speed
from casadi import *

X = 0
Y = 1
PSI = 2
V = 3
CTE = 4
E_PSI = 5
A = 0
DELTA = 1

class NonlinearMPC(Controller):

    @dataclass
    class Parameters:
        horizon: int = 10
        dt: float = 0.05
        w_cte: float = 50.0
        w_eth: float = 100.0


    def __init__(self, vehicle: carla.Vehicle, params = Parameters()) -> None:
        self._vehicle = vehicle
        self._target_speed = 0.0
        self._target_waypoint = None
        self._acceleration_pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        self._steering_pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        self._params = params

    def _dynamics(self, x, u):
        # x = [x, y, psi, v, cte, e_psi]
        # u = [a, delta]
        a = u[A]
        delta = u[DELTA]
        x_dot = x[V] * cos(x[PSI])
        y_dot = x[V] * sin(x[PSI])
        psi_dot = x[V] * tan(delta)
        v_dot = a
        cte_dot = x[V] * sin(x[PSI] - atan(x[CTE]))
        e_psi_dot = psi_dot - atan(x[CTE])
        return vertcat(x_dot, y_dot, psi_dot, v_dot, cte_dot, e_psi_dot)



    def _get_state(self) -> np.ndarray:
        location = self._vehicle.get_location()
        rotation = self._vehicle.get_transform().rotation
        return np.array([
            location.x,
            location.y,
            rotation.yaw,
            get_speed(self._vehicle) / 3.6,

        ])




    def update(self, dt: float) -> T:
        opti = Opti()
        T = self._params.horizon


        x = opti.variable(6, T)
        u = opti.variable(2, T)
        for t in range(T-1):
            opti.subject_to(x[:, t+1] == self._dynamics(x[:, t], u[:, t], self._params.dt))


        return super().update(dt)