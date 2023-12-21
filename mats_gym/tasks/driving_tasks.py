from __future__ import annotations

import logging

import numpy as np

from mats_gym.tasks import Task


class TrackRouteMinCruiseSpeed(Task):
    """
    Reward for tracking the reference route while keeping a minimal cruise speed.
    Termination on route completion.
    """

    def __init__(self, env, agent: str, target_velocity: float = 5.0, weights: list[float] = None):
        super().__init__(agent=agent)
        self._env = env
        self._target_velocity = target_velocity
        self._waypoints = None
        self._weights = weights or [1.0, 1.0, 0.5, 5.0]
        assert len(self._weights) == 4, f"expected 4 weights for (x, y, theta, v) error, got {len(self._weights)}"

    def reward(self, obs: dict, action: dict, info: dict) -> float:
        assert self._waypoints is not None, f"route is not defined for agent {self.agent}"

        position = obs[self.agent]["location"][:2]  # x, y position
        theta = np.deg2rad(obs[self.agent]["rotation"][-1])  # yaw angle in radians
        pose = np.array([position[0], position[1], theta])
        speed = obs[self.agent]["speed"]

        # process speed
        speed = min(speed, self._target_velocity)  # saturate to target speed -> not penalized for going faster

        # find 2 closest waypoints
        nearest_point, nearest_dist, t, i = self._nearest_point_on_trajectory(position, self._waypoints[:, :2])

        if i < len(self._waypoints) - 1:
            closest_waypoint = self._waypoints[i]
            next_waypoint = self._waypoints[i + 1]
        else:
            closest_waypoint = self._waypoints[i - 1]
            next_waypoint = self._waypoints[i]

        # project the 3d pose onto the line segment between the 2 closest waypoints
        reference_pose = self._point_on_line(closest_waypoint, next_waypoint, pose)

        # compute the error between the actual pose and the reference project pose
        state = np.concatenate([pose, [speed]])
        reference_state = np.concatenate([reference_pose, [self._target_velocity]])

        error = np.linalg.norm((state - reference_state) * self._weights)
        reward = np.exp(-error)

        logging.debug(
            f"agent {self.agent}: speed: {obs[self.agent]['speed']}, proc speed: {speed}, tracking error: {error:.2f}, reward: {reward:.2f}")

        return reward

    def terminated(self, obs: dict, action: dict, info: dict) -> bool:
        for event in info[self.agent]["events"]:
            if event["event"] == "ROUTE_COMPLETED":
                return True
        return False

    def reset(self) -> None:
        assert hasattr(self._env,
                       "current_scenario"), "env does not have current_scenario, move the TaskWrapper after base env"

        scenario = self._env.current_scenario
        assert scenario, "current scenario is not initialized"

        matching_vconfigs = [v_config for v_config in scenario.config.ego_vehicles if v_config.rolename == self.agent]
        assert len(matching_vconfigs) == 1, f"found {matching_vconfigs} vehicles matching agent id {self.agent}"

        route = matching_vconfigs[0].route
        assert route, f"agent {self.agent} does not have a route defined"

        xys = np.array([[wp.x, wp.y] for wp, _ in route])

        dx = xys[1:, 0] - xys[:-1, 0]
        dy = xys[1:, 1] - xys[:-1, 1]
        thetas = np.arctan2(dy, dx)
        thetas = np.hstack([thetas, thetas[-1]]).reshape(-1, 1)

        self._waypoints = np.hstack([xys, thetas])

    def _point_on_line(self, a, b, p):
        """
        From https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y
        """
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))  # if you need the the closest point belonging to the segment
        result = a + t * ab
        return result

    def _nearest_point_on_trajectory(self, point, trajectory):
        """
        From f1tenth_gym.examples.waypoint_follow
        """
        diffs = trajectory[1:, :] - trajectory[:-1, :]
        l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
        # this is equivalent to the elementwise dot product
        # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
        dots = np.empty((trajectory.shape[0] - 1,))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
        t = dots / l2s
        t[t < 0.0] = 0.0
        t[t > 1.0] = 1.0
        # t = np.clip(dots / l2s, 0.0, 1.0)
        projections = trajectory[:-1, :] + (t * diffs.T).T
        # dists = np.linalg.norm(point - projections, axis=1)
        dists = np.empty((projections.shape[0],))
        for i in range(dists.shape[0]):
            temp = point - projections[i]
            dists[i] = np.sqrt(np.sum(temp * temp))
        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


class SmoothDrivingTask(Task):
    """
    This task rewards the agent for driving smoothly.
    """

    def __init__(self, agent: str):
        super().__init__(agent)
        self._prev_action = None

    def reward(self, obs: dict, action: dict, info: dict) -> float:
        assert isinstance(action[self.agent], np.ndarray), "Action must be a numpy array"
        if self._prev_action is not None:
            scale = np.linalg.norm(np.ones(action[self.agent].shape))
            diff = self._prev_action - action[self.agent]
            norm_magnitude = np.linalg.norm(diff) / scale
            reward = np.exp(-norm_magnitude)
        else:
            reward = 0.0

        self._prev_action = action[self.agent]
        return reward

    def terminated(self, obs: dict, action: dict, info: dict) -> bool:
        return False

    def reset(self) -> None:
        self._prev_action = None


class DriveMinVelocityTask(Task):
    """
    A task that rewards the agent for driving at a minimum velocity.
    """

    def __init__(self, agent: str, target_velocity: float = 5.0, max_velocity: float = 15.0):
        """
        Constructor.

        Args:
            agent (str): Name of the agent to monitor.
            target_velocity (float, optional): Minimum velocity to drive at, in m/s.
        """
        super().__init__(agent)
        self._target_velocity = target_velocity
        self._max_velocity = max_velocity

    def reward(self, obs: dict, action: dict, info: dict) -> float:
        assert "speed" in obs[self.agent]
        speed = obs[self.agent]["speed"]

        if speed > self._max_velocity:
            reward = 0.0
        else:
            # saturate the speed signal
            speed = min(speed, self._target_velocity)

            # normalize the speed signal
            reward = speed / self._target_velocity

        return reward

    def terminated(self, obs: dict, action: dict, info: dict) -> bool:
        return False

    def reset(self) -> None:
        pass


class SmoothControlTask(Task):
    """
    A task that rewards the agent for driving without jerking.
    """

    def __init__(self, agent: str):
        super().__init__(agent)

    def reward(self, obs: dict, action: dict, info: dict) -> float:
        assert isinstance(action[self.agent], np.ndarray), "Action must be a numpy array"

        # compute the normalized action magnitude
        scale = np.linalg.norm(np.ones(action[self.agent].shape))
        norm_magnitude = np.linalg.norm(action[self.agent]) / scale

        # compute the reward
        reward = np.exp(-norm_magnitude)

        return reward

    def terminated(self, obs: dict, action: dict, info: dict) -> bool:
        return False

    def reset(self) -> None:
        pass
