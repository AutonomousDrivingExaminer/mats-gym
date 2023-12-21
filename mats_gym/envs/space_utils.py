import gymnasium
import numpy as np


def get_vehicle_action_space():
    return gymnasium.spaces.Box(
        low=np.array([0.0, -1.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float64,
    )


def get_walker_action_space():
    return gymnasium.spaces.Dict(
        {
            "direction": gymnasium.spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.inf]),
                high=np.array([np.inf, np.inf, np.inf]),
                dtype=np.float64,
            ),
            "speed": gymnasium.spaces.Box(0, 12.5, shape=(1,), dtype=np.float32),
            "jump": gymnasium.spaces.Discrete(2),
        }
    )
