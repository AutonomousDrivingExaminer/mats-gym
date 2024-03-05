from typing import Any

import carla
import numpy as np

from mats_gym import BaseScenarioEnv
from mats_gym.envs.replays import SimulationHistory
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper


class ReplayWrapper(BaseScenarioEnvWrapper):
    def __init__(self, env: BaseScenarioEnvWrapper | BaseScenarioEnv, replay_dir: str = "/home/carla") -> None:
        super().__init__(env)
        self._action_records = []
        self._current_episode = []
        self._replay_frames = 0
        self._current_step = 0
        self._replay_dir = replay_dir

    @property
    def num_records(self) -> int:
        return len(self._action_records)

    @property
    def history(self) -> SimulationHistory:
        """
        Returns the simulation history.
        @return: The simulation history of the current episode.
        """
        history = self.client.show_recorder_file_info(
            f"{self._replay_dir}/scenario-env.log", True
        )
        return SimulationHistory(history)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict[Any, dict]]:
        options = options or {}
        self.client.stop_recorder()
        obs, info = self.env.reset(seed=seed, options=options)
        self.client.start_recorder(f"{self._replay_dir}/scenario-env.log")
        replay_options = options.get("replay", {})
        self._current_step = 0
        if replay_options.get("reset", False):
            self._action_records = []
            self._num_episodes = 0
        elif len(self._current_episode) > 0:
            self._action_records.append(self._current_episode)

        self._current_episode = []
        self._replay_episode = replay_options.get("replay_episode", -1)
        self._replay_frames = replay_options.get("num_frames", 0)
        self._replay_actors = replay_options.get("replay_actors", list(self.env.actors.keys()))
        return obs, info

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        if self._current_step < self._replay_frames:
            controls = self._action_records[self._replay_episode][self._current_step]
            commands = []
            for name, ctrl in controls.items():
                actor = self.env.actors[name]
                if isinstance(actor, carla.Vehicle) and name not in actions:
                    commands.append(carla.command.ApplyVehicleControl(actor.id, ctrl))
            self.client.apply_batch_sync(commands)
        step = self.env.step(actions)
        self._current_episode.append({
            name: actor.get_control() for name, actor in self.env.actors.items()
        })
        self._current_step += 1
        return step
