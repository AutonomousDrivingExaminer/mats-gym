from __future__ import annotations

import copy
import logging
import sys
from functools import partial
from typing import Callable

import gymnasium
import multiprocess
import numpy as np
from gymnasium.vector.utils import clear_mpi_env_vars, CloudpickleWrapper
from multiprocess import Queue, Process, Pipe
from multiprocess.connection import Connection
from pettingzoo.utils.env import AgentID, ObsType, ActionType

import mats_gym
from mats_gym.envs import renderers
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.wrappers import ServerWrapper


def worker(
    index: int,
    env_fn: Callable[[], BaseScenarioEnvWrapper],
    pipe: Connection,
    parent_pipe: Connection,
    error_queue: Queue,
):
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, args = pipe.recv()
            logging.debug(f"Worker {index} received '{command}' command.")
            if command == "reset":
                obs, info = env.reset(**args)
                logging.debug(f"Worker {index} finished reset.")
                pipe.send(((obs, info), index, True))
            elif command == "action_space":
                action_space = env.action_space(args)
                logging.debug(f"Worker {index} retrieved action space.")
                pipe.send((action_space, index, True))
            elif command == "observation_space":
                observation_space = env.observation_space(args)
                logging.debug(f"Worker {index} retrieved observation space.")
                pipe.send((observation_space, index, True))
            elif command == "step":
                obs, reward, terminated, truncated, info = env.step(args)
                pipe.send(((obs, reward, terminated, truncated, info), index, True))
            elif command == "render":
                img = env.render()
                logging.debug(f"Worker {index} rendered image.")
                pipe.send((img, index, True))
            elif command == "close":
                break
            elif command == "_getattr":
                name, args, kwargs = args
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), index, True))
                else:
                    pipe.send((function, index, True))
            elif command == "_setattr":
                name, value = args
                setattr(env, name, value)
                pipe.send((None, index, True))
            else:
                raise RuntimeError(f"Unknown command {command}")
    except (KeyboardInterrupt, Exception):
        logging.debug(f"Worker {index} failed. Sending error message.")
        logging.debug(f"Worker {index} error: {sys.exc_info()[:2]}")
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, index, False))
    finally:
        env.close()
        logging.debug(f"Worker {index} closed environment. Exiting.")


class VecEnvWrapper(BaseScenarioEnvWrapper):
    """
    A wrapper that wraps a list of environments and runs them in parallel.
    Work in progress.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], BaseScenarioEnvWrapper]],
        termination_fn: Callable[
            [dict[AgentID, bool], dict[AgentID, bool]], bool
        ] = None,
        daemon: bool = False,
        num_tries: int = 5,
        timeout: float = 30.0,
    ):
        """
        @param env_fns: A list of environment constructors.
        @param termination_fn: A function that determines when to terminate an episode.
        @param daemon: Whether to run the workers as daemons.
        @param num_tries: The number of times to try to reset a worker before giving up.
        @param timeout: The timeout for communication with the workers.
        """
        if termination_fn is None:
            termination_fn = lambda terminated, truncated: all(
                terminated.values()
            ) or all(truncated.values())
        self.metadata = {}
        self.num_envs = len(env_fns)
        self.parent_pipes, self.processes = [], []
        self.timeout = timeout
        self.error_queue = multiprocess.Queue()
        self.env_fns = env_fns
        self.termination_fn = termination_fn
        self.ctx = ""
        self.daemon = daemon
        self.num_tries = num_tries
        self._last_steps = [None for _ in range(self.num_envs)]
        self._options = [None for _ in range(self.num_envs)]
        self._seeds = [None for _ in range(self.num_envs)]
        self._action_spaces, self._observation_spaces, self._agents = {}, {}, {}
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = Pipe()
                process = Process(
                    target=worker,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        self.error_queue,
                    ),
                )
                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)
                process.daemon = daemon
                process.start()
                child_pipe.close()

    def seed(self, seed: int | list[int]):
        if isinstance(seed, int):
            seed = [seed for _ in range(self.num_envs)]
        assert (
            len(seed) == self.num_envs
        ), f"Expected {self.num_envs} seeds, got {len(seed)}."
        self._seeds = seed

    def update_options(self, options: dict | list[dict]):
        if isinstance(options, dict):
            options = [options for _ in range(self.num_envs)]
        assert (
            len(options) == self.num_envs
        ), f"Expected {self.num_envs} options, got {len(options)}."
        self._options = options

    def _spawn_envs(self, indices: list[int]):
        for idx in indices:
            logging.debug(f"Restarting worker {idx}.")
            self.parent_pipes[idx].close()
            self.processes[idx].terminate()
            env_fn = self.env_fns[idx]
            parent_pipe, child_pipe = Pipe()
            process = Process(
                target=worker,
                name=f"Worker<{type(self).__name__}>-{idx}",
                args=(
                    idx,
                    CloudpickleWrapper(env_fn),
                    child_pipe,
                    parent_pipe,
                    self.error_queue,
                ),
            )
            self.parent_pipes[idx] = parent_pipe
            self.processes[idx] = process
            process.daemon = self.daemon
            process.start()
            child_pipe.close()

    @property
    def agents(self, env_idx: int = 0) -> list[AgentID]:
        self.parent_pipes[env_idx].send(("_getattr", ("agents", [], {})))
        result, index, success = self._receive_results(
            self.parent_pipes[env_idx], timeout=self.timeout
        )
        assert result is not None, f"Could not get agent list."
        return result

    def single_action_space(
        self, agent: AgentID, idx: int = 0
    ) -> gymnasium.spaces.Space:
        self.parent_pipes[idx].send(("action_space", agent))
        result, index, success = self._receive_results(
            self.parent_pipes[idx], timeout=self.timeout
        )
        assert result is not None, f"Could not get action space for agent {agent}."
        return result

    def single_observation_space(
        self, agent: AgentID, idx: int = 0
    ) -> gymnasium.spaces.Space:
        self.parent_pipes[idx].send(("observation_space", agent))
        result, index, success = self._receive_results(
            self.parent_pipes[idx], timeout=self.timeout
        )
        assert result is not None, f"Could not get observation space for agent {agent}."
        return result

    def action_space(self, agent: list[AgentID]) -> gymnasium.spaces.Space:
        for pipe, a in zip(self.parent_pipes, agent):
            pipe.send(("action_space", a))
        results = self._receive_results(self.parent_pipes, timeout=self.timeout)
        assert all(results), f"Could not get action space for agents {agent}."
        return gymnasium.spaces.Tuple(results)

    def observation_space(self, agent: list[AgentID]) -> gymnasium.spaces.Space:
        for pipe, a in zip(self.parent_pipes, agent):
            pipe.send(("observation_space", a))
        results = self._receive_results(self.parent_pipes, timeout=self.timeout)
        assert all(results), f"Could not get observation space for agents {agent}."
        return gymnasium.spaces.Tuple(results)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        seed = seed or [None for _ in range(self.num_envs)]
        options = options or [None for _ in range(self.num_envs)]
        seedv, optionsv = [], []
        for i in range(self.num_envs):
            seedv.append(seed[i] or self._seeds[i])
            optionsv.append(options[i] or self._options[i])
        indices = list(range(self.num_envs))
        obs, info = self._try_reset(indices=indices, options=optionsv, seeds=seedv)
        self._last_steps = [(obs[i], info[i]) for i in range(self.num_envs)]
        return obs, info

    def step(self, actions: list[dict[AgentID, ActionType]]) -> list[tuple]:
        logging.debug(f"Sending {len(actions)} actions to workers.")
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))

        failed, results = [], []
        for idx, pipe in enumerate(self.parent_pipes):
            if pipe.poll(timeout=self.timeout):
                logging.debug(f"Received result from worker {idx}.")
                try:
                    results.append(pipe.recv())
                    logging.debug(f"Finished reading result from worker {idx}.")
                except EOFError:
                    logging.debug(f"Worker {idx} failed: EOF")
                    failed.append(idx)
            else:
                logging.debug(f"Stepping worker {idx} timed out.")
                failed.append(idx)

        result, index, success = zip(*results)

        errors = self.get_errors()
        needs_reset = []
        obs, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        for result, idx, success in results:
            if success:
                logging.debug(f"Worker {idx} succeeded.")
                (
                    obs[idx],
                    rewards[idx],
                    terminated[idx],
                    truncated[idx],
                    info[idx],
                ) = result
                if self.termination_fn(terminated[idx], truncated[idx]):
                    logging.debug(
                        f"Worker {index} detected termination condition. Resetting."
                    )
                    needs_reset.append(idx)
            else:
                logging.debug(f"Worker {idx} failed. Error: {errors.get(idx, None)}.")
                failed.append(idx)

        if len(failed) > 0:
            logging.debug(f"Restarting {len(failed)} workers.")
            self._spawn_envs(failed)

        resets = needs_reset + failed
        next_obs, next_info = self._try_reset(
            indices=needs_reset,
            seeds=[self._seeds[i] for i in resets],
            options=[self._options[i] for i in resets],
        )

        for idx, o, i in zip(resets, next_obs, next_info):
            if idx in failed:
                prev_obs, prev_info = self._last_steps[idx]
                truncated[idx] = {agent: True for agent in prev_obs.keys()}
                terminated[idx] = {agent: False for agent in prev_obs.keys()}
                rewards[idx] = {agent: 0.0 for agent in prev_obs.keys()}
            else:
                prev_obs, prev_info = obs[idx], info[idx]

            i["__final__"] = {
                "obs": copy.deepcopy(prev_obs),
                "info": copy.deepcopy(prev_info),
            }
            obs[idx] = o
            info[idx] = i

        obs = [obs[i] for i in range(self.num_envs)]
        rewards = [rewards[i] for i in range(self.num_envs)]
        terminated = [terminated[i] for i in range(self.num_envs)]
        truncated = [truncated[i] for i in range(self.num_envs)]
        info = [info[i] for i in range(self.num_envs)]
        self._last_steps = [(obs[i], info[i]) for i in range(self.num_envs)]

        return obs, rewards, terminated, truncated, info

    def render(self) -> None | np.ndarray | str | list:
        for pipe in self.parent_pipes:
            pipe.send(("render", None))
        result = self._receive_results(self.parent_pipes, timeout=self.timeout)
        img, index, success = zip(*result)
        return img

    def close(self) -> None:
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for pipe in self.parent_pipes:
            pipe.close()
        for process in self.processes:
            process.join(timeout=0)

    def _try_reset(
        self, indices: list[int], seeds: list[int | None], options: list[dict | None]
    ):
        left_to_reset = indices.copy()
        seeds = {i: s for i, s in zip(indices, seeds)}
        options = {i: o for i, o in zip(indices, options)}
        obsv, infov = {}, {}
        attempt = 0
        while len(left_to_reset) > 0 and attempt < self.num_tries:
            logging.debug(
                f"Attempt {attempt + 1}/{self.num_tries} to reset {len(left_to_reset)} environments."
            )
            for idx in left_to_reset:
                pipe = self.parent_pipes[idx]
                kwargs = {"seed": seeds[idx], "options": options[idx]}
                logging.debug(f"Sending reset command to worker {idx}.")
                pipe.send(("reset", kwargs))

            responses = []
            for idx in left_to_reset:
                pipe = self.parent_pipes[idx]
                logging.debug(f"Waiting for response from worker {idx}.")
                if pipe.poll(timeout=self.timeout):
                    try:
                        responses.append(pipe.recv())
                    except EOFError:
                        logging.debug(f"Worker {idx} failed: EOF.")
                else:
                    logging.debug(f"Worker {idx} failed: Timeout.")

            for result, idx, success in responses:
                if success:
                    logging.debug(f"Resetting worker {idx} succeeded.")
                    obs, info = result
                    obsv[idx], infov[idx] = obs, info
                    left_to_reset.remove(idx)

            errors = self.get_errors()
            for i in left_to_reset:
                logging.debug(f"Worker {i} failed. Restarting.")
                logging.debug(f"Error: {errors.get(i, None)}")
                self._spawn_envs(left_to_reset)

        if len(left_to_reset) > 0:
            raise RuntimeError("Could not reset all environments.")

        logging.debug(f"Successfully reset {len(indices)} environments.")
        obs = [obsv[i] for i in indices]
        info = [infov[i] for i in indices]
        return obs, info

    def get_errors(self):
        errors = {}
        while True:
            try:
                idx, error = self.error_queue.get(timeout=1)
                errors[idx] = error
            except:
                break
        return errors

    def _receive_results(
        self, pipe: Connection | list[Connection], timeout: float = 5.0
    ):
        single = False
        if isinstance(pipe, Connection):
            single = True
            pipe = [pipe]
        results = []
        for p in pipe:
            if p.poll(timeout=timeout):
                try:
                    results.append(p.recv())
                except EOFError:
                    results.append(None)
            else:
                results.append(None)
        if single:
            return results[0]
        else:
            return results


def env_fn(port):
    env = mats_gym.scenic_env(
        host="localhost",  # The host to connect to
        scenario_specification="scenarios/scenic/four_way_route_scenario.scenic",
        scenes_per_scenario=1,
        resample_scenes=False,
        agent_name_prefixes=["vehicle"],
        render_mode="rgb_array",
        render_config=renderers.camera_pov(agent="vehicle_0"),
        traffic_manager_port=port - 2000 + 8000,
    )
    env = ServerWrapper(env, world_port=port, gpus=[str(port % 4)], wait_time=10)
    return env


if __name__ == "__main__":
    import cv2
    import time

    NUM_ENVS = 4
    logging.basicConfig(level=logging.DEBUG)
    env_fns = [partial(env_fn, port=2000 + (i * 3)) for i in range(NUM_ENVS)]
    vec_env = VecEnvWrapper(env_fns=env_fns, daemon=False)
    obs, info = vec_env.reset()
    done = False
    agents = [f"vehicle_{i}" for i in range(4)]
    action_space = gymnasium.spaces.Dict(
        {agent: vec_env.single_action_space(agent) for agent in agents}
    )
    start = time.time()
    num_steps = 0
    for i in range(500):
        actions = [action_space.sample() for _ in range(NUM_ENVS)]
        obs, reward, done, truncated, info = vec_env.step(actions)
        for img in vec_env.render():
            cv2.imshow("img", img[:, :, ::-1])
            cv2.waitKey(1)
        num_steps += NUM_ENVS
        print(f"Step {num_steps} FPS: {num_steps / (time.time() - start)}")

    vec_env.close()
