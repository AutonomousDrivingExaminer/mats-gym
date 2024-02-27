from __future__ import annotations

import copy
import logging
import random
import typing
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Union, List

import carla
import gymnasium
import numpy as np
import psutil
import py_trees
from gymnasium.core import RenderFrame, ObsType
from pettingzoo import ParallelEnv
from srunner.scenarioconfigs.scenario_configuration import (
    ScenarioConfiguration,
)
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType
from srunner.scenarios.basic_scenario import BasicScenario

from mats_gym.envs import renderers, space_utils
from mats_gym.envs.evaluation import RouteEvaluator
from mats_gym.envs.replays import SimulationHistory, Frame
from mats_gym.scenarios.scenario_wrapper import ScenarioWrapper
from mats_gym.sensors import SensorSuite


def is_used(port):
    """Checks whether a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def get_next_free_port(port):
    return next(filter(lambda p: not is_used(p), range(port, 32000)))


class BaseScenarioEnv(ParallelEnv):
    """
    Base class for scenario environments. This class manages the execution of a scenario.
    """

    metadata = {"render_modes": ["human", "rgb_array", "rgb_array_list"]}

    def __init__(
            self,
            config: ScenarioConfiguration,
            scenario_fn: typing.Callable[
                [carla.Client, ScenarioConfiguration], BasicScenario
            ],
            client: carla.Client = None,
            scenario_wrappers: list[ScenarioWrapper] | ScenarioWrapper = None,
            sensor_specs: dict[str, list[dict]] = None,
            seed: int = None,
            no_rendering_mode: bool = False,
            render_mode: str = None,
            render_config: renderers.RenderConfig = renderers.RenderConfig(),
            infractions_penalties: dict[TrafficEventType, float] = None,
            replay_dir: str = "/home/carla",
            debug_mode: bool = False,
            timestep: float = 0.05,
            traffic_manager_port: int = None,
    ) -> None:
        """
        @param config: A scenario configuration instance.
        @param scenario_fn: A function that, given a client and a scenario configuration, returns a scenario instance.
        @param client: A carla client instance. Optional.
        @param scenario_wrappers: A list of scenario wrappers to apply to the scenario. Optional.
        @param sensor_specs: A dictionary mapping actor names to a list of sensor specifications. Optional.
        @param seed: Random seed to use for the environment. Optional.
        @param no_rendering_mode: Whether to use no rendering mode in carla. Default: False.
        @param render_mode: The render mode to use. One of "human" or "rgb_array". Default: None.
        @param render_config: A render configuration instance. Default: @see mats_gym.envs.renderers.RenderConfig.
        @param infractions_penalties: A dictionary mapping traffic event types to penalties. Default: None.
        @param replay_dir: The directory to store replays in. Default: "/home/carla".
        @param debug_mode: Whether to use debug mode. Enabling debug mode will print additional information. Default: False.
        @param timestep: The timestep to use for the environment. Default: 0.05.
        @param traffic_manager_port: The port to use for the traffic manager. Default: 8000 or the next free port.
        """
        if traffic_manager_port is None:
            traffic_manager_port = get_next_free_port(8000)

        self._traffic_manager_port = traffic_manager_port
        CarlaDataProvider.set_traffic_manager_port(traffic_manager_port)
        self._seed = seed
        self._timestep = timestep

        if client:
            CarlaDataProvider.set_client(client)
            CarlaDataProvider.set_world(client.get_world())

        self._render_config = render_config
        self._scenario_fn = scenario_fn
        if isinstance(scenario_wrappers, ScenarioWrapper):
            scenario_wrappers = [scenario_wrappers]
        self._scenario_wrappers = scenario_wrappers or []
        self._client = client
        self._debug_mode = debug_mode
        self._events = defaultdict(list)
        self._current_step = 0
        self._num_resets = 0
        self._sensor_specs = sensor_specs or {}
        self._replay_dir = replay_dir
        self._sensors = self._make_sensor_suites(self._sensor_specs)
        self._evaluator = RouteEvaluator(infractions_penalties)

        self._make_rolenames_unique(config)
        self._config = config

        self._scenario = None
        self._no_rendering_mode = no_rendering_mode

        self.render_mode = render_mode
        self._renderer = (
            renderers.make_renderer(
                type=render_config.renderer,
                mode=render_mode,
                focused_actor=render_config.agent,
                client=client,
                **(render_config.kwargs or {}),
            )
            if render_mode is not None
            else None
        )
        self.possible_agents = [agent.rolename for agent in self._config.ego_vehicles]
        self.agents = self.possible_agents[:]

    def _make_rolenames_unique(self, config):
        actor_rolenames = [actor.rolename for actor in config.other_actors]
        for i, actor in enumerate(config.other_actors):
            if not actor.rolename:
                actor.rolename = f"npc_{i}"
            count = actor_rolenames.count(actor.rolename)
            if count > 1:
                actor.rolename = f"{actor.rolename}_{count}"
                actor_rolenames.remove(actor.rolename)

    def _make_sensor_suites(self, sensor_specs):
        return {actor: SensorSuite(sensor_specs[actor]) for actor in sensor_specs}

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        config = [c for c in self._config.ego_vehicles if c.rolename == agent]
        assert len(config) != 0, f"Unknown agent: {agent}"
        assert len(config) == 1, f"Multiple agents with name: {agent}"
        agent_cfg = config[0]
        if agent_cfg.model.startswith("vehicle"):
            return space_utils.get_vehicle_action_space()
        elif agent_cfg.model.startswith("walker"):
            return space_utils.get_walker_action_space()
        else:
            raise ValueError(f"Unknown action type for model: {type(agent_cfg.model)}")

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        config = [c for c in self._config.ego_vehicles if c.rolename == agent]
        assert len(config) != 0, f"Unknown agent: {agent}"
        assert len(config) == 1, f"Multiple agents with name: {agent}"
        spaces = {
            "location": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "velocity": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "rotation": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "speed": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=()),
        }
        if agent in self._sensors:
            suite = self._sensors[agent]
            spaces.update(suite.observation_space)
        return gymnasium.spaces.Dict(spaces)

    @property
    def actors(self) -> dict[str, carla.Vehicle | carla.Walker]:
        if self._scenario is None:
            return {agent: None for agent in self.agents}
        else:
             return {
                agent.attributes.get("role_name", agent.id): agent
                for agent in self._scenario.ego_vehicles + self._scenario.other_actors
            }

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

    @property
    def scenario_status(self) -> py_trees.common.Status:
        """
        Returns the status of the scenario.
        @return: The status of the scenario. Either RUNNING, SUCCESS, or FAILURE.
        """
        return self._scenario.scenario_tree.status

    @property
    def client(self) -> carla.Client:
        """
        Returns the carla client.
        @return: The carla client.
        """
        return self._client

    @property
    def current_scenario(self) -> BasicScenario:
        """
        Returns the current scenario instance.
        @return: The current scenario instance.
        """
        return self._scenario

    def observe(self, agent: str) -> dict[str, ObsType]:
        """
        Returns the observations for the given agent.
        @param agent: The agent name.
        @return: The observations for the agent.
        """
        obs = {}
        actor = self.actors[agent]
        transform = actor.get_transform()
        velocity = actor.get_velocity()

        obs["location"] = np.array([
            transform.location.x,
            transform.location.y,
            transform.location.z,
        ], dtype=np.float32)

        obs["rotation"] = np.array([
            transform.rotation.roll,
            transform.rotation.pitch,
            transform.rotation.yaw,
        ], dtype=np.float32)

        obs["velocity"] = np.array([
            velocity.x,
            velocity.y,
            velocity.z,
        ], dtype=np.float32)

        # compute speed by projecting velocity into forward direction
        forward_vector = transform.get_forward_vector()
        velocity_vector = obs["velocity"]
        obs["speed"] = np.dot(
            velocity_vector,
            np.array([forward_vector.x, forward_vector.y, forward_vector.z]),
        )

        # retrieve sensor observations
        if agent in self._sensors:
            sensor_obs = self._sensors[agent].get_observations()
            obs.update(sensor_obs)

        return obs

    def _reload_world(self):
        world = self._client.get_world()
        world.tick()
        settings: carla.WorldSettings = world.get_settings()
        map_name = world.get_map().name.split("/")[-1]

        if (
                not settings.synchronous_mode
                or settings.fixed_delta_seconds != self._timestep
                or settings.no_rendering_mode != self._no_rendering_mode
        ):
            logging.debug(
                f"Setting world settings to sync mode with timestep {self._timestep}."
            )
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self._timestep
            settings.no_rendering_mode = self._no_rendering_mode
            world.apply_settings(settings)

        logging.debug(f"Destroying all actors.")
        self._client.apply_batch_sync(
            [carla.command.DestroyActor(x) for x in world.get_actors()]
        )
        self._client.get_world().tick()

        if map_name == self._config.town:
            logging.debug(f"Reloading world.")
            world = self._client.reload_world(reset_settings=False)
        else:
            logging.debug(f"Loading world with map {self._config.town}.")
            world = self._client.load_world(self._config.town, reset_settings=False)

        if self._seed is not None:
            logging.debug(
                f"Seeding traffic manager at port {self._traffic_manager_port} with seed {self._seed}."
            )
            tm = self._client.get_trafficmanager(self._traffic_manager_port)
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(self._seed)

        world.tick()
        logging.debug("Setting up CarlaDataProvider.")
        CarlaDataProvider.cleanup()
        CarlaDataProvider.set_client(self._client)
        CarlaDataProvider.set_world(world)
        CarlaDataProvider.get_map()

    def reset(
            self,
            seed: int | None = None,
            options: dict | None = None,
    ) -> tuple[dict[str, ObsType], dict[str, dict]]:
        """
        Resets the environment.
        @param seed: The seed to use for the environment. Optional.
        @param options: Additional options to pass to the environment. Accepted options are:
            - client: A carla client instance.
            - traffic_manager_port: The port to use for the traffic manager.
            - scenario_config: A scenario configuration instance.
            - scenario_wrappers: A list of scenario wrappers to apply to the scenario.
            - replay: A dictionary containing
                 - history: A simulation history instance.
                 - num_frames: The number of frames to replay. Default: len(history).
        @return: A tuple containing the observations and info.
        """
        options = options or {}
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._seed = seed

        if "client" in options:
            self._client = options["client"]
            CarlaDataProvider.set_client(self._client)

        if "traffic_manager_port" in options:
            self._traffic_manager_port = options["traffic_manager_port"]

        CarlaDataProvider.set_traffic_manager_port(self._traffic_manager_port)

        assert self._client is not None, "Client is not set."
        logging.debug("Resetting scenario environment.")

        if self._scenario is not None:
            self._scenario.terminate()

        self.client.stop_recorder()
        for suite in self._sensors.values():
            suite.cleanup()
        self._sensors = self._make_sensor_suites(self._sensor_specs)

        self._current_step += 1
        if "scenario_config" in options:
            config = options["scenario_config"]
            self._make_rolenames_unique(config)
            self._config = config
            logging.info(f"Reset with new config.")

        if "scenario_wrappers" in options:
            wrappers = options["scenario_wrappers"]
            if not isinstance(wrappers, list):
                wrappers = [wrappers]
            self._scenario_wrappers = wrappers
            logging.info(f"Reset with new scenario wrappers.")

        logging.info(f"Resetting world.")
        self._reload_world()
        logging.debug(f"Number of actors: {len(self.client.get_world().get_actors())}.")


        logging.info(f"Calling scenario function to reset scenario.")
        scenario = self._scenario_fn(self.client, self._config)

        for wrapper in self._scenario_wrappers:
            scenario = wrapper.wrap(scenario)

        self.client.start_recorder(f"{self._replay_dir}/scenario-env.log")
        self._scenario: BasicScenario = scenario
        self.possible_agents = [agent.rolename for agent in self._config.ego_vehicles]
        self.agents = self.possible_agents[:]
        # Set up sensors
        for agent, sensor_suite in self._sensors.items():
            logging.debug(f"Setting up sensors for agent {agent}.")
            sensor_suite.setup_sensors(vehicle=self.actors[agent])

        # Tick the world
        logging.debug("Ticking the world.")
        CarlaDataProvider.get_world().tick()
        CarlaDataProvider.get_world().tick()

        # Get first snapshot and reset GameTime
        logging.debug("Initializing GameTime.")
        GameTime.restart()
        GameTime.on_carla_tick(CarlaDataProvider.get_world().get_snapshot())

        # Update CarlaDataProvider and the scenario tree
        logging.debug("Updating CarlaDataProvider and scenario tree.")
        CarlaDataProvider.on_carla_tick()
        self._scenario.scenario_tree.tick_once()

        if self._renderer:
            self._renderer.reset(self.client)
            self._renderer.update()

        self._events = {node.id: [] for node in self._scenario.get_criteria()}
        self._current_step = 0


        obs = {agent: self.observe(agent) for agent in self.agents}
        info = self._get_simulation_info()

        logging.debug("Resetting scenario environment done.")
        self.controls = []
        return obs, info

    def step(
            self, action: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, ObsType],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """
        Executes a step in the environment.
        @param action: A dictionary mapping agent names to actions.
        @return: A tuple containing the observations, rewards, terminated, truncated, and info.
        """

        # Apply actions and keep track of action history
        self._apply_actions(action)

        # Update simulation and monitors
        CarlaDataProvider.get_world().tick()
        self._current_step += 1
        CarlaDataProvider.on_carla_tick()
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        GameTime.on_carla_tick(snapshot.timestamp)
        self._scenario.scenario_tree.tick_once()

        # Retrieve observations, rewards, and info
        finished = self.scenario_status != py_trees.common.Status.RUNNING

        info = self._get_simulation_info()
        obs = {agent: self.observe(agent) for agent in self.agents}
        terminated = {agent: finished for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        logging.debug(f"Scenario status: {self.scenario_status.name}")

        if finished:
            logging.info(
                "Scenario terminated."
                f"Simulation time: {GameTime.get_time():.3f}s, "
                f"Frames: {self._current_frame()}."
            )
            events = self._get_events(self._scenario.get_criteria())
            eval_info, rewards = {}, {}
            for agent in self.agents:
                score, score_info = self._evaluator.compute_score(events=events[agent])
                if self._scenario.timeout_node.timeout:
                    score_info["infractions"]["route_timeout"] = ["Route timeout."]
                eval_info[agent] = score_info

                rewards[agent] = score
                info[agent].update(score_info)

            self._scenario.terminate()
        else:
            rewards = {agent: 0.0 for agent in self.agents}

        if self._renderer:
            self._renderer.update()

        return obs, rewards, terminated, truncated, info

    def _apply_actions(self, actions):
        commands = []
        controls = {}
        actors = self.actors
        for agent, act in actions.items():
            if isinstance(actors[agent], carla.Vehicle):
                control = carla.VehicleControl(
                    throttle=round(act[0].item(), 2),
                    steer=round(act[1].item(), 2),
                    brake=round(act[2].item(), 2),
                )
                # logging.debug(
                #    f"Applying controls: "
                #    f"throttle={control.throttle:.2f}, steer={control.steer:.2f}, brake={control.brake:.2f} "
                #    f"to agent {agent}."
                # )
                command = carla.command.ApplyVehicleControl(actors[agent], control)
                controls[agent] = control
            elif isinstance(actors[agent], carla.Walker):
                direction, speed, jump = act["direction"], act["speed"], act["jump"]
                control = carla.WalkerControl(
                    direction=carla.Vector3D(
                        x=direction[0].item(),
                        y=direction[1].item(),
                        z=direction[2].item(),
                    ),
                    speed=speed.item(),
                    jump=bool(jump),
                )
                command = carla.command.ApplyWalkerControl(actors[agent], control)
                logging.debug(
                    f"Applying controls: "
                    f"speed={control.speed:.2f}, jump={control.jump} "
                    f"to agent {agent}."
                )
            else:
                raise ValueError(f"Unknown agent type: {type(actors[agent])}")

            commands.append(command)
            controls[agent] = control

        self.client.apply_batch_sync(commands)
        self.controls.append(controls)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """
        Renders the environment.
        @return: A rgb array if "rgb_array" is set as render mode, otherwise None.
        """
        return self._renderer.render() if self._renderer else None

    def close(self):
        """
        Closes the environment. Terminates the scenario, cleans up CarlaDataProvider, and closes the renderer.
        @return: None.
        """
        logging.debug("Closing scenario environment.")

        for agent, sensor_suite in self._sensors.items():
            logging.debug(f"Destroying sensors for agent {agent}.")
            sensor_suite.cleanup()

        if self._scenario:
            self._scenario.terminate()
            logging.debug("Terminated scenario.")
        CarlaDataProvider.cleanup()
        logging.debug("Cleaned up CarlaDataProvider.")
        if self._renderer:
            self._renderer.close()
            logging.debug("Closed renderer.")

    def _get_simulation_info(self) -> dict:
        info = {
            "__common__": {
                "current_frame": self._current_frame(),
                "simulation_time": GameTime.get_time(),
            }
        }
        events = self._get_events(self._scenario.get_criteria())
        for agent in self.agents:
            info[agent] = {}
            agent_events = []
            for event in events[agent]:
                if isinstance(event.get_type(), TrafficEventType):
                    type = event.get_type().name
                else:
                    type = str(event.get_type())

                agent_events.append({"event": type, **event.get_dict()})

            info[agent]["events"] = agent_events
        return info

    def _current_frame(self):
        return self._current_step + 1

    def _get_events(self, criteria: Criterion) -> dict[str, list[TrafficEvent]]:
        events = defaultdict(list)
        for node in criteria:
            agent_id = node.actor.attributes.get("role_name")
            if node.id not in self._events:
                self._events[node.id] = []

            if len(node.events) > len(self._events[node.id]):
                logging.debug("New traffic events detected.")
                for event in node.events[len(self._events[node.id]):]:
                    event.set_dict(
                        {
                            "frame": self._current_frame(),
                            "simulation_time": GameTime.get_time(),
                            **(event.get_dict() or {}),
                        }
                    )
                    self._events[node.id].append(event)
            events[agent_id].extend(self._events[node.id])
        return events
