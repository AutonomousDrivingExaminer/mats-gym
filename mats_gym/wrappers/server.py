from __future__ import annotations

import logging
import time
from typing import Any

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.servers.docker_server import DockerCarlaServer
from mats_gym.servers.server import CarlaServer


class ServerWrapper(BaseScenarioEnvWrapper):
    """
    A wrapper that handles the initialization and termination of a Carla simulation server.
    """

    def __init__(
        self,
        env: BaseScenarioEnvWrapper,
        world_port: int,
        type: str = "docker",
        headless: bool = True,
        sensor_port: None | int = None,
        control_port: None | int = None,
        traffic_manager_port: None | int = None,
        wait_time: float = 5.0,
        server_kwargs: None | dict = None,
    ) -> None:
        """
        @param env: The environment to wrap.
        @param world_port: The world port of the server.
        @param type: The type of server to use. Currently only 'docker' is supported. Needs docker to be installed.
        @param headless: Whether to run the server in headless mode (no CARLA window).
        @param sensor_port: The sensor port of the server. Defaults to world_port + 1.
        @param control_port: The control port of the server. Defaults to world_port + 2.
        @param traffic_manager_port: The traffic manager port of the server. Defaults to None.
        @param wait_time: The time to wait after starting the server before connecting to it.
        @param server_kwargs: Additional keyword arguments to pass to the server constructor.
        """
        super().__init__(env)
        self._world_port = world_port
        self._traffic_manager_port = traffic_manager_port
        self._wait_time = wait_time
        self._server = self._make_server(
            type=type,
            world_port=world_port,
            sensor_port=sensor_port,
            control_port=control_port,
            headless=headless,
            **(server_kwargs or {}),
        )
        time.sleep(wait_time)

    def _make_server(
        self,
        type: str,
        world_port: int,
        sensor_port: int = None,
        control_port: int = None,
        headless: bool = True,
        **kwargs,
    ) -> CarlaServer:
        self._client = None
        if type == "docker":
            return DockerCarlaServer(
                container_name=f"carla-server-{world_port}",
                world_port=world_port,
                sensor_port=sensor_port,
                control_port=control_port,
                headless=headless,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown server type {type}. Only 'docker' is supported for now."
            )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        if not self._server.is_running():
            logging.debug(
                f"Carla server for env is not running on port {self._world_port}. Starting it now."
            )
            self._server.start()
            time.sleep(self._wait_time)
            self._client = self._server.get_client()
            self._client.set_timeout(10.0)
        assert self._client is not None
        options = options or {}
        options = options.copy()
        options["client"] = self._client
        if self._traffic_manager_port is not None:
            options["traffic_manager_port"] = self._traffic_manager_port
        return self.env.reset(seed=seed, options=options)

    def close(self) -> None:
        super().close()
        self._server.stop()
