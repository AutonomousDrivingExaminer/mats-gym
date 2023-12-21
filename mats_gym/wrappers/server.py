from __future__ import annotations

import logging
import time
from typing import Any

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.servers.docker_server import DockerCarlaServer
from mats_gym.servers.server import CarlaServer


class ServerWrapper(BaseScenarioEnvWrapper):
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
            gpus=None,
    ) -> None:
        super().__init__(env)
        if gpus is None:
            gpus = ["all"]
        self._world_port = world_port
        self._traffic_manager_port = traffic_manager_port
        self._wait_time = wait_time
        self._server = self._make_server(
            type,
            world_port=world_port,
            sensor_port=sensor_port,
            control_port=control_port,
            headless=headless,
            gpus=gpus,
            **(server_kwargs or {})
        )
        time.sleep(wait_time)

    def _make_server(
            self,
            type: str,
            world_port: int,
            sensor_port: int = None,
            control_port: int = None,
            headless: bool = True,
            gpus=None,
            **kwargs
    ) -> CarlaServer:
        if gpus is None:
            gpus = ["all"]
        self._client = None
        if type == "docker":
            return DockerCarlaServer(
                container_name=f"carla-server-{world_port}",
                world_port=world_port,
                sensor_port=sensor_port,
                control_port=control_port,
                headless=headless,
                gpus=gpus,
                **kwargs
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
