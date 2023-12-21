from __future__ import annotations

import ctypes
import logging
import os
import signal
import time
from signal import SIGABRT

import carla
import docker
from docker.errors import NotFound

from mats_gym.servers.server import CarlaServer

c_globals = ctypes.CDLL(None)  # POSIX


def make_handler(container_name: str):
    def handler(signum, frame):
        logging.info(f"Received signal {signum}. Stopping container {container_name}.")
        client = docker.from_env()
        if client:
            try:
                container = client.containers.get(container_name)
                container.kill()
                time.sleep(2.0)
                container.remove(force=True)
                logging.info(f"Removed container with name {container_name}.")
            except NotFound:
                logging.info(f"Container with name {container_name} not found.")

    return handler


class DockerCarlaServer(CarlaServer):
    def __init__(
        self,
        world_port: int,
        sensor_port: int = None,
        control_port: int = None,
        headless: bool = False,
        gpus: list[str] = None,
        container_name: str = "carla-server",
        image: str = "carlasim/carla:0.9.13",
        num_connection_retries: int = 10,
    ):
        """
        @param world_port: The world port of the server.
        @param sensor_port: The sensor port of the server. Defaults to world_port + 1.
        @param control_port: The control port of the server. Defaults to world_port + 2.
        @param headless: Whether to start the server in headless mode.
        @param gpus: A list of gpus to pass to the docker container. Defaults to ["all"].
        @param container_name: The name of the docker container.
        @param image: The docker image to use.
        @param num_connection_retries: The number of retries to connect to the server.
        """
        if gpus is None:
            gpus = ["all"]
        if sensor_port is None:
            sensor_port = world_port + 1
        if control_port is None:
            control_port = world_port + 2

        self._name = container_name
        self._world_port = world_port
        self._sensor_port = sensor_port
        self._control_port = control_port
        self._gpus = gpus

        self._ports = {
            f"{world_port}/tcp": world_port,
            f"{sensor_port}/tcp": sensor_port,
            f"{control_port}/tcp": control_port,
        }
        self._image = image
        self._headless = headless
        self._container = None
        self._carla_client = None
        self._num_connection_retries = num_connection_retries

    def get_client(self) -> carla.Client:
        if self._container is None:
            raise ValueError("Carla server is not running.")

        logging.debug("Created new carla client.")
        client = carla.Client("localhost", self._world_port)
        client.set_timeout(10.0)
        return client

    def start(self):
        client = docker.from_env()
        try:
            container = client.containers.get(self._name)
            logging.info(
                f"Found existing container with name {self._name}. Stopping and removing."
            )
            container.kill()
            time.sleep(2.0)
            container.remove(force=True)
            time.sleep(2.0)
            logging.info(f"Removed existing container with name {self._name}.")
        except (docker.errors.NotFound, docker.errors.APIError) as e:
            pass

        logging.info(f"Creating new container with name {self._name}")
        container = self._create_container(client)
        self._register_signal_handlers(self._name)

        logging.info(f"Started CARLA server on port {self._world_port}.")
        self._container = container

    def _register_signal_handlers(self, container_name: str):
        handler = make_handler(container_name)
        signal.signal(SIGABRT, handler)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGSEGV, handler)

    def _create_container(
        self, client: docker.DockerClient
    ) -> docker.models.containers.Container:
        command = (
            "./CarlaUE4.sh -carla-server"
            f" -world-port={self._world_port}"
            f"{' -RenderOffScreen' if self._headless else ''}"
        )
        environment = [f'DISPLAY={os.environ.get("DISPLAY", ":0")}']
        if not self._headless:
            environment.append(f"SDL_VIDEODRIVER=x11")
        gpus = ",".join(self._gpus)
        logging.info(f"Starting container access to gpus: {gpus}")
        return client.containers.run(
            name=self._name,
            image=self._image,
            ports=self._ports,
            environment=environment,
            volumes={"/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"}},
            privileged=True,
            device_requests=[
                docker.types.DeviceRequest(device_ids=[gpus], capabilities=[["gpu"]])
            ],
            detach=True,
            remove=True,
            command=command,
        )

    def is_running(self) -> bool:
        if self._container is None:
            return False
        return True

    def stop(self):
        if self._container is not None:
            try:
                self._container.kill()
                time.sleep(2.0)
                self._container.remove(force=True)
            except:
                pass
        self._client = None
