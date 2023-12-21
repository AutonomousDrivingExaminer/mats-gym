import abc
from abc import ABC

import carla


class CarlaServer(ABC):
    """
    A server that can be used to start and stop a Carla simulation.
    """

    @abc.abstractmethod
    def get_client(self) -> carla.Client:
        """
        Returns a Carla client that can be used to connect to the server.
        @return: A Carla client instance.
        """
        pass

    @abc.abstractmethod
    def start(self) -> None:
        """
        Starts the server.
        @return: None
        """
        pass

    @abc.abstractmethod
    def is_running(self) -> bool:
        """
        Returns whether the server is running.
        @return: True if the server is running, False otherwise.
        """
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """
        Stops the server.
        @return: None
        """
        pass
