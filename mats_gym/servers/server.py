from abc import ABC
import carla
import abc

class CarlaServer(ABC):

    @abc.abstractmethod
    def get_client(self) -> carla.Client:
        pass

    @abc.abstractmethod
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def is_running(self) -> bool:
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        pass

