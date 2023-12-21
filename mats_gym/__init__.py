from typing import Type

import carla
import gymnasium
import pettingzoo

from mats_gym.envs.base_env import BaseScenarioEnv
from mats_gym.envs.adapters import ScenarioRunnerEnv, ScenicEnv, OpenScenarioEnv

def _make_env(ctor: Type[gymnasium.Env], **kwargs):
    host = kwargs.pop("host", None)
    port = kwargs.pop("port", None)
    if host is not None and port is not None:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
    else:
        client = None
    kwargs['client'] = client
    return ctor(**kwargs)


def raw_env(host: str = None, port: int = None, **kwargs) -> BaseScenarioEnv:
    return _make_env(BaseScenarioEnv, host=host, port=port, **kwargs)

def scenic_env(host: str = None, port: int = None, **kwargs) -> ScenicEnv:
    return _make_env(ScenicEnv, host=host, port=port, **kwargs)

def srunner_env(host: str = None, port: int = None, **kwargs) -> ScenarioRunnerEnv:
    return _make_env(ScenarioRunnerEnv, host=host, port=port, **kwargs)

def openscenario_env(host: str = None, port: int = None, **kwargs) -> OpenScenarioEnv:
    assert "client" in kwargs or (host is not None and port is not None), "Must specify client or host and port."
    return _make_env(OpenScenarioEnv, host=host, port=port, **kwargs)