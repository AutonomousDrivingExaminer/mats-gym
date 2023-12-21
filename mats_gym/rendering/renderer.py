from typing import Protocol, Union, List
import carla

import numpy as np


class Renderer(Protocol):
    def render(self) -> Union[None, np.ndarray, List[np.ndarray]]:
        ...

    def update(self) -> None:
        ...

    def reset(self, client: carla.Client = None) -> None:
        ...

    def close(self) -> None:
        ...

    def __init__(self, render_mode: str):
        pass
