from __future__ import annotations
from typing import List, Union
import carla
import numpy as np


from mats_gym.rendering.renderer import Renderer


class StackedRenderer(Renderer):
    def __init__(
        self, renderers: list[Renderer], render_mode: str, layout: list[int] = None
    ):
        assert render_mode in ["rgb_array", "rgb_array_list"]
        super().__init__(render_mode)
        self._renderers = renderers
        self._layout = layout

    def render(self) -> Union[None, np.ndarray, List[np.ndarray]]:
        images = [r.render() for r in self._renderers]
        if self._layout is None:
            return np.hstack(images)
        else:
            rows = []
            num_imgs = 0
            for row in self._layout:
                rows.append(np.hstack([images[i + num_imgs] for i in range(row)]))
                num_imgs += row
            return np.vstack(rows)

    def update(self) -> None:
        for r in self._renderers:
            r.update()

    def reset(self, client: carla.Client = None) -> None:
        for r in self._renderers:
            r.reset(client)

    def close(self) -> None:
        for r in self._renderers:
            r.close()
