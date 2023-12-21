from typing import Union, List, NamedTuple

import carla
import numpy as np
import pygame

from mats_gym.rendering.pygame_modules import HUD, World, InputControl, colors
from mats_gym.rendering.renderer import Renderer


class BirdViewArgs(NamedTuple):
    actor: str
    client: carla.Client
    map: str = None
    no_rendering: bool = True
    show_triggers: bool = False
    show_connections: bool = False
    show_spawn_points: bool = False

class BirdViewRenderer(Renderer):

    def __init__(self, render_mode: str,
                 actor_id: str,
                 client: carla.Client = None,
                 width: int = 1280,
                 height: int = 720
                 ):
        self._actor_id = actor_id
        self._width = width
        self._height = height
        self._clock = pygame.time.Clock()
        self._render_mode = render_mode
        self._frames = []
        self._client = client

        pygame.init()
        if render_mode == 'human':
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        else:
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.HIDDEN
        self._display: pygame.Surface = pygame.display.set_mode(size=(self._width, self._height),  flags=flags)
        pygame.display.set_caption("ADEX BirdView Renderer")
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_surface = font.render('Rendering map...', True, pygame.Color((255, 255, 255)))
        self._display.blit(text_surface, text_surface.get_rect(center=(self._width / 2, self._height / 2)))

    def render(self) -> Union[None, np.ndarray, List[np.ndarray]]:
        # Render all modules
        if self._render_mode == 'rgb_array':
            if len(self._frames) == 0:
                return np.zeros((self._height, self._width, 3))
            else:
                return self._frames[-1]
        elif self._render_mode == 'rgb_array_list':
            return self._frames
        else:
            return None

    def update(self) -> None:
        self._clock.tick_busy_loop(60)
        self._world.tick(self._clock)
        self._hud.tick(self._clock)
        self._input_control.tick(self._clock)

        self._display.fill(colors.COLOR_ALUMINIUM_4)
        self._world.render(self._display)
        self._hud.render(self._display)
        self._input_control.render(self._display)
        pygame.display.flip()
        if self._render_mode != 'human':
            img = pygame.surfarray.array3d(self._display).swapaxes(0, 1)
            self._frames.append(img)


    def reset(self, client: carla.Client = None) -> None:
        if self._client is None and client is None:
            raise ValueError("Client is not set.")
        if client:
            self._client = client
            
        self._clock = pygame.time.Clock()
        self._frames = []
        TITLE_WORLD = 'WORLD'
        TITLE_HUD = 'HUD'
        TITLE_INPUT = 'INPUT'
        self._hud = HUD(TITLE_HUD, self._width, self._height)
        self._world = World(
            name=TITLE_WORLD,
            args=BirdViewArgs(
                actor=self._actor_id,
                client=self._client,
            ),
            timeout=2.0)
        self._input_control = InputControl(TITLE_INPUT)

        self._input_control.start(self._hud, self._world)
        self._hud.start()
        self._world.start(self._hud, self._input_control)

    def close(self) -> None:
        pygame.quit()





