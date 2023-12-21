import carla
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_h
from pygame.locals import K_i
from pygame.locals import K_m
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_s
from pygame.locals import K_w

MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0

class InputControl(object):
    """Class that handles input received such as keyboard and mouse."""

    def __init__(self, name):
        """Initializes input member variables when instance is created."""
        self.name = name
        self.mouse_pos = (0, 0)
        self.mouse_offset = [0.0, 0.0]
        self.wheel_offset = 0.1
        self.wheel_amount = 0.025
        self._steer_cache = 0.0
        self.control = None
        self._autopilot_enabled = False

        # Modules that input will depend on
        self._hud = None
        self._world = None

    def start(self, hud, world):
        """Assigns other initialized modules that input module needs."""
        self._hud = hud
        self._world = world

        self._hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def render(self, display):
        """Does nothing. Input module does not need render anything."""

    def tick(self, clock):
        """Executed each frame. Calls method for parsing input."""
        self.parse_input(clock)

    def _parse_events(self):
        """Parses input events. These events are executed only once when pressing a key."""
        self.mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    pygame.quit()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    self._hud.help.toggle()
                elif event.key == K_TAB:
                    # Toggle between hero and map mode
                    if self._world.hero_actor is None:
                        self._world.select_hero_actor()
                        self.wheel_offset = HERO_DEFAULT_SCALE
                        self.control = carla.VehicleControl()
                        self._hud.notification('Hero Mode')
                    else:
                        self.wheel_offset = MAP_DEFAULT_SCALE
                        self.mouse_offset = [0, 0]
                        self.mouse_pos = [0, 0]
                        self._world.scale_offset = [0, 0]
                        self._world.hero_actor = None
                        self._hud.notification('Map Mode')
                elif event.key == K_F1:
                    self._hud.show_info = not self._hud.show_info
                elif event.key == K_i:
                    self._hud.show_actor_ids = not self._hud.show_actor_ids
                elif isinstance(self.control, carla.VehicleControl):
                    if event.key == K_q:
                        self.control.gear = 1 if self.control.reverse else -1
                    elif event.key == K_m:
                        self.control.manual_gear_shift = not self.control.manual_gear_shift
                        self.control.gear = self._world.hero_actor.get_control().gear
                        self._hud.notification('%s Transmission' % (
                            'Manual' if self.control.manual_gear_shift else 'Automatic'))
                    elif self.control.manual_gear_shift and event.key == K_COMMA:
                        self.control.gear = max(-1, self.control.gear - 1)
                    elif self.control.manual_gear_shift and event.key == K_PERIOD:
                        self.control.gear = self.control.gear + 1
                    elif event.key == K_p:
                        # Toggle autopilot
                        if self._world.hero_actor is not None:
                            self._autopilot_enabled = not self._autopilot_enabled
                            self._world.hero_actor.set_autopilot(self._autopilot_enabled)
                            self._hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle mouse wheel for zooming in and out
                if event.button == 4:
                    self.wheel_offset += self.wheel_amount
                    if self.wheel_offset >= 1.0:
                        self.wheel_offset = 1.0
                elif event.button == 5:
                    self.wheel_offset -= self.wheel_amount
                    if self.wheel_offset <= 0.1:
                        self.wheel_offset = 0.1

    def _parse_keys(self, milliseconds):
        """Parses keyboard input when keys are pressed"""
        keys = pygame.key.get_pressed()
        self.control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self.control.steer = round(self._steer_cache, 1)
        self.control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self.control.hand_brake = keys[K_SPACE]

    def _parse_mouse(self):
        """Parses mouse input"""
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            self.mouse_offset[0] += (1.0 / self.wheel_offset) * (x - self.mouse_pos[0])
            self.mouse_offset[1] += (1.0 / self.wheel_offset) * (y - self.mouse_pos[1])
            self.mouse_pos = (x, y)

    def parse_input(self, clock):
        """Parses the input, which is classified in keyboard events and mouse"""
        self._parse_events()
        self._parse_mouse()
        if not self._autopilot_enabled:
            if isinstance(self.control, carla.VehicleControl):
                self._parse_keys(clock.get_time())
                self.control.reverse = self.control.gear < 0
            if (self._world.hero_actor is not None):
                self._world.hero_actor.apply_control(self.control)

    @staticmethod
    def _is_quit_shortcut(key):
        """Returns True if one of the specified keys are pressed"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
