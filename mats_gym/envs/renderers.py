from __future__ import annotations
from dataclasses import dataclass

import carla

from mats_gym.rendering import CameraRenderer, BirdViewRenderer
from mats_gym.rendering.renderer import Renderer
from mats_gym.rendering.stacked import StackedRenderer


@dataclass
class RenderConfig:
    agent: str = "sut"
    renderer: str = "camera_pov"
    kwargs: dict = None

    def __init__(self, agent: str = "sut", renderer: str = "camera_pov", **kwargs):
        self.kwargs = kwargs
        self.agent = agent
        self.renderer = renderer

def stacked(top_agent: str = "sut", bottom_agents: tuple[str, str] = ("adv_1", "adv_2")):
    """
    Returns a stacked renderer with a bird view on top and two camera povs on the bottom.
    :param top_agent: The agent to focus on in the top renderer.
    :param bottom_agents: The agents to focus on in the bottom renderers.
    :return: The corresponding rendering configuration.
    """
    return RenderConfig(
        renderer="stacked",  # 'camera_top', 'camera_pov', 'bird_view' or 'stacked'
        layout=[1, 2],
        renderers=[
            bird_view(top_agent, width=1440, height=480),
            camera_pov(bottom_agents[0], display_actor_id=True, width=720, height=480),
            camera_pov(bottom_agents[1], display_actor_id=True, width=720, height=480)
        ]
    )

def camera_pov(agent: str = "sut", **kwargs):
    """
    Returns a camera pov rendering configuration.
    :param agent: The agent to focus on.
    :return: The corresponding rendering configuration.
    """
    return RenderConfig(renderer="camera_pov", agent=agent, **kwargs)

def camera_top(agent: str = "sut", **kwargs):
    """
    Returns a camera top rendering configuration.
    :param agent: The agent to focus on.
    :return: The corresponding rendering configuration.
    """
    return RenderConfig(renderer="camera_top", agent=agent, **kwargs)

def bird_view(agent: str = "sut", **kwargs):
    """
    Returns a bird view rendering configuration.
    :param agent: The agent to focus on.
    :return: The corresponding rendering configuration.
    """
    return RenderConfig(renderer="bird_view", agent=agent, **kwargs)

def make_renderer(
    type: str, mode: str, focused_actor: str, **kwargs
) -> Renderer:
    """
    Factory method for creating renderers.
    :param type: The type of renderer to create. One of ['camera_top', 'camera_pov', 'bird_view'].
    :param mode: The mode of the renderer. One of ['human', 'rgb_array'].
    :param focused_actor: The actor id to focus on.
    :param client: The carla client.
    :return: The renderer.
    """
    if type == "camera_top":
        renderer = CameraRenderer(
            render_mode=mode,
            actor_id=focused_actor,
            height=kwargs.get("height", 720),
            width=kwargs.get("width", 1280),
            camera_transform=carla.Transform(
                carla.Location(x=0, y=0, z=40), carla.Rotation(pitch=-90)
            ),
        )
    elif type == "camera_pov":
        renderer = CameraRenderer(
            render_mode=mode,
            actor_id=focused_actor,
            **kwargs,
        )
    elif type == "bird_view":
        renderer = BirdViewRenderer(
            render_mode=mode, actor_id=focused_actor, **kwargs
        )
    elif type == "stacked":
        render_configs = kwargs.get("renderers")
        renderer = StackedRenderer(
            render_mode=mode,
            renderers=[
                make_renderer(
                    type=args.renderer,
                    mode=mode,
                    focused_actor=args.agent,
                    **args.kwargs,
                )
                for args in render_configs
            ],
            layout=kwargs.get("layout", None),
        )
    else:
        raise ValueError(f"Unknown renderer: {type}")
    return renderer