import carla
import cv2
import numpy as np
import pygame
from carla import ColorConverter as cc
import weakref


class CameraRenderer:
    def __init__(
        self,
        render_mode,
        actor_id: str,
        client: carla.Client = None,
        display_actor_id: bool = False,
        camera_transform: carla.Transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)
        ),
        width: int = 1280,
        height: int = 720,
        camera: str = "sensor.camera.rgb",
    ):
        self._render_mode = render_mode
        self._client: carla.Client = client or None
        self._world: carla.World = client.get_world() if client is not None else None
        self._actor_id = actor_id
        self._camera_type = camera
        self._width = width
        self._height = height
        self._camera = None
        self._frames = []
        self._camera_transform = camera_transform
        self._display = None
        self._display_actor_id = display_actor_id
        self._setup(render_mode=render_mode)

    def _setup(self, render_mode: str):
        if render_mode == "human":
            pygame.init()
            self._display = pygame.display.set_mode(
                size=(self._width, self._height),
                flags=pygame.HWSURFACE | pygame.DOUBLEBUF,
            )

    def _spawn_camera(self, camera_type: str, width: int, height: int):
        bp_library = self._world.get_blueprint_library()
        camera = bp_library.find(camera_type)
        camera.set_attribute("image_size_x", str(width))
        camera.set_attribute("image_size_y", str(height))
        camera: carla.Sensor = self._world.spawn_actor(
            camera, self._camera_transform, attach_to=self._actor
        )
        camera.listen(
            lambda image: CameraRenderer._parse_image(
                weak_self=weakref.ref(self), image=image
            )
        )
        return camera

    def _get_actor(self, actor_id: str):
        actors: carla.ActorList = self._world.get_actors()
        for v in actors:
            if v.attributes.get("role_name", None) == actor_id:
                return v
        return None

    def reset(self, client: carla.Client = None):
        if self._client is None and client is None:
            raise ValueError("Client is not set.")
        if client:
            self._client = client
        self._world: carla.World = self._client.get_world()
        self._actor: carla.Actor = self._get_actor(self._actor_id)
        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
        self._camera = self._spawn_camera(
            camera_type=self._camera_type, width=self._width, height=self._height
        )
        self._frames = []

    def update(self):
        self._world: carla.World = self._client.get_world()

    def _render_human(self):
        if len(self._frames) > 0:
            surface = pygame.surfarray.make_surface(self._frames[-1].swapaxes(0, 1))
            self._display.blit(surface, (0, 0))
        pygame.display.flip()

    def _render_rgb_array(self):
        return self._frames[-1]

    def _render_rgb_array_list(self):
        return self._frames

    def render(self):
        if self._render_mode == "rgb_array":
            if len(self._frames) == 0:
                return np.zeros((self._height, self._width, 3))
            else:
                return self._render_rgb_array()
        elif self._render_mode == "human":
            self._render_human()
        elif self._render_mode == "rgb_array_list":
            return self._render_rgb_array_list()

    def _add_text(self, text: str, image: np.ndarray):
        scale, thickness = 2.0, 2
        (width, height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_PLAIN, scale, thickness
        )
        x = int((image.shape[1] - width) / 2)
        y = int(height + 20)
        cv2.putText(
            img=image,
            text=text.upper(),
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = np.array(array[:, :, ::-1])
        if self._display_actor_id:
            self._add_text(text=self._actor_id, image=array)
        self._frames.append(array)

    def close(self):
        if self._camera is not None:
            self._camera.stop()
        if self._display:
            pygame.quit()
