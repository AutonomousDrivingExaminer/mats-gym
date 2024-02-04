from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional, List, Any

import carla
import gymnasium
import gymnasium.spaces
import numpy as np

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from mats_gym.wrappers.birdseye_view.birdseye import (
    BirdViewCropType,
    BirdViewMasks,
    BirdViewProducer,
)
from mats_gym.wrappers.birdseye_view.mask import PixelDimensions


@dataclass
class ObservationConfig:
    width: int = 150  # Width in pixels
    height: int = 336  # Height in pixels
    as_rgb: bool = (
        False  # Whether to return the bird view as RGB image or as a stack of masks
    )
    pixels_per_meter: int = 4  # The number of pixels per meter
    vehicle_class_prefixes: Optional[
        List[str]
    ] = None  # Each vehicle class prefix will be rendered as a separate mask.
    crop_type: BirdViewCropType = (
        BirdViewCropType.FRONT_AND_REAR_AREA
    )  # The crop type to use
    masks: Optional[
        List[BirdViewMasks]
    ] = BirdViewMasks  # The masks to include in the bird view


class BirdViewObservationWrapper(BaseScenarioEnvWrapper):
    """
    A wrapper for a scenario environment that adds a bird view observation.
    """

    def __init__(
        self,
        env: BaseScenarioEnvWrapper,
        obs_config: ObservationConfig
        | dict[str, ObservationConfig] = ObservationConfig(),
    ):
        """
        @param env: The environment to wrap.
        @param obs_config: A ObservationConfig instance or a dictionary mapping agent names to ObservationConfig instances.
        """
        super().__init__(env)
        if not isinstance(obs_config, dict):
            obs_config = {agent: obs_config for agent in self.agents}

        self._config = obs_config
        self._map = None
        self._producers = {}

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        obs, info = super().reset(seed=seed, options=options)
        map = self.client.get_world().get_map()
        if (
            self._map is None
            or self._map.name != map.name
            or self._producers.keys() != self.agents
        ):
            self._map = map
            for agent, config in self._config.items():
                vehicle_classes = config.vehicle_class_prefixes or []
                self._producers[agent] = BirdViewProducer(
                    client=self.env.client,
                    vehicle_class_classification=[
                        self._make_classifier(prefix) for prefix in vehicle_classes
                    ],
                    render_lanes_on_junctions=True,
                    target_size=PixelDimensions(
                        width=config.width, height=config.height
                    ),
                    pixels_per_meter=config.pixels_per_meter,
                    crop_type=config.crop_type,
                )
        obs = self._add_birdview_to_obs(obs)
        return obs, info

    def _make_classifier(self, prefix: str) -> typing.Callable[[carla.Actor], bool]:
        return lambda vehicle: vehicle.attributes.get("role_name", "").startswith(
            prefix
        )

    def step(
        self, actions: dict
    ) -> tuple[
        dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict]
    ]:
        obs, *rest = super().step(actions)
        return self._add_birdview_to_obs(obs), *rest

    def observation_space(self, agent: Any) -> gymnasium.spaces.Dict:
        obs_space = self.env.observation_space(agent)
        if agent in self._producers:
            config: ObservationConfig = self._config[agent]
            high = 1
            if config.as_rgb:
                num_layers = 3
                high = 255
            elif config.vehicle_class_prefixes is None:
                num_layers = len(config.masks) + 1
            else:
                num_layers = len(config.masks) + len(config.vehicle_class_prefixes)

            bv_space = gymnasium.spaces.Box(
                low=0,
                high=high,
                shape=(config.height, config.width, num_layers),
                dtype=np.uint8,
            )
            obs_space = gymnasium.spaces.Dict({**obs_space.spaces, "birdview": bv_space})
        return obs_space

    def _get_actor_config(self, agent: str) -> ActorConfiguration:
        configs = [
            c
            for c in self.env.current_scenario.config.ego_vehicles
            if c.rolename == agent
        ]
        if len(configs) > 0:
            return configs[0]
        else:
            return None

    def _get_birdview(self, agent: str) -> dict:
        if not agent in self._producers:
            return {}
        actor = self.env.actors[agent]
        actor_config = self._get_actor_config(agent)
        bv_config = self._config[agent]
        if actor_config.route:
            route = [loc for loc, wp in actor_config.route]
        else:
            route = None

        producer = self._producers[agent]
        bv = producer.produce(actor, route=route)

        # Zero out masks that are not in the config
        for m in BirdViewMasks:
            if m not in bv_config.masks:
                bv[..., m] = 0

        if bv_config.as_rgb:
            # Convert to RGB if requested
            bv = producer.as_rgb(bv)
        else:
            # Remove masks that are not in the config
            for m in BirdViewMasks:
                if m not in bv_config.masks:
                    np.delete(bv, m, axis=2)
        return bv

    def _add_birdview_to_obs(self, obs: dict) -> dict:
        for agent in self.agents:
            if agent in self._producers:
                obs[agent]["birdview"] = self._get_birdview(agent)
        return obs

    def observe(self, agent: str) -> dict:
        obs = self.env.observe(agent)
        if agent in self._producers:
            obs["birdview"] = self._get_birdview(agent)
        return obs

