from __future__ import annotations
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData
from agents.navigation.local_planner import RoadOption
import carla

class ActorConfiguration(ActorConfigurationData):

    route: list[tuple[carla.Location, RoadOption]] = None

    def __init__(self, model, transform, rolename='other', speed=0, autopilot=False, random=False, color=None, category="car", route=None, args=None, **kwargs):
        super().__init__(model, transform, rolename, speed, autopilot, random, color, category, args)
        self.route = route
        for key, value in kwargs.items():
            setattr(self, key, value)
    
