from __future__ import annotations

import dataclasses
import logging
import typing

import carla
import numpy as np
import py_trees
import scenic
import scenic.core.scenarios as scenic_scenarios
import scenic.syntax.veneer as veneer
from scenic.core.dynamics import Behavior
from scenic.core.object_types import disableDynamicProxyFor
from scenic.core.regions import PolylineRegion
from scenic.core.simulators import TerminationType, EndSimulationAction
from scenic.domains.driving.roads import Lane
from scenic.simulators.carla.misc import is_within_distance_ahead
from scenic.simulators.carla.simulator import CarlaSimulation
from scenic.simulators.carla.utils.utils import scenicToCarlaLocation, carlaToScenicPosition, scenicToCarlaRotation
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, \
    ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics import atomic_criteria
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenarios.basic_scenario import BasicScenario

from mats_gym.criterions import CollisionTest
from mats_gym.navigation.common import RoadOption


class ScenicSimulation(AtomicBehavior):

    def __init__(self, simulation: CarlaSimulation, max_time_steps: int = None):
        super().__init__('ScenicSimulation')
        self.max_time_steps = max_time_steps
        self._num_steps = 0
        self.dynamic_scenario = simulation.scene.dynamicScenario
        self.scene = simulation.scene
        self.simulation = simulation

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))
        if self.dynamic_scenario._isRunning:
            self.dynamic_scenario._stop(reason='new simulation started')
            for agent in self.dynamic_scenario._agents:
                assert isinstance(agent.behavior, Behavior), agent.behavior
                agent.behavior._stop(agent)
            for monitor in self.dynamic_scenario._monitors:
                monitor._stop(monitor)

        if veneer.currentSimulation is not None:
            veneer.endSimulation(self.simulation)
        veneer.beginSimulation(self.simulation)
        self.dynamic_scenario._start()
        for obj in self.simulation.objects:
            obj.startDynamicSimulation()
        self.simulation.updateObjects()
        for agent in self.simulation.scheduleForAgents():
            if not agent.behavior._runningIterator:
                agent.behavior._start(agent)


    def terminate(self, new_status):
        for scenario in tuple(veneer.runningScenarios):
            scenario._stop('simulation terminated')
        #self.simulation.destroy()
        for obj in self.simulation.scene.objects:
            disableDynamicProxyFor(obj)
        for agent in self.simulation.agents:
            if agent.behavior._isRunning:
                agent.behavior._stop()
        for monitor in self.simulation.scene.monitors:
            if monitor._isRunning:
                monitor._stop()
        # If the simulation was terminated by an exception (including rejections),
        # some scenarios may still be running; we need to clean them up without
        # checking their requirements, which could raise rejection exceptions.
        for scenario in tuple(veneer.runningScenarios):
            scenario._stop('exception', quiet=True)
        veneer.endSimulation(self.simulation)


    def update(self):
        termination_reason = self.dynamic_scenario._step()
        termination_type = TerminationType.scenarioComplete
        self.simulation.recordCurrentState()
        new_reason = self.dynamic_scenario._runMonitors()
        if new_reason is not None:
            termination_reason = new_reason
            termination_type = TerminationType.terminatedByMonitor
        if termination_reason is not None:
            self.logger.debug(f"{self.__class__.__name__}: Terminating due to {termination_type}: {termination_reason}")
            if termination_type == TerminationType.scenarioComplete:
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        termination_reason = self.dynamic_scenario._checkSimulationTerminationConditions()
        if termination_reason is not None:
            self.logger.debug(f"{self.__class__.__name__}: Terminating due to simulation termination condition: {termination_reason}")
            return py_trees.common.Status.SUCCESS
        if self.max_time_steps and self._num_steps >= self.max_time_steps:
            self.logger.debug( f"{self.__class__.__name__}: Reached maximum number of steps: {self.max_time_steps}")
            return py_trees.common.Status.SUCCESS

        all_actions = {}
        for agent in self.simulation.scheduleForAgents():
            actions = agent.behavior._step()
            if isinstance(actions, EndSimulationAction):
                self.logger.debug(f"{self.__class__.__name__}: Terminating due to agent {agent} action: {actions}")
                return py_trees.common.Status.SUCCESS
            assert isinstance(actions, tuple)
            if len(actions) == 1 and isinstance(actions[0], (list, tuple)):
                actions = tuple(actions[0])
            #if not self.simulation.actionsAreCompatible(agent, actions):
            #    raise InvalidScenarioError(f'agent {agent} tried incompatible action(s) {actions}')
            all_actions[agent] = actions
        self.simulation.executeActions(all_actions)
        self._num_steps += 1
        self.simulation.updateObjects()
        return py_trees.common.Status.RUNNING

    def setup(self, unused_timeout=15):
        return super().setup(unused_timeout)

@dataclasses.dataclass
class ScenicScenarioConfiguration(ScenarioConfiguration):
    scene: scenic.core.scenarios.Scene
    town: str = None
    timestep: float = 0.05
    max_time_steps: int | None = None
    traffic_manager_port: int = 8000
    ego_vehicles: typing.List[ActorConfigurationData] = None
    seed: int = None
    route = None
    trigger_points = None

class ScenicActorConfigurationData(ActorConfigurationData):
    route: typing.Optional[list[carla.Waypoint]] = None

    def __init__(self, model, transform, rolename='other', route=None, speed=0, autopilot=False, random=False,
                 color=None, category="car", args=None):

        self.route = route
        super().__init__(model, transform, rolename, speed, autopilot, random, color, category,
                         args)


class ScenicScenario(BasicScenario):

    def __init__(self,
                 client: carla.Client,
                 config: ScenicScenarioConfiguration,
                 terminate_on_failure=False,
                 criteria_enable=False,
                 debug_mode=False):

        self._scene = config.scene
        self._debug_mode = debug_mode
        self._max_time_steps = config.max_time_steps or 10000000000000
        self._tm_port = config.traffic_manager_port
        self._simulation: typing.Optional[CarlaSimulation] = None
        self._ego_vehicle_names = [ego_vehicle.rolename for ego_vehicle in config.ego_vehicles] or ['hero']
        self._seed = config.seed
        self.world = client.get_world()
        self.timestep = config.timestep
        self.timeout = self._max_time_steps
        self.client = client
        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(self.world)
        self.ego_vehicles = []
        self.other_actors = []
        self.config = config

        super().__init__(
            name='ScenicScenario',
            ego_vehicles=[],
            config=config,
            world=self.world,
            debug_mode=debug_mode,
            terminate_on_failure=terminate_on_failure,
            criteria_enable=criteria_enable
        )

    def _initialize_environment(self, world: carla.World):
        logging.debug("Initializing environment. Creating scenic simulation...")
        self._simulation = CarlaSimulation(
            client=self.client,
            scene=self._scene,
            tm=self.client.get_trafficmanager(self._tm_port),
            timestep=self.timestep,
            render=False,
            record=False,
            scenario_number=0
        )
        self._simulation.initializeReplay(
            replay=None,
            enableReplay=True,
            allowPickle=True,
            enableDivergenceCheck=False
        )
        world.tick()
        logging.debug("Finished initializing environment.")

    def _initialize_actors(self, config: ScenarioConfiguration):
        logging.debug("Initializing actors.")
        self.other_actors = []
        self.ego_vehicles = []
        controllable_objects = [
            o for o in self._simulation.objects
            if isinstance(o.carlaActor, carla.Vehicle) or isinstance(o.carlaActor, carla.Walker)
        ]
        for obj in controllable_objects:
            actor = obj.carlaActor

            if 'role_name' not in actor.attributes:
                actor.attributes['role_name'] = obj.name

            CarlaDataProvider.register_actor(actor)

            if actor.attributes['role_name'] in self._ego_vehicle_names:
                actor_config = next((ego_vehicle for ego_vehicle in config.ego_vehicles if ego_vehicle.rolename == actor.attributes['role_name']), None)
                if hasattr(obj, "route"):
                    actor_config.route = self._generate_route(actor=actor, route=obj.route)

                self.ego_vehicles.append(actor)
                logging.debug(f"Created ego vehicle '{actor.attributes['role_name']}'.")
            else:
                self.other_actors.append(actor)
                logging.debug(f"Created non-ego actor '{actor.attributes['role_name']}'.")

    def _generate_route(self, actor: carla.Vehicle, route: list[Lane], resolution=1.0) -> list[tuple[carla.Waypoint, RoadOption]]:
        network = self.config.scene.workspace.network
        map = CarlaDataProvider.get_map()
        carlaToScenicPosition(actor.get_location())
        # If the start of the route is ahead of the vehicle, we need to add its current lane
        curr_lane = network.laneAt(carlaToScenicPosition(actor.get_location()))
        if curr_lane and curr_lane.id != route[0].id:
            route = [curr_lane] + route

        # Concatenate the centerline to a single polyline and discretize it
        centerline = PolylineRegion.unionAll([lane.centerline for lane in route])
        points = centerline.pointsSeparatedBy(resolution)

        # route should start close to the vehicle's current location
        waypoints = [
            (actor.get_transform(), RoadOption.LANEFOLLOW)
        ]
        # Keep track of whether the waypoint is ahead or behind the vehicle (we start from behind)
        passed_self = False
        for point in points:
            loc = scenicToCarlaLocation(point, world=CarlaDataProvider.get_world())
            wp = map.get_waypoint(loc)

            if self._debug_mode:
                world = CarlaDataProvider.get_world()
                world.debug.draw_point(loc, size=0.1, color=carla.Color(0, 0, 255), life_time=1000)

            # If the waypoint is ahead of the vehicle, add it to the route
            is_ahead = is_within_distance_ahead(wp.transform, actor.get_transform(), max_distance=np.inf)
            if passed_self or is_ahead:
                passed_self = True
                waypoints.append((wp.transform, RoadOption.VOID))

        return waypoints


    def _create_behavior(self):
        scenic_behavior = ScenicSimulation(simulation=self._simulation, max_time_steps=self._max_time_steps)
        logging.debug("Finished creating behaviors for scenic scenario.")
        return scenic_behavior

    def _create_test_criteria(self):
        logging.debug("Creating test criteria.")
        criteria = py_trees.composites.Parallel(
            name="ScenicCriteria",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        for actor in self.ego_vehicles:
            actor_config = next(filter(lambda x: x.rolename == actor.attributes['role_name'], self.config.ego_vehicles))
            actor_criteria = self._create_tests_for_actor(actor, actor_config.route)
            criteria.add_child(actor_criteria)
        
        return criteria

    def _create_tests_for_actor(self, actor: carla.Vehicle, route=None) -> typing.List[atomic_criteria.Criterion]:
        collision_criterion = CollisionTest(actor, terminate_on_failure=False)
        red_light_criterion = atomic_criteria.RunningRedLightTest(actor)
        stop_criterion = atomic_criteria.RunningStopTest(actor)
        sidewalk_criterion = atomic_criteria.OnSidewalkTest(actor)
        wrong_lane_criterion = atomic_criteria.WrongLaneTest(actor)
        tests =  [
            collision_criterion,
            red_light_criterion,
            stop_criterion,
            sidewalk_criterion,
            wrong_lane_criterion
        ]
        if route:
            tests.extend([
                atomic_criteria.RouteCompletionTest(actor, route),
                # Disabled because it terminates the scenario on failure
                #atomic_criteria.InRouteTest(actor, route, terminate_on_failure=False)

            ])
        return py_trees.composites.Parallel(
            name=f"ScenicCriteria_{actor.attributes['role_name']}",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
            children=tests
        )
