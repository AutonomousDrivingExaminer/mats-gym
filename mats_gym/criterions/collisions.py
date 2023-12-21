import weakref
import math
import numpy as np
import py_trees
import shapely.geometry

from srunner.scenariomanager.scenarioatomics import atomic_criteria
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType
import carla


class CollisionTest(atomic_criteria.Criterion):

    """
    This class contains an atomic test for collisions.

    Args:
    - actor (carla.Actor): CARLA actor to be used for this test
    - other_actor (carla.Actor): only collisions with this actor will be registered
    - other_actor_type (str): only collisions with actors including this type_id will count.
        Additionally, the "miscellaneous" tag can also be used to include all static objects in the scene
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    MIN_AREA_OF_COLLISION = 3       # If closer than this distance, the collision is ignored
    MAX_AREA_OF_COLLISION = 5       # If further than this distance, the area is forgotten
    MAX_ID_TIME = 5                 # Amount of time the last collision if is remembered

    def __init__(self, actor, other_actor=None, other_actor_type=None,
                 optional=False, name="CollisionTest", terminate_on_failure=False):
        """
        Construction with sensor setup
        """
        super(CollisionTest, self).__init__(name, actor, optional, terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        world = self.actor.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._collision_sensor.listen(lambda event: self._count_collisions(weakref.ref(self), event))

        self.other_actor = other_actor
        self.other_actor_type = other_actor_type
        self.registered_collisions = []
        self.last_id = None
        self.collision_time = None

    def update(self):
        """
        Check collision count
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        actor_location = CarlaDataProvider.get_location(self.actor)
        new_registered_collisions = []

        # Loops through all the previous registered collisions
        for collision_location in self.registered_collisions:

            # Get the distance to the collision point
            distance_vector = actor_location - collision_location
            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

            # If far away from a previous collision, forget it
            if distance <= self.MAX_AREA_OF_COLLISION:
                new_registered_collisions.append(collision_location)

        self.registered_collisions = new_registered_collisions

        if self.last_id and GameTime.get_time() - self.collision_time > self.MAX_ID_TIME:
            self.last_id = None

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self._collision_sensor is not None:
            self._collision_sensor.destroy()
        self._collision_sensor = None

        super(CollisionTest, self).terminate(new_status)

    @staticmethod
    def _count_collisions(weak_self, event: carla.CollisionEvent):     # pylint: disable=too-many-return-statements
        """
        Callback to update collision count
        """
    
        self = weak_self()
        if not self:
            return

        actor_location = CarlaDataProvider.get_location(self.actor)

        # Ignore the current one if it is the same id as before
        if self.last_id == event.other_actor.id:
            return

        # Filter to only a specific actor
        if self.other_actor and self.other_actor.id != event.other_actor.id:
            return

        # Filter to only a specific type
        if self.other_actor_type:
            if self.other_actor_type == "miscellaneous":
                if "traffic" not in event.other_actor.type_id \
                        and "static" not in event.other_actor.type_id:
                    return
            else:
                if self.other_actor_type not in event.other_actor.type_id:
                    return

        # Ignore it if its too close to a previous collision (avoid micro collisions)
        for collision_location in self.registered_collisions:

            distance_vector = actor_location - collision_location
            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

            if distance <= self.MIN_AREA_OF_COLLISION:
                return
        
        velocity = event.actor.get_velocity()
        impulse = event.normal_impulse
        collision_info = {
            'type': event.other_actor.type_id,
            'id': event.other_actor.id,
            'velocity': [velocity.x, velocity.y, velocity.z],
            'x': actor_location.x,
            'y': actor_location.y,
            'z': actor_location.z,
            'impulse': [impulse.x, impulse.y, impulse.z],
            'impulse_magnitude': math.sqrt(math.pow(impulse.x, 2) + math.pow(impulse.y, 2) + math.pow(impulse.z, 2)),
        }

        if ('static' in event.other_actor.type_id or 'traffic' in event.other_actor.type_id) \
                and 'sidewalk' not in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_STATIC
        elif 'vehicle' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_VEHICLE
            other_velocity = event.other_actor.get_velocity()
            collision_info['other_velocity'] = [other_velocity.x, other_velocity.y, other_velocity.z]
        elif 'walker' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_PEDESTRIAN
        else:
            return

        collision_event = TrafficEvent(event_type=actor_type, frame=GameTime.get_frame())
        collision_event.set_dict(collision_info)
        collision_event.set_message(
            "Agent collided against object with type={} and id={} at (x={}, y={}, z={})".format(
                event.other_actor.type_id,
                event.other_actor.id,
                round(actor_location.x, 3),
                round(actor_location.y, 3),
                round(actor_location.z, 3)))

        self.test_status = "FAILURE"
        self.actual_value += 1
        self.collision_time = GameTime.get_time()

        self.registered_collisions.append(actor_location)
        self.events.append(collision_event)

        # Number 0: static objects -> ignore it
        if event.other_actor.id != 0:
            self.last_id = event.other_actor.id