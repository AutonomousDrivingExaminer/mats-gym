from typing import NamedTuple, List, Callable

import carla

VehicleClassificationFn = Callable[[carla.Vehicle], bool]
is_vehicle = lambda actor: "vehicle" in actor.type_id
is_pedestrian = lambda actor: "pedestrian" in actor.type_id
is_traffic_light = lambda actor: "traffic_light" in actor.type_id


class SegregatedActors(NamedTuple):
    vehicle_classes: List[List[carla.Actor]]
    pedestrians: List[carla.Actor]
    traffic_lights: List[carla.Actor]


def segregate_by_type(
    actors: List[carla.Actor],
    vehicle_classification_fns: List[VehicleClassificationFn] = None,
) -> SegregatedActors:
    if vehicle_classification_fns is None:
        vehicle_classification_fns = []

    # Unclassified vehicles are put in the first class
    vehicle_classes = [[] for _ in range(len(vehicle_classification_fns) + 1)]
    traffic_lights = []
    pedestrians = []
    for actor in actors:
        if is_traffic_light(actor):
            traffic_lights.append(actor)
            continue
        if is_pedestrian(actor):
            pedestrians.append(actor)
            continue
        if is_vehicle(actor):
            in_at_least_one_class = False
            for i, fn in enumerate(vehicle_classification_fns):
                if fn(actor):
                    in_at_least_one_class = True
                    vehicle_classes[i + 1].append(actor)
            if not in_at_least_one_class:
                vehicle_classes[0].append(actor)

    return SegregatedActors(
        vehicle_classes=vehicle_classes,
        pedestrians=pedestrians,
        traffic_lights=traffic_lights,
    )


def query_all(world: carla.World) -> List[carla.Actor]:
    snapshot: carla.WorldSnapshot = world.get_snapshot()
    all_actors = []
    for actor_snapshot in snapshot:
        actor = world.get_actor(actor_snapshot.id)
        if actor is not None:
            all_actors.append(actor)
    return all_actors
