from __future__ import annotations

import re

import carla


class Frame:
    def __init__(self, frame_info: str):
        self._frame_info = frame_info
        self._sections = self._get_sections(frame_info)

    def _get_sections(self, frame_info: str):
        sections = re.findall(r" (.*): .*\n((?:  .*\n)*)", self._frame_info)
        section_dict = {}
        for section, content in sections:
            section_dict[section] = content
        return section_dict

    @property
    def vehicle_controls(self) -> dict[int, carla.VehicleControl]:
        control_section = self._sections["Vehicle animations"]
        control_list = re.findall(
            r"Id: (\d+) Steering: (.*) Throttle: (.*) Brake:? (.*) Handbrake: ([0,1]) Gear: (\d+)",
            control_section,
        )
        controls = {}
        for control in control_list:
            id, steer, throttle, brake, handbrake, gear = control
            controls[int(id)] = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
                hand_brake=bool(int(handbrake)),
                gear=int(gear),
            )
        return controls

    @property
    def vehicle_light_states(self) -> dict[int, carla.VehicleLightState]:
        vehicle_light_section = self._sections["Vehicle light animations"]
        state_list = re.findall(r"Id: (\d+) (.*)\n", vehicle_light_section)
        light_states = {}
        for id, state in state_list:
            if state == "None":
                state = "NONE"
            light_states[int(id)] = carla.VehicleLightState.names[state]
        return light_states

    @property
    def traffic_light_states(self) -> dict[int, carla.TrafficLightState]:
        traffic_light_section = self._sections["State traffic lights"]
        states = re.findall(
            r"Id: (\d+) state: (.*) frozen: (.*) elapsedTime: (\d+)",
            traffic_light_section,
        )
        traffic_lights = {}
        for tl_state in states:
            id, state, frozen, elapsed_time = tl_state
            traffic_lights[int(id)] = carla.TrafficLightState(int(state))
        return traffic_lights


class SimulationHistory:
    def __init__(self, history: str):
        self._history = history
        self._frames = self._get_frames(history)

    def __len__(self):
        return len(self._frames)

    @property
    def role_names(self) -> dict[int, str]:
        pattern = r"Create (\d+): vehicle.*\n(?:  .*\n)*  role_name = (\w+)"
        role_names = {}
        for match in re.findall(pattern, self._history):
            id, role_name = match
            role_names[int(id)] = role_name
        return role_names

    @property
    def frames(self) -> list[Frame]:
        return self._frames

    def __str__(self):
        return self._history

    def _get_frames(self, history: str) -> list[Frame]:
        frames = re.findall(r"Frame (\d+) at (.+) seconds\n((?:[ ]+.*\n)*)", history)
        return [Frame(frame) for i, time, frame in frames]
