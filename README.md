# Multi-Agent Traffic Scenario Gym

This gym environment allows you to train autonomous driving agents to navigate traffic scenarios in the CARLA simulator.
Scenario Gym connects multiple scenario generation engines for scenario based testing and exposes agent control via a 
Gymnasium and PettingZoo interface.
Currently we support scenarios that are expressed in [Scenic](https://github.com/BerkeleyLearnVerify/Scenic),
[ScenarioRunner](https://github.com/carla-simulator/scenario_runner) and [OpenSCENARIO](https://www.asam.net/standards/detail/openscenario/) (via ScenarioRunner).

The main features of this environment are:
- Supports multiple scenario generation engines
  - Scenic, OpenScenario, ScenarioRunner
  - Full compatibility with [CARLA Challenge](https://leaderboard.carla.org/challenge/) scenarios
- Multi-Agent environment 
- Determinism and reproducibility (via replay functionality)
- Useful wrappers included:
  - Observation space wrapper for [birdview observations](https://github.com/deepsense-ai/carla-birdeye-view)
  - High level meta-actions (e.g. turn left, turn right, go straight)
  - Autonomous agent wrapper to easily run a pre-trained autonomous agent

## Installation
For now, you need to install the package from source. We recommend using a virtual environment.

To install the package, run the following command:
```bash
pip install git+https://github.com/AutonomousDrivingExaminer/adex-gym
```

## Usage

### Overview
The main idea of Scenario Gym is to run scenarios that are implemented as subclasses of [BasicScenario](https://carla-scenariorunner.readthedocs.io/en/latest/creating_new_scenario/), 
from the [ScenarioRunner](https://https://github.com/carla-simulator/scenario_runner) package. 
The main class, [BaseScenarioEnv](mats_gym/envs/base_env.py), handles most of the logic for running scenarios and
controlling the agents. 
Furthermore, we provide adapters that handle sampling and initialization of scenarios from configuration or scenario files:
- [ScenicEnv](mats_gym/envs/adapters/scenic_env.py): samples scenes from Scenic scenarios and initializes [ScenicScenario](mats_gym/scenarios/scenic_scenario.py) instances.
- [ScenarioRunnerEnv](mats_gym/envs/adapters/scenario_runner_env.py): creates scenarios from scenario_runner configurations.

### Examples
First, make sure that CARLA is running (if you have docker installed, you can use `./scripts/start-carla.sh` to run it).

The following code snippet, shows you how the general workflow of using the environment looks like. There we first implement
a function `make_scenario` that handles the setup and initialization of the scenario. Then we pass this function to the
`BaseScenarioEnv` class, which handles the rest of the simulation.

```python
import gymnasium
import mats_gym
from mats_gym.envs import renderers


def make_scenario() -> BasicScenario:
  scenario = ...
  return scenario


env = gymnasium.make(
  id="ScenarioEnv-v0",
  scenario_fn=make_scenario,
  render_mode="human",
  render_config=renderers.camera_pov()
)

obs, info = env.reset()
terminated = False
while not terminated:
  action = {agent: ... for agent in env.agents}
  obs, reward, terminated, truncated, info = env.step(action)
...
env.close()
```

For more examples, have a look at the [examples](mats_gym/examples) directory.

### Gymnasium Interface: Action, Observations and Rewards
**Action Space:**
The unwrapped action space is a dictionary which maps the role-name of the agent to a numpy array of dimension 3.
The action will be mapped to a `carla.VehicleControl` object. The dimensions of the array are mapped as follows:
```python
vehicle_control = carla.VehicleControl(
    throttle=action[0],
    steer=action[1],
    brake=action[2],
)
```
The bounds of the action space for each agent are:

| Action   |min|max|type|
|----------|---|---|---|
| Throttle |0  |1  | float32 |
| Steer    |-1 |1  | float32 |
| Brake    |0  |1  | float32 |

For instance, the action space of an environment with three agents, could look like this:
```python
Dict({
  "adv_1": Box(low=[0.,-1.,0.], high=[1.0,-1.0, 0.0], shape=(3,), dtype=float32), 
  "adv_2": Box(low=[0.,-1.,0.], high=[1.0,-1.0, 0.0], shape=(3,), dtype=float32), 
  "sut": Box(low=[0.,-1.,0.], high=[1.0,-1.0, 0.0], shape=(3,), dtype=float32)
})
```

**Observation Space:**
The observation space is a dictionary which maps the role-name of the agent to a numpy array of dimension 6, which
contains the following information:
- x, y, z position of the agent
- x, y, z velocity component of the agent

Wrappers, such as the [BirdViewObservationWrapper](mats_gym/wrappers/birdview.py) can be used to transform the observation
space. For more information, have a look at the class definitions.

**Reward Function:**
The reward function resembles the evaluation score of the carla leaderboard challenge. Based on the evaluation criteria
for each agent, the reward is calculated according to the [driving score](https://leaderboard.carla.org/#evaluation-and-metrics).
Note that rewards are only computed at the end of the episode. This can be changed by using wrappers for instance.

**Info dictionary:**
The info dictionary contains lots additional information about the environment. 
The following keys are always present:
- `current_frame`: The current frame of the simulation
- `history`: The current recorder log of the carla simulation (see [https://carla.readthedocs.io/en/latest/adv_recorder/](https://carla.readthedocs.io/en/latest/adv_recorder/))
- `agent_info`: A dictionary that holds agent-specific information. Each agent has a dictionary with the following keys:
  - `events`: A list of traffic events (tracked by the scenario) t*hat occurred during the episode.*
  - `eval_info`: A dictionary that holds the evaluation information for the agent. This is only present at the end of the episode.