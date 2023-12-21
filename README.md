# Multi-Agent Traffic Scenario Gym

MATS-Gym is a [PettingZoo](https://pettingzoo.farama.org/index.html) environment for training and evaluating autonomous driving agents in CARLA.
Environments can be created from scenarios that are implemented as [ScenarioRunner](https://github.com/carla-simulator/scenario_runner) scenarios,
allowing to leverage a large number of existing driving scenarios. 
Furthermore, we provide an integration with [Scenic](https://github.com/BerkeleyLearnVerify/Scenic) to which allows us to sample scenarios from Scenic specifications.

## Main Features
- Supports multiple scenario specification standards:
  - [Scenic](https://github.com/BerkeleyLearnVerify/Scenic)
  - [OpenSCENARIO](https://www.asam.net/standards/detail/openscenario/)
  - [ScenarioRunner](https://github.com/carla-simulator/scenario_runner)
  - Full compatibility with [CARLA Challenge](https://leaderboard.carla.org/challenge/) scenarios
- Multi-Agent environment
- Determinism and reproducibility (via replay functionality)
- Various types of observations:
  - [Birdview](https://github.com/deepsense-ai/carla-birdeye-view)
  - Sensor data (e.g. RGB camera, depth camera, lidar)
  - Map data (e.g. lane centerlines, lane boundaries, traffic lights)
- Action spaces:
  - High level meta-actions (e.g. turn left, turn right, go straight, change lane)
  - Low level continuous control (throttle, brake, steer)

## Installation
For now, you need to install the package from source. We recommend using a virtual environment.

To install the package, run the following command:
```bash
pip install git+https://github.com/AutonomousDrivingExaminer/mats-gym
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

The following code snippet, shows you how to use a scenic scenario to create an environment instance:

```python
import gymnasium
import mats_gym
from mats_gym.envs import renderers


env = mats_gym.scenic_env(
  host="localhost",  # The host to connect to
  port=2000,  # The port to connect to
  scenario_specification="scenarios/scenic/carla_challenge_08.scenic",
  scenes_per_scenario=5,  # How many scenes should be generated per scenario
  agent_name_prefixes=["sut", "adv"], # Each actor whose role-name starts with one of the prefixes is an agent.
  render_mode="human",  # The render mode. Can be "human" or "rgb_array".
  render_config=renderers.camera_pov(agent="sut"),
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