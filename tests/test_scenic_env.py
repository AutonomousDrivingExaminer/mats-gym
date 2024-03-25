import time
import unittest
import mats_gym
from pettingzoo.test import parallel_api_test

from tests import utils
from .env_tests import seed_test
from mats_gym.envs import renderers
from mats_gym.servers import DockerCarlaServer



class TestScenicEnvAPI(unittest.TestCase):

    def setUp(self) -> None:
        self.server = utils.start_server(port=2000, image="carlasim/carla:0.9.15")

    def test_scenic_env_api(self):
        env = mats_gym.scenic_env(
            host="localhost",
            port=2000,
            scenario_specification="tests/scenarios/scenic/simple_town05.scenic",
            scenes_per_scenario=1,
            resample_scenes=False,
            agent_name_prefixes=["car"],
            render_mode="rgb_array",
            render_config=renderers.camera_pov(agent="car_1"),
        )
        parallel_api_test(env, num_cycles=100)

    @unittest.skip("Temporarily disabled.")
    def test_scenic_env_seeding(self):
        def make_env():
            return mats_gym.scenic_env(
                host="localhost",
                port=2000,
                scenario_specification="tests/scenarios/simple_town05.scenic",
                scenes_per_scenario=1,
                resample_scenes=False,
                agent_name_prefixes=["car"],
                render_mode="rgb_array",
                render_config=renderers.camera_pov(agent="car_1"),
            )
        seed_test(make_env)

    def tearDown(self) -> None:
        self.server.stop()

if __name__ == "__main__":
    unittest.main()