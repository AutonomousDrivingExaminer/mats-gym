import time
import unittest
import mats_gym
from pettingzoo.test import parallel_api_test
from .env_tests import seed_test
from mats_gym.envs import renderers
from . import utils
from mats_gym.servers import DockerCarlaServer


class TestScenicEnvAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.server = utils.start_server(port=2000)

    unittest.skip("Temporarily disabled.")
    def test_openscenario_env_api(self):
        env = mats_gym.openscenario_env(
            scenario_files="tests/scenarios/open_scenario/openscenario_test.xosc",
            host="localhost",
            port=2000,
            render_mode="rgb_array",
            timeout=10,
            render_config=renderers.camera_pov(agent="hero"),
        )
        parallel_api_test(env, num_cycles=100)

    @unittest.skip("Temporarily disabled.")
    def test_scenic_env_seeding(self):
        def make_env():
            return mats_gym.openscenario_env(
                scenario_files="tests/scenarios/open_scenario/openscenario_test.xosc",
                host="localhost",
                port=2000,
                render_mode="rgb_array",
                timeout=10,
                render_config=renderers.camera_pov(agent="hero"),
            )

        seed_test(make_env)

    def tearDown(self) -> None:
        self.server.stop()
        time.sleep(2.0)


if __name__ == "__main__":
    unittest.main()
