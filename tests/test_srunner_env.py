import time
import unittest
import mats_gym
from pettingzoo.test import parallel_api_test
from .env_tests import seed_test
from mats_gym.envs import renderers
from mats_gym.servers import DockerCarlaServer


def _start_server(port: int):
    server = DockerCarlaServer(
        image="carlasim/carla:0.9.13",
        world_port=port,
        headless=True,
        container_name=f"carla_server_{port}",
    )
    server.start()
    return server


class TestScenarioRunnerEnvAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _start_server(2000)
        self.server.start()
        time.sleep(5.0)

    @unittest.skip("Temporarily disabled.")
    def test_srunner_env_api(self):
        env = mats_gym.srunner_env(
            host="localhost",
            port=2000,
            scenario_name="group:srunner.scenarios.cut_in.CutIn",
            config_file="tests/scenarios/CutIn.xml",
            render_mode="human",
            timeout=10,
            render_config=renderers.camera_pov(agent="scenario"),
        )
        parallel_api_test(env, num_cycles=100)

    @unittest.skip("Temporarily disabled.")
    def test_srunner_env_seeding(self):
        def make_env():
            return mats_gym.srunner_env(
                host="localhost",
                port=2000,
                scenario_name="group:srunner.scenarios.cut_in.CutIn",
                config_file="tests/scenarios/CutIn.xml",
                render_mode="human",
                timeout=10,
                render_config=renderers.camera_pov(agent="scenario"),
            )

        seed_test(make_env)

    def tearDown(self) -> None:
        self.server.stop()
        time.sleep(2.0)

if __name__ == "__main__":
    unittest.main()
