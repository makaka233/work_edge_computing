import unittest

from edge_drl.agents.baselines import HeuristicScheduler
from edge_drl.config import load_config
from edge_drl.env.environment import EdgeComputingEnv


class EnvironmentSmokeTest(unittest.TestCase):
    def test_environment_runs_seconds(self):
        cfg = load_config("config/default.yaml")
        cfg["simulation"]["seconds_per_episode"] = 5
        env = EdgeComputingEnv(cfg)
        scheduler = HeuristicScheduler(env.path_manager, env.compute_capacity, env.bandwidth)
        obs = env.reset()
        self.assertGreater(obs.shape[0], 0)
        total_tasks = 0
        for _ in range(5):
            obs, reward, done, info = env.step(scheduler)
            self.assertGreater(obs.shape[0], 0)
            self.assertIsInstance(reward, float)
            total_tasks += info["num_tasks"]
        self.assertTrue(done)
        self.assertGreaterEqual(total_tasks, 0)


if __name__ == "__main__":
    unittest.main()

