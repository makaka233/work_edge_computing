import unittest

from edge_drl.agents.baselines import HeuristicScheduler
from edge_drl.agents.networks import AgentSActorCritic
from edge_drl.agents.torch_schedulers import TorchAgentSScheduler
from edge_drl.config import load_config
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.training import IntegratedTrainer


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

    def test_neural_scheduler_uses_top_k_mask(self):
        cfg = load_config("config/default.yaml")
        cfg["simulation"]["seconds_per_episode"] = 1
        cfg["simulation"]["agent_s_top_k_actions"] = 3
        env = EdgeComputingEnv(cfg)
        env.reset()
        model = AgentSActorCritic(env.task_obs_dim, env.path_manager.num_actions, hidden_dim=64)
        scheduler = TorchAgentSScheduler(model, deterministic=True)

        env.step(scheduler)

        self.assertGreater(len(scheduler.decisions), 0)
        self.assertTrue(all(int(record.mask.sum().item()) <= 3 for record in scheduler.decisions))

    def test_second_reward_is_shared_across_task_decisions(self):
        cfg = load_config("config/default.yaml")
        cfg["simulation"]["seconds_per_episode"] = 1
        trainer = IntegratedTrainer(cfg, device="cpu")
        trainer.env.reset()
        trainer.scheduler.reset_records()

        _, reward, done, _ = trainer.env.step(trainer.scheduler)
        trainer._append_scheduler_records(reward, done)

        self.assertGreater(len(trainer.rollout.rewards), 0)
        self.assertAlmostEqual(sum(trainer.rollout.rewards), reward, places=6)


if __name__ == "__main__":
    unittest.main()
