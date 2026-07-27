import unittest

from edge_drl.agents.networks import AgentSActorCritic
from edge_drl.config import load_config
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.training.behavior_cloning import collect_bc_samples, pretrain_agent_s_bc


class BehaviorCloningTest(unittest.TestCase):
    def test_collect_and_pretrain(self):
        cfg = load_config("config/default.yaml")
        cfg["simulation"]["seconds_per_episode"] = 2
        cfg["simulation"]["agent_s_top_k_actions"] = 8
        samples = collect_bc_samples(cfg, seconds=1, max_samples=64)
        self.assertGreater(samples.size, 0)
        self.assertLessEqual(int(samples.masks.sum(axis=1).max()), 8)
        env = EdgeComputingEnv(cfg)
        env.reset()
        model = AgentSActorCritic(env.task_obs_dim, env.path_manager.num_actions, hidden_dim=64)
        stats = pretrain_agent_s_bc(model, samples, epochs=1, batch_size=32)
        self.assertIn("bc_loss", stats)
        self.assertIn("bc_accuracy", stats)
        batch = samples.sample_torch_batch(8, "cpu")
        self.assertEqual(batch[0].shape[0], min(8, samples.size))


if __name__ == "__main__":
    unittest.main()
