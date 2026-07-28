import tempfile
import unittest

import torch

from edge_drl.config import load_config
from edge_drl.training import IntegratedTrainer


class CheckpointRestoreTest(unittest.TestCase):
    def test_checkpoint_restores_model_and_ppo_lr(self):
        cfg = load_config("config/default.yaml")
        cfg["simulation"]["seconds_per_episode"] = 1
        trainer = IntegratedTrainer(cfg, device="cpu", ppo_lr=1e-4)
        original = next(trainer.agent_s.parameters()).detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/checkpoint.pt"
            trainer.save_checkpoint(path, episode=0)

            with torch.no_grad():
                next(trainer.agent_s.parameters()).add_(1.0)
            trainer.scale_ppo_lr(0.5)

            trainer.load_checkpoint(path)

        restored = next(trainer.agent_s.parameters()).detach()
        self.assertTrue(torch.allclose(restored, original))
        self.assertAlmostEqual(float(trainer.ppo.opt.param_groups[0]["lr"]), 1e-4)


if __name__ == "__main__":
    unittest.main()
