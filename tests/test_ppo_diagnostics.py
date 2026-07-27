import unittest

import torch

from edge_drl.agents.mappo import PPOUpdater, RolloutBuffer
from edge_drl.agents.networks import AgentSActorCritic


class PPODiagnosticsTest(unittest.TestCase):
    def test_update_reports_convergence_metrics(self):
        model = AgentSActorCritic(obs_dim=5, num_actions=4, hidden_dim=16)
        updater = PPOUpdater(model, lr=1e-4)
        buffer = RolloutBuffer()

        for index in range(8):
            obs = torch.randn(5)
            mask = torch.tensor([True, True, False, True])
            with torch.no_grad():
                logits = model.logits(obs.unsqueeze(0), mask.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().squeeze(0)
                log_prob = dist.log_prob(action.unsqueeze(0)).squeeze(0)
                value = model.value(obs.unsqueeze(0)).squeeze(0)
            buffer.obs.append(obs)
            buffer.masks.append(mask)
            buffer.actions.append(action)
            buffer.log_probs.append(log_prob)
            buffer.rewards.append(float(index) / 10.0)
            buffer.dones.append(index == 7)
            buffer.values.append(value)

        stats = updater.update(buffer, epochs=2)

        for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
        ):
            self.assertIn(name, stats)
            self.assertTrue(torch.isfinite(torch.tensor(stats[name])))


if __name__ == "__main__":
    unittest.main()
