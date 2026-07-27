from __future__ import annotations

import torch
from torch import nn


class WorldModel(nn.Module):
    """One-step latent world model for next-state and reward prediction."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.reward = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([state, action_features], dim=-1))
        return self.next_state(h), self.reward(h).squeeze(-1)

    def loss(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
        target_next_state: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> torch.Tensor:
        pred_state, pred_reward = self.forward(state, action_features)
        return nn.functional.mse_loss(pred_state, target_next_state) + nn.functional.mse_loss(
            pred_reward, target_reward
        )

