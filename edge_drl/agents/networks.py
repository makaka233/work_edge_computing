from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2):
        super().__init__()
        blocks: list[nn.Module] = []
        last = input_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(last, hidden_dim), nn.ReLU()])
            last = hidden_dim
        blocks.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgentSActorCritic(nn.Module):
    """Masked path actor and centralized value head for task scheduling."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = MLP(obs_dim, hidden_dim, num_actions, layers=2)
        self.critic = MLP(obs_dim, hidden_dim, 1, layers=2)

    def logits(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.actor(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class AgentDActorCritic(nn.Module):
    """Deployment actor-critic that scores each service-stage-node placement."""

    def __init__(self, obs_dim: int, num_services: int, max_stages: int, num_nodes: int, hidden_dim: int = 256):
        super().__init__()
        self.num_services = num_services
        self.max_stages = max_stages
        self.num_nodes = num_nodes
        self.actor = MLP(obs_dim, hidden_dim, num_services * max_stages * num_nodes, layers=2)
        self.critic = MLP(obs_dim, hidden_dim, 1, layers=2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.placement_logits(obs)

    def placement_logits(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.actor(obs)
        return logits.view(-1, self.num_services, self.max_stages, self.num_nodes)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


AgentDActor = AgentDActorCritic
