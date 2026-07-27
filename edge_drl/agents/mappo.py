from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class RolloutBuffer:
    obs: list[torch.Tensor] = field(default_factory=list)
    masks: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear()
        self.masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class PPOUpdater:
    """Compact PPO update helper for masked actor-critic policies."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.001,
        normalize_rewards: bool = True,
    ):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_rewards = normalize_rewards

    def update(
        self,
        buffer: RolloutBuffer,
        epochs: int = 4,
    ) -> dict[str, float]:
        if not buffer.obs:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
            }

        device = next(self.model.parameters()).device
        obs = torch.stack(buffer.obs).to(device)
        masks = torch.stack(buffer.masks).to(device)
        actions = torch.stack(buffer.actions).to(device)
        old_log_probs = torch.stack(buffer.log_probs).detach().to(device)
        values = torch.stack(buffer.values).detach().to(device)
        rewards = torch.as_tensor(buffer.rewards, dtype=torch.float32, device=device)
        if self.normalize_rewards and rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        value_targets, advantages = self._gae_returns(rewards, buffer.dones, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        for _ in range(epochs):
            logits = self.model.logits(obs, masks)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs - old_log_probs)
            approx_kl = (old_log_probs - log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value = self.model.value(obs)
            value_loss = torch.mean((value - value_targets) ** 2)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            totals["loss"] += float(loss.detach().item())
            totals["policy_loss"] += float(policy_loss.detach().item())
            totals["value_loss"] += float(value_loss.detach().item())
            totals["entropy"] += float(entropy.detach().item())
            totals["approx_kl"] += float(approx_kl.detach().item())
            totals["clip_fraction"] += float(clip_fraction.detach().item())

        stats = {name: value / max(epochs, 1) for name, value in totals.items()}
        return_variance = torch.var(value_targets, unbiased=False)
        if return_variance > 1e-8:
            prediction_error = value_targets - values
            explained_variance = 1.0 - torch.var(prediction_error, unbiased=False) / return_variance
            stats["explained_variance"] = float(explained_variance.item())
        else:
            stats["explained_variance"] = 0.0
        return stats

    def _gae_returns(
        self,
        rewards: torch.Tensor,
        dones: list[bool],
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(values)
        last_advantage = torch.tensor(0.0, dtype=torch.float32, device=values.device)
        next_value = torch.tensor(0.0, dtype=torch.float32, device=values.device)
        for index in reversed(range(rewards.numel())):
            non_terminal = 0.0 if dones[index] else 1.0
            delta = rewards[index] + self.gamma * next_value * non_terminal - values[index]
            last_advantage = delta + self.gamma * self.gae_lambda * non_terminal * last_advantage
            advantages[index] = last_advantage
            next_value = values[index]
        returns = advantages + values
        return returns.detach(), advantages.detach()
