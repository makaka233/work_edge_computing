from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from edge_drl.agents.networks import AgentSActorCritic
from edge_drl.env.requests import Task


@dataclass(slots=True)
class DecisionRecord:
    obs: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


class TorchAgentSScheduler:
    """Adapter that makes Agent-S usable inside the event-driven environment."""

    def __init__(self, model: AgentSActorCritic, device: str = "cpu", deterministic: bool = False):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.decisions: list[DecisionRecord] = []
        self.action_ids: list[int] = []

    def reset_records(self) -> None:
        self.decisions.clear()
        self.action_ids.clear()

    def select_path_with_obs(self, task: Task, mask: np.ndarray, task_obs: np.ndarray) -> int:
        del task
        obs_t = torch.as_tensor(task_obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.model.logits(obs_t.unsqueeze(0), mask_t.unsqueeze(0)).squeeze(0)
            if self.deterministic:
                action = torch.argmax(logits)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            value = self.model.value(obs_t.unsqueeze(0)).squeeze(0)
        record = DecisionRecord(
            obs=obs_t.detach().cpu(),
            mask=mask_t.detach().cpu(),
            action=action.detach().cpu(),
            log_prob=log_prob.detach().cpu(),
            value=value.detach().cpu(),
        )
        self.decisions.append(record)
        action_id = int(action.item())
        self.action_ids.append(action_id)
        return action_id

    def action_features(self, num_actions: int, num_tasks: int, invalid_count: int) -> np.ndarray:
        if self.action_ids:
            mean_action = float(np.mean(self.action_ids)) / max(num_actions - 1, 1)
            std_action = float(np.std(self.action_ids)) / max(num_actions - 1, 1)
        else:
            mean_action = 0.0
            std_action = 0.0
        return np.array(
            [
                mean_action,
                std_action,
                float(num_tasks) / 1000.0,
                float(invalid_count) / 100.0,
            ],
            dtype=np.float32,
        )
