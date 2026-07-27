from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from edge_drl.agents.baselines import HeuristicScheduler
from edge_drl.agents.networks import AgentSActorCritic
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.requests import Task


@dataclass(slots=True)
class BCSamples:
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])

    def sample_torch_batch(self, batch_size: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.masks[idx], dtype=torch.bool, device=device),
            torch.as_tensor(self.actions[idx], dtype=torch.long, device=device),
        )


class RecordingHeuristicScheduler(HeuristicScheduler):
    """Heuristic teacher that records Agent-S-compatible supervised samples."""

    def __init__(self, env: EdgeComputingEnv, max_samples: int | None = None):
        super().__init__(env.path_manager, env.compute_capacity, env.bandwidth)
        self.max_samples = max_samples
        self.obs: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[int] = []
        self.use_top_k_mask = True

    def record_teacher_sample(self, task: Task, mask: np.ndarray, task_obs: np.ndarray, action: int) -> None:
        del task
        if self.max_samples is not None and len(self.actions) >= self.max_samples:
            return
        self.obs.append(task_obs.astype(np.float32))
        self.masks.append(mask.astype(bool))
        self.actions.append(int(action))

    def samples(self) -> BCSamples:
        if not self.actions:
            raise RuntimeError("No behavior-cloning samples were collected.")
        return BCSamples(
            obs=np.stack(self.obs).astype(np.float32),
            masks=np.stack(self.masks).astype(bool),
            actions=np.asarray(self.actions, dtype=np.int64),
        )


def collect_bc_samples(config: dict, seconds: int, max_samples: int | None = None) -> BCSamples:
    env = EdgeComputingEnv(config)
    env.reset()
    scheduler = RecordingHeuristicScheduler(env, max_samples=max_samples)
    for _ in range(seconds):
        _, _, done, _ = env.step(scheduler)
        if done or (max_samples is not None and len(scheduler.actions) >= max_samples):
            break
    return scheduler.samples()


def pretrain_agent_s_bc(
    model: AgentSActorCritic,
    samples: BCSamples,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, float]:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    obs = torch.as_tensor(samples.obs, dtype=torch.float32, device=device)
    masks = torch.as_tensor(samples.masks, dtype=torch.bool, device=device)
    actions = torch.as_tensor(samples.actions, dtype=torch.long, device=device)
    n = samples.size
    last_loss = 0.0
    last_acc = 0.0

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        correct = 0
        total = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model.logits(obs[idx], masks[idx])
            loss = nn.functional.cross_entropy(logits, actions[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            opt.step()
            pred = torch.argmax(logits.detach(), dim=-1)
            correct += int((pred == actions[idx]).sum().item())
            total += int(idx.numel())
            last_loss = float(loss.detach().cpu().item())
        last_acc = float(correct / max(total, 1))

    return {"bc_loss": last_loss, "bc_accuracy": last_acc, "bc_samples": float(n)}
