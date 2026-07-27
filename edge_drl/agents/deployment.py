from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from edge_drl.agents.networks import AgentDActorCritic


@dataclass(slots=True)
class DeploymentDecision:
    obs: torch.Tensor
    raw_action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    deployment: np.ndarray


class TorchAgentDDeployer:
    """Agent-D adapter for slow-timescale service deployment decisions."""

    def __init__(
        self,
        model: AgentDActorCritic,
        service_stage_mask: np.ndarray,
        stage_memory: np.ndarray,
        stage_storage: np.ndarray,
        memory_capacity: np.ndarray,
        storage_capacity: np.ndarray,
        max_replicas: int,
        device: str = "cpu",
        deterministic: bool = False,
    ):
        self.model = model.to(device)
        self.service_stage_mask = service_stage_mask.astype(bool)
        self.stage_memory = stage_memory
        self.stage_storage = stage_storage
        self.memory_capacity = memory_capacity
        self.storage_capacity = storage_capacity
        self.max_replicas = int(max_replicas)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.last_decision: DeploymentDecision | None = None

    def decide(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model.placement_logits(obs_t).squeeze(0)
            probs = torch.sigmoid(logits)
            active_mask = torch.as_tensor(self.service_stage_mask, dtype=torch.bool, device=self.device).unsqueeze(-1)
            if self.deterministic:
                raw = probs >= 0.5
            else:
                raw = torch.bernoulli(probs).bool()
            raw = raw & active_mask
            dist = torch.distributions.Bernoulli(probs=probs.clamp(1e-6, 1.0 - 1e-6))
            log_prob = dist.log_prob(raw.float())[active_mask.expand_as(raw)].sum()
            entropy = dist.entropy()[active_mask.expand_as(raw)].sum()
            value = self.model.value(obs_t).squeeze(0)

        deployment = self._repair(raw.detach().cpu().numpy().astype(np.int64), probs.detach().cpu().numpy())
        self.last_decision = DeploymentDecision(
            obs=obs_t.squeeze(0).detach().cpu(),
            raw_action=raw.detach().cpu(),
            log_prob=log_prob.detach().cpu(),
            entropy=entropy.detach().cpu(),
            value=value.detach().cpu(),
            deployment=deployment,
        )
        return deployment

    def _repair(self, raw: np.ndarray, probs: np.ndarray) -> np.ndarray:
        deployment = raw.copy()
        deployment[~self.service_stage_mask] = 0

        for i, j in np.argwhere(self.service_stage_mask):
            selected = np.flatnonzero(deployment[i, j])
            if selected.size > self.max_replicas:
                keep = selected[np.argsort(-probs[i, j, selected])[: self.max_replicas]]
                deployment[i, j] = 0
                deployment[i, j, keep] = 1
            elif selected.size == 0:
                best = int(np.argmax(probs[i, j]))
                deployment[i, j, best] = 1

        deployment = self._prune_resource_excess(deployment, probs)
        for i, j in np.argwhere(self.service_stage_mask):
            if deployment[i, j].sum() == 0:
                self._force_best_feasible(deployment, probs, int(i), int(j))
        return deployment.astype(np.int64)

    def _resource_usage(self, deployment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        memory = np.einsum("ijn,ij->n", deployment, self.stage_memory)
        storage = np.einsum("ijn,ij->n", deployment, self.stage_storage)
        return memory, storage

    def _prune_resource_excess(self, deployment: np.ndarray, probs: np.ndarray) -> np.ndarray:
        memory, storage = self._resource_usage(deployment)
        while np.any(memory > self.memory_capacity) or np.any(storage > self.storage_capacity):
            overloaded = np.flatnonzero((memory > self.memory_capacity) | (storage > self.storage_capacity))
            candidates = []
            for node in overloaded:
                for i, j in np.argwhere(deployment[:, :, node] > 0):
                    if deployment[i, j].sum() <= 1:
                        continue
                    candidates.append((float(probs[i, j, node]), int(i), int(j), int(node)))
            if not candidates:
                break
            _, i, j, node = min(candidates, key=lambda item: item[0])
            deployment[i, j, node] = 0
            memory, storage = self._resource_usage(deployment)
        return deployment

    def _force_best_feasible(self, deployment: np.ndarray, probs: np.ndarray, service: int, stage: int) -> None:
        memory, storage = self._resource_usage(deployment)
        order = np.argsort(-probs[service, stage])
        mem_req = self.stage_memory[service, stage]
        st_req = self.stage_storage[service, stage]
        for node in order:
            if memory[node] + mem_req <= self.memory_capacity[node] and storage[node] + st_req <= self.storage_capacity[node]:
                deployment[service, stage, node] = 1
                return
        deployment[service, stage, int(order[0])] = 1


class DeploymentActorCriticUpdater:
    """Slow-timescale actor-critic update for Agent-D."""

    def __init__(self, model: AgentDActorCritic, lr: float = 3e-4, value_coef: float = 0.5, entropy_coef: float = 0.001):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, decisions: list[DeploymentDecision], returns: list[float]) -> dict[str, float]:
        if not decisions:
            return {"agent_d_loss": 0.0}
        obs = torch.stack([d.obs for d in decisions])
        raw = torch.stack([d.raw_action.float() for d in decisions])
        target = torch.as_tensor(returns, dtype=torch.float32, device=obs.device)
        logits = self.model.placement_logits(obs)
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        dist = torch.distributions.Bernoulli(probs=probs)
        log_probs = dist.log_prob(raw).sum(dim=(1, 2, 3))
        entropy = dist.entropy().sum(dim=(1, 2, 3)).mean()
        values = self.model.value(obs)
        advantages = target - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = torch.mean((values - target) ** 2)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return {"agent_d_loss": float(loss.detach().cpu().item())}
