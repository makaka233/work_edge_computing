from __future__ import annotations

from dataclasses import dataclass, field
import sys

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - only used when tqdm is unavailable.
    tqdm = None


@dataclass
class RolloutBuffer:
    states: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        state: np.ndarray,
        mask: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.masks.append(np.asarray(mask, dtype=bool))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def extend_rewards_for_pending(self, reward: float, done: bool) -> None:
        missing = len(self.actions) - len(self.rewards)
        for _ in range(missing):
            self.rewards.append(float(reward))
            self.dones.append(bool(done))

    def clear(self) -> None:
        self.states.clear()
        self.masks.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.actions)


class MaskedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor, masks: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.shared(states)
        logits = self.actor(features)
        safe_masks = masks.bool()
        fallback = torch.zeros_like(safe_masks)
        fallback[:, 0] = True
        safe_masks = torch.where(safe_masks.any(dim=1, keepdim=True), safe_masks, fallback)
        logits = logits.masked_fill(~safe_masks, -1e9)
        dist = Categorical(logits=logits)
        values = self.critic(features).squeeze(-1)
        return dist, values


class NodeScoringActorCritic(nn.Module):
    """Shared node encoder with per-candidate scoring.

    The state layout is `[global_features, node_0_features, ..., node_N_features]`.
    This avoids treating node ids as unrelated class labels and matches the
    device/candidate scoring pattern used by successful graph/resource DRL code.
    """

    def __init__(
        self,
        *,
        global_dim: int,
        node_feature_dim: int,
        num_nodes: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.global_dim = global_dim
        self.node_feature_dim = node_feature_dim
        self.num_nodes = num_nodes
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, masks: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        global_features = states[:, : self.global_dim]
        node_features = states[:, self.global_dim :].reshape(
            -1,
            self.num_nodes,
            self.node_feature_dim,
        )
        global_emb = self.global_encoder(global_features)
        node_emb = self.node_encoder(node_features)
        expanded_global = global_emb.unsqueeze(1).expand(-1, self.num_nodes, -1)
        pair_emb = torch.cat([expanded_global, node_emb], dim=-1)
        logits = self.scorer(pair_emb).squeeze(-1)

        safe_masks = masks.bool()
        fallback = torch.zeros_like(safe_masks)
        fallback[:, 0] = True
        safe_masks = torch.where(safe_masks.any(dim=1, keepdim=True), safe_masks, fallback)
        logits = logits.masked_fill(~safe_masks, -1e9)
        dist = Categorical(logits=logits)

        mask_float = safe_masks.float().unsqueeze(-1)
        pooled_nodes = (node_emb * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        values = self.critic(torch.cat([global_emb, pooled_nodes], dim=-1)).squeeze(-1)
        return dist, values


class GraphAttentionNodeScoringActorCritic(nn.Module):
    """Topology-aware node scorer with masked graph attention.

    State layout:
    `[global_features, node_features, edge_features]`, where edge features are
    flattened as `[src_node, dst_node, edge_feature]`.
    """

    def __init__(
        self,
        *,
        global_dim: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        num_nodes: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.global_dim = global_dim
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_nodes = num_nodes
        self.node_offset = global_dim
        self.edge_offset = global_dim + num_nodes * node_feature_dim

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.attn_src = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, masks: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        global_features = states[:, : self.global_dim]
        node_features = states[:, self.node_offset : self.edge_offset].reshape(
            -1,
            self.num_nodes,
            self.node_feature_dim,
        )
        edge_features = states[:, self.edge_offset :].reshape(
            -1,
            self.num_nodes,
            self.num_nodes,
            self.edge_feature_dim,
        )

        global_emb = self.global_encoder(global_features)
        node_emb = self.node_encoder(node_features)
        graph_emb = self._graph_attention(node_emb, edge_features)

        expanded_global = global_emb.unsqueeze(1).expand(-1, self.num_nodes, -1)
        pair_emb = torch.cat([expanded_global, graph_emb], dim=-1)
        logits = self.scorer(pair_emb).squeeze(-1)

        safe_masks = masks.bool()
        fallback = torch.zeros_like(safe_masks)
        fallback[:, 0] = True
        safe_masks = torch.where(safe_masks.any(dim=1, keepdim=True), safe_masks, fallback)
        logits = logits.masked_fill(~safe_masks, -1e9)
        dist = Categorical(logits=logits)

        mask_float = safe_masks.float().unsqueeze(-1)
        pooled_nodes = (graph_emb * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        values = self.critic(torch.cat([global_emb, pooled_nodes], dim=-1)).squeeze(-1)
        return dist, values

    def _graph_attention(self, node_emb: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        adjacency = edge_features[..., 0] > 0.5
        eye = torch.eye(self.num_nodes, dtype=torch.bool, device=edge_features.device).unsqueeze(0)
        adjacency = adjacency | eye

        src_score = self.attn_src(node_emb).expand(-1, -1, self.num_nodes)
        dst_score = self.attn_dst(node_emb).transpose(1, 2).expand(-1, self.num_nodes, -1)
        scores = torch.nn.functional.leaky_relu(src_score + dst_score + self.edge_bias(edge_features).squeeze(-1), 0.2)
        scores = scores.masked_fill(~adjacency, -1e9)
        attention = torch.softmax(scores, dim=-1)
        messages = torch.matmul(attention, self.message(node_emb))
        return self.update(torch.cat([node_emb, messages], dim=-1))


class PPOAgent:
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        target_kl: float | None = 0.03,
        minibatch_size: int = 512,
        policy_kind: str = "flat",
        global_dim: int | None = None,
        node_feature_dim: int | None = None,
        edge_feature_dim: int | None = None,
        num_nodes: int | None = None,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.target_kl = target_kl
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)
        if policy_kind == "flat":
            self.policy = MaskedActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        elif policy_kind == "node_scorer":
            if global_dim is None or node_feature_dim is None or num_nodes is None:
                raise ValueError("node_scorer requires global_dim, node_feature_dim, and num_nodes")
            self.policy = NodeScoringActorCritic(
                global_dim=global_dim,
                node_feature_dim=node_feature_dim,
                num_nodes=num_nodes,
                hidden_dim=hidden_dim,
            ).to(self.device)
        elif policy_kind == "gat_node_scorer":
            if global_dim is None or node_feature_dim is None or edge_feature_dim is None or num_nodes is None:
                raise ValueError("gat_node_scorer requires global_dim, node_feature_dim, edge_feature_dim, and num_nodes")
            self.policy = GraphAttentionNodeScoringActorCritic(
                global_dim=global_dim,
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                num_nodes=num_nodes,
                hidden_dim=hidden_dim,
            ).to(self.device)
        else:
            raise ValueError(f"unknown policy_kind: {policy_kind}")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def act(self, state: np.ndarray, mask: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state_t, mask_t)
            action_t = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
            logprob_t = dist.log_prob(action_t)
        return int(action_t.item()), float(logprob_t.item()), float(value.item())

    def action_stats(self, state: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state_t, mask_t)
            probs = dist.probs.squeeze(0)
            sorted_probs, sorted_actions = torch.sort(probs, descending=True)
            top1 = float(sorted_probs[0].item())
            top2 = float(sorted_probs[1].item()) if sorted_probs.numel() > 1 else 0.0
            action = int(sorted_actions[0].item())
            entropy = float(dist.entropy().item())
        return {
            "action": action,
            "entropy": entropy,
            "top1_prob": top1,
            "top1_margin": top1 - top2,
            "value": float(value.item()),
        }

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        if len(self.buffer) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        if len(self.buffer.rewards) != len(self.buffer.actions):
            raise ValueError("buffer contains pending transitions without rewards")

        advantages_np, returns_np = self._gae_advantages_and_returns()
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

        actions_np = np.asarray(self.buffer.actions, dtype=np.int64)
        old_logprobs_np = np.asarray(self.buffer.logprobs, dtype=np.float32)
        n = len(actions_np)
        minibatch_size = max(1, min(self.minibatch_size, n))
        batches_per_epoch = int(np.ceil(n / minibatch_size))
        progress = None
        if tqdm is not None and progress_interval_seconds > 0 and progress_label:
            progress = tqdm(
                total=self.k_epochs * batches_per_epoch,
                desc=progress_label,
                unit="mb",
                dynamic_ncols=True,
                mininterval=progress_interval_seconds,
                leave=True,
                bar_format="{desc}: {percentage:5.1f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                file=sys.stdout,
            )

        last_metrics: dict[str, float] = {}
        for epoch_idx in range(self.k_epochs):
            order = np.random.permutation(n)
            approx_kls: list[float] = []
            for start in range(0, n, minibatch_size):
                idx = order[start : start + minibatch_size]
                states = torch.as_tensor(np.stack([self.buffer.states[i] for i in idx]), dtype=torch.float32, device=self.device)
                masks = torch.as_tensor(np.stack([self.buffer.masks[i] for i in idx]), dtype=torch.bool, device=self.device)
                actions = torch.as_tensor(actions_np[idx], dtype=torch.long, device=self.device)
                old_logprobs = torch.as_tensor(old_logprobs_np[idx], dtype=torch.float32, device=self.device)
                advantages = torch.as_tensor(advantages_np[idx], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(returns_np[idx], dtype=torch.float32, device=self.device)

                dist, values = self.policy(states, masks)
                logprobs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                ratios = torch.exp(logprobs - old_logprobs)
                approx_kl = (old_logprobs - logprobs).mean()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                approx_kls.append(float(approx_kl.item()))
                last_metrics = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                }
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(
                        "epoch={} loss={:.4f} kl={:.5f}".format(
                            epoch_idx + 1,
                            last_metrics["loss"],
                            last_metrics["approx_kl"],
                        ),
                        refresh=False,
                    )
            if self.target_kl is not None and approx_kls and float(np.mean(approx_kls)) > self.target_kl:
                break
        if progress is not None:
            progress.close()

        self.buffer.clear()
        return last_metrics

    def behavior_clone(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        *,
        epochs: int = 3,
        batch_size: int = 256,
    ) -> dict[str, float]:
        if len(actions) == 0:
            return {"bc_loss": 0.0, "bc_accuracy": 0.0}

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        n = actions_t.shape[0]
        last_loss = 0.0
        last_acc = 0.0
        for _ in range(epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                dist, _ = self.policy(states_t[idx], masks_t[idx])
                loss = -dist.log_prob(actions_t[idx]).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                with torch.no_grad():
                    pred = torch.argmax(dist.probs, dim=-1)
                    acc = (pred == actions_t[idx]).float().mean()
                last_loss = float(loss.item())
                last_acc = float(acc.item())
        return {"bc_loss": last_loss, "bc_accuracy": last_acc}

    def _gae_advantages_and_returns(self) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def _discounted_returns(self) -> list[float]:
        returns: list[float] = []
        discounted = 0.0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted = 0.0
            discounted = reward + self.gamma * discounted
            returns.insert(0, discounted)
        return returns
