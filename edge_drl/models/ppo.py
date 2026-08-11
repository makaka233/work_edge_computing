from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - only used when tqdm is unavailable.
    tqdm = None


def _center_advantages_by_group(
    advantages: np.ndarray,
    weights: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    """Remove each comparison group's weighted advantage mean.

    Slow Count actions for different service stages have very different base
    latency scales.  Centering within a stage makes the actor compare replica
    counts for the same stage instead of learning that intrinsically expensive
    stages are globally bad actions.
    """

    centered = np.asarray(advantages, dtype=np.float32).copy()
    weights_np = np.asarray(weights, dtype=np.float32)
    groups_np = np.asarray(group_ids)
    if centered.ndim != 1 or weights_np.shape != centered.shape or groups_np.shape != centered.shape:
        raise ValueError("advantages, weights, and group_ids must have the same 1D shape")
    for group_id in np.unique(groups_np):
        group_mask = groups_np == group_id
        group_weights = weights_np[group_mask]
        denominator = float(group_weights.sum())
        if denominator <= 1e-8:
            continue
        group_mean = float(np.sum(centered[group_mask] * group_weights) / denominator)
        centered[group_mask] -= group_mean
    return centered


def _normalize_advantages_by_group(
    advantages: np.ndarray,
    weights: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    """Normalize advantages independently so each operating regime has usable scale."""

    normalized = np.asarray(advantages, dtype=np.float32).copy()
    weights_np = np.asarray(weights, dtype=np.float32)
    groups_np = np.asarray(group_ids)
    if normalized.ndim != 1 or weights_np.shape != normalized.shape or groups_np.shape != normalized.shape:
        raise ValueError("advantages, weights, and group_ids must have the same 1D shape")
    for group_id in np.unique(groups_np):
        group_mask = groups_np == group_id
        group_weights = weights_np[group_mask]
        denominator = float(group_weights.sum())
        if denominator <= 1e-8:
            continue
        group_values = normalized[group_mask]
        group_mean = float(np.sum(group_values * group_weights) / denominator)
        group_variance = float(
            np.sum(np.square(group_values - group_mean) * group_weights) / denominator
        )
        normalized[group_mask] = (group_values - group_mean) / (np.sqrt(group_variance) + 1e-8)
    return normalized


def _balance_group_weights(
    weights: np.ndarray,
    group_ids: np.ndarray,
    target_group_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Assign each sampled regime its configured share of optimizer weight.

    Equal shares remain the compatibility default.  Supplying target weights
    lets stratified PPO retain coverage without changing the intended traffic
    distribution into a uniform objective.
    """

    balanced = np.asarray(weights, dtype=np.float32).copy()
    groups_np = np.asarray(group_ids)
    if balanced.ndim != 1 or groups_np.shape != balanced.shape:
        raise ValueError("weights and group_ids must have the same 1D shape")
    unique_groups = np.unique(groups_np)
    if unique_groups.size <= 1:
        return balanced
    total_weight = float(balanced.sum())
    if target_group_weights is None:
        present_targets = {group_id: 1.0 for group_id in unique_groups}
    else:
        targets = np.asarray(target_group_weights, dtype=np.float32)
        present_targets = {}
        for group_id in unique_groups:
            group_index = int(group_id)
            if not np.isclose(float(group_id), float(group_index)) or not 0 <= group_index < len(targets):
                raise ValueError("group IDs must be integer indices into target_group_weights")
            present_targets[group_id] = float(targets[group_index])
    present_target_sum = float(sum(present_targets.values()))
    if present_target_sum <= 1e-8:
        raise ValueError("present target group weights must have positive total weight")
    for group_id in unique_groups:
        group_mask = groups_np == group_id
        group_weight = float(balanced[group_mask].sum())
        if group_weight > 1e-8:
            target_weight = total_weight * present_targets[group_id] / present_target_sum
            balanced[group_mask] *= target_weight / group_weight
    return balanced


def _normalized_masked_entropy(
    entropy: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Scale categorical entropy by the maximum entropy of each action mask.

    A state with one feasible action is already maximally exploratory, so it is
    assigned one instead of spuriously pulling an adaptive entropy controller
    upward.
    """

    feasible = masks.bool().sum(dim=-1).to(dtype=entropy.dtype)
    denominator = torch.log(feasible.clamp_min(2.0))
    normalized = entropy / denominator
    return torch.where(feasible > 1.0, normalized.clamp(0.0, 1.0), torch.ones_like(normalized))


def _stratified_minibatches(
    sample_count: int,
    minibatch_size: int,
    group_ids: np.ndarray,
) -> list[np.ndarray]:
    """Shuffle within groups and spread every group across a complete epoch."""

    if sample_count < 1:
        return []
    groups_np = np.asarray(group_ids)
    if groups_np.shape != (sample_count,):
        raise ValueError("group_ids must contain one entry per sample")
    batch_count = int(np.ceil(sample_count / max(int(minibatch_size), 1)))
    unique_groups = np.unique(groups_np)
    group_chunks = {
        group_id: np.array_split(np.random.permutation(np.flatnonzero(groups_np == group_id)), batch_count)
        for group_id in unique_groups
    }
    batches: list[np.ndarray] = []
    for batch_idx in range(batch_count):
        parts = [group_chunks[group_id][batch_idx] for group_id in unique_groups]
        non_empty = [part for part in parts if len(part)]
        if not non_empty:
            continue
        batches.append(np.random.permutation(np.concatenate(non_empty)))
    return batches


@dataclass
class RolloutBuffer:
    states: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    sample_groups: list[float] = field(default_factory=list)

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
        weight: float = 1.0,
        sample_group: float | None = None,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.masks.append(np.asarray(mask, dtype=bool))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.weights.append(float(weight))
        if sample_group is not None:
            self.sample_groups.append(float(sample_group))

    def extend_rewards_for_pending(self, reward: float, done: bool, weight: float = 1.0) -> None:
        missing = len(self.actions) - len(self.rewards)
        for _ in range(missing):
            self.rewards.append(float(reward))
            self.dones.append(bool(done))
            self.weights.append(float(weight))

    def clear(self) -> None:
        self.states.clear()
        self.masks.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.weights.clear()
        self.sample_groups.clear()

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


class GraphAttentionActorCritic(nn.Module):
    """Graph actor-critic for either node placement or ordered count actions.

    Slow deployment has two action heads: placement scores every node, while
    replica count is represented by a discretized Gaussian over the ordered
    actions. Both heads consume the same topology-aware graph representation so
    they can see resource pressure and execution feedback over the whole edge
    network.
    """

    def __init__(
        self,
        *,
        global_dim: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        num_nodes: int,
        action_dim: int,
        action_mode: str,
        hidden_dim: int = 128,
        detach_critic_backbone: bool = False,
    ):
        super().__init__()
        if action_mode not in {"node", "count"}:
            raise ValueError("action_mode must be 'node' or 'count'")
        self.global_dim = global_dim
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_nodes = num_nodes
        self.action_dim = action_dim
        self.action_mode = action_mode
        self.detach_critic_backbone = detach_critic_backbone
        self.count_min_scale = max(float(action_dim) / 12.0, 1.0)
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
        if action_mode == "node":
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                # Predict an ordered distribution center and log scale instead
                # of one unrelated logit per possible replica count.
                nn.Linear(hidden_dim, 2),
            )
            with torch.no_grad():
                self.actor[-1].bias[0] = 0.0
                # Start broad enough to explore nearby replica counts without
                # making the center gradient vanish across the full action
                # range.  The previous action_dim / 3 scale kept 32-count
                # policies close to uniform and diluted the ordinal signal.
                self.actor[-1].bias[1] = math.log(max(float(action_dim) / 6.0, 1.0))
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
        pooled_nodes = graph_emb.mean(dim=1)
        graph_context = torch.cat([global_emb, pooled_nodes], dim=-1)

        if self.action_mode == "node":
            logits = self.actor(torch.cat(
                [global_emb.unsqueeze(1).expand(-1, self.num_nodes, -1), graph_emb],
                dim=-1,
            )).squeeze(-1)
        else:
            count_parameters = self.actor(graph_context)
            center = torch.sigmoid(count_parameters[:, 0]) * max(float(self.action_dim - 1), 0.0)
            log_scale = torch.clamp(
                count_parameters[:, 1],
                min=math.log(self.count_min_scale),
                max=math.log(max(float(self.action_dim), 1.0)),
            )
            scale = torch.exp(log_scale)
            count_indices = torch.arange(
                self.action_dim,
                dtype=graph_context.dtype,
                device=graph_context.device,
            ).unsqueeze(0)
            logits = -0.5 * torch.square((count_indices - center.unsqueeze(1)) / scale.unsqueeze(1))

        safe_masks = masks.bool()
        fallback = torch.zeros_like(safe_masks)
        fallback[:, 0] = True
        safe_masks = torch.where(safe_masks.any(dim=1, keepdim=True), safe_masks, fallback)
        logits = logits.masked_fill(~safe_masks, -1e9)
        dist = Categorical(logits=logits)
        critic_context = graph_context.detach() if self.detach_critic_backbone else graph_context
        values = self.critic(critic_context).squeeze(-1)
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
        detach_critic_backbone: bool = False,
        group_balanced_updates: bool = False,
        group_weight_targets: tuple[float, ...] | None = None,
        full_batch_kl_stop: bool = False,
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
        self.group_balanced_updates = bool(group_balanced_updates)
        self.group_weight_targets = (
            None
            if group_weight_targets is None
            else np.asarray(group_weight_targets, dtype=np.float32)
        )
        if self.group_weight_targets is not None:
            if self.group_weight_targets.ndim != 1 or len(self.group_weight_targets) == 0:
                raise ValueError("group_weight_targets must be a non-empty 1D sequence")
            if np.any(self.group_weight_targets < 0.0) or float(self.group_weight_targets.sum()) <= 0.0:
                raise ValueError("group_weight_targets must be non-negative with positive total weight")
            self.group_weight_targets /= float(self.group_weight_targets.sum())
        self.full_batch_kl_stop = bool(full_batch_kl_stop)
        self.device = torch.device(device)
        self.last_group_diagnostics: dict[float, dict[str, float]] = {}
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
        elif policy_kind in {"slow_gat_node", "slow_gat_count"}:
            if global_dim is None or node_feature_dim is None or edge_feature_dim is None or num_nodes is None:
                raise ValueError(f"{policy_kind} requires global_dim, node_feature_dim, edge_feature_dim, and num_nodes")
            self.policy = GraphAttentionActorCritic(
                global_dim=global_dim,
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                num_nodes=num_nodes,
                action_dim=action_dim,
                action_mode="node" if policy_kind == "slow_gat_node" else "count",
                hidden_dim=hidden_dim,
                detach_critic_backbone=detach_critic_backbone,
            ).to(self.device)
        else:
            raise ValueError(f"unknown policy_kind: {policy_kind}")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def set_learning_rate(self, learning_rate: float) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def _clip_policy_gradients(self, max_norm: float = 0.5) -> None:
        """Clip detached actor/backbone and critic gradients independently."""

        if not bool(getattr(self.policy, "detach_critic_backbone", False)):
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm)
            return
        critic_parameters = list(self.policy.critic.parameters())
        critic_parameter_ids = {id(parameter) for parameter in critic_parameters}
        actor_parameters = [
            parameter
            for parameter in self.policy.parameters()
            if id(parameter) not in critic_parameter_ids
        ]
        nn.utils.clip_grad_norm_(actor_parameters, max_norm)
        nn.utils.clip_grad_norm_(critic_parameters, max_norm)

    def act(self, state: np.ndarray, mask: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state_t, mask_t)
            action_t = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
            logprob_t = dist.log_prob(action_t)
        return int(action_t.item()), float(logprob_t.item()), float(value.item())

    def act_batch(
        self,
        states: list[np.ndarray] | np.ndarray,
        masks: list[np.ndarray] | np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of masked actions with one policy forward pass."""

        state_array = np.asarray(states, dtype=np.float32)
        mask_array = np.asarray(masks, dtype=bool)
        if state_array.ndim != 2 or mask_array.ndim != 2 or state_array.shape[0] == 0:
            raise ValueError("act_batch expects non-empty 2D states and masks")
        if state_array.shape[0] != mask_array.shape[0]:
            raise ValueError("states and masks must have the same batch size")
        state_t = torch.as_tensor(state_array, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask_array, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            dist, values = self.policy(state_t, mask_t)
            actions = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
            logprobs = dist.log_prob(actions)
        return (
            actions.detach().cpu().numpy().astype(np.int64),
            logprobs.detach().cpu().numpy().astype(np.float32),
            values.detach().cpu().numpy().astype(np.float32),
        )

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

    def action_probabilities(self, state: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state_t, mask_t)
            probs = dist.probs.squeeze(0).detach().cpu().numpy()
        return probs, float(value.item())

    def _buffer_policy_diagnostics(
        self,
        actions_np: np.ndarray,
        old_logprobs_np: np.ndarray,
        weights_np: np.ndarray,
        group_ids_np: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Measure the final policy against the complete behavior-policy batch."""

        weighted_entropy = 0.0
        weighted_normalized_entropy = 0.0
        weighted_kl = 0.0
        weighted_clip = 0.0
        total_weight = 0.0
        group_sums: dict[float, dict[str, float]] = {}
        batch_size = max(1, min(self.minibatch_size, len(actions_np)))
        with torch.no_grad():
            for start in range(0, len(actions_np), batch_size):
                end = start + batch_size
                states = torch.as_tensor(
                    np.stack(self.buffer.states[start:end]), dtype=torch.float32, device=self.device
                )
                masks = torch.as_tensor(
                    np.stack(self.buffer.masks[start:end]), dtype=torch.bool, device=self.device
                )
                actions = torch.as_tensor(actions_np[start:end], dtype=torch.long, device=self.device)
                old_logprobs = torch.as_tensor(
                    old_logprobs_np[start:end], dtype=torch.float32, device=self.device
                )
                batch_weights = torch.as_tensor(
                    weights_np[start:end], dtype=torch.float32, device=self.device
                )
                dist, _ = self.policy(states, masks)
                logprobs = dist.log_prob(actions)
                entropy_values = dist.entropy()
                normalized_entropy_values = _normalized_masked_entropy(entropy_values, masks)
                log_ratios = logprobs - old_logprobs
                ratios = torch.exp(log_ratios)
                kl_values = (ratios - 1.0) - log_ratios
                clip_values = (torch.abs(ratios - 1.0) > self.eps_clip).float()
                weighted_entropy += float((entropy_values * batch_weights).sum().item())
                weighted_normalized_entropy += float(
                    (normalized_entropy_values * batch_weights).sum().item()
                )
                weighted_kl += float((kl_values * batch_weights).sum().item())
                weighted_clip += float((clip_values * batch_weights).sum().item())
                total_weight += float(batch_weights.sum().item())
                if group_ids_np is not None:
                    batch_groups = group_ids_np[start:end]
                    for group_id in np.unique(batch_groups):
                        group_mask = torch.as_tensor(
                            batch_groups == group_id,
                            dtype=torch.bool,
                            device=self.device,
                        )
                        group_weight = float(batch_weights[group_mask].sum().item())
                        sums = group_sums.setdefault(
                            float(group_id),
                            {
                                "weight": 0.0,
                                "kl": 0.0,
                                "clip": 0.0,
                                "entropy": 0.0,
                                "normalized_entropy": 0.0,
                            },
                        )
                        sums["weight"] += group_weight
                        sums["kl"] += float(
                            (kl_values[group_mask] * batch_weights[group_mask]).sum().item()
                        )
                        sums["clip"] += float(
                            (clip_values[group_mask] * batch_weights[group_mask]).sum().item()
                        )
                        sums["entropy"] += float(
                            (entropy_values[group_mask] * batch_weights[group_mask]).sum().item()
                        )
                        sums["normalized_entropy"] += float(
                            (
                                normalized_entropy_values[group_mask]
                                * batch_weights[group_mask]
                            ).sum().item()
                        )
        denominator = max(total_weight, 1e-8)
        self.last_group_diagnostics = {
            group_id: {
                "approx_kl": sums["kl"] / max(sums["weight"], 1e-8),
                "clip_fraction": sums["clip"] / max(sums["weight"], 1e-8),
                "entropy": sums["entropy"] / max(sums["weight"], 1e-8),
                "normalized_entropy": sums["normalized_entropy"] / max(sums["weight"], 1e-8),
            }
            for group_id, sums in sorted(group_sums.items())
        }
        group_kls = [metrics["approx_kl"] for metrics in self.last_group_diagnostics.values()]
        return {
            "entropy": weighted_entropy / denominator,
            "normalized_entropy": weighted_normalized_entropy / denominator,
            "approx_kl": weighted_kl / denominator,
            "clip_fraction": weighted_clip / denominator,
            "group_count": float(len(group_kls)),
            "max_group_approx_kl": float(max(group_kls)) if group_kls else 0.0,
            "min_group_approx_kl": float(min(group_kls)) if group_kls else 0.0,
            "group_approx_kl_std": float(np.std(group_kls)) if group_kls else 0.0,
        }

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        if len(self.buffer) == 0:
            self.last_group_diagnostics = {}
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "epochs_completed": 0.0,
                "kl_early_stop": 0.0,
                "optimizer_steps": 0.0,
                "minibatches_completed": 0.0,
                "minibatches_planned": 0.0,
                "samples_seen_fraction": 0.0,
                "min_group_seen_fraction": 0.0,
                "full_batch_kl_checks": 0.0,
                "group_count": 0.0,
                "max_group_approx_kl": 0.0,
                "min_group_approx_kl": 0.0,
                "group_approx_kl_std": 0.0,
            }
        if len(self.buffer.rewards) != len(self.buffer.actions):
            raise ValueError("buffer contains pending transitions without rewards")

        advantages_np, returns_np = self._gae_advantages_and_returns()
        if self.buffer.weights:
            if len(self.buffer.weights) != len(self.buffer.actions):
                raise ValueError("buffer sample weights do not match actions")
            weights_np = np.asarray(self.buffer.weights, dtype=np.float32)
        else:
            weights_np = np.ones(len(self.buffer.actions), dtype=np.float32)
        if self.buffer.sample_groups:
            if len(self.buffer.sample_groups) != len(self.buffer.actions):
                raise ValueError("buffer sample groups do not match actions")
            group_ids_np = np.asarray(self.buffer.sample_groups, dtype=np.float32)
        else:
            group_ids_np = np.zeros(len(self.buffer.actions), dtype=np.float32)
        if self.group_balanced_updates:
            weights_np = _balance_group_weights(
                weights_np,
                group_ids_np,
                self.group_weight_targets,
            )
        weights_np /= max(float(weights_np.mean()), 1e-8)
        advantage_mean = float(np.average(advantages_np, weights=weights_np))
        advantage_variance = float(np.average((advantages_np - advantage_mean) ** 2, weights=weights_np))
        advantage_std = float(np.sqrt(advantage_variance))
        if self.group_balanced_updates and np.unique(group_ids_np).size > 1:
            advantages_np = _normalize_advantages_by_group(advantages_np, weights_np, group_ids_np)
        else:
            advantages_np = (advantages_np - advantage_mean) / (advantage_std + 1e-8)

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

        last_metrics: dict[str, float] = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
        }
        optimizer_steps = 0
        epochs_completed = 0
        kl_early_stop = False
        minibatches_completed = 0
        full_batch_kl_checks = 0
        samples_seen = np.zeros(n, dtype=bool)
        final_diagnostics: dict[str, float] | None = None
        for epoch_idx in range(self.k_epochs):
            if self.group_balanced_updates and np.unique(group_ids_np).size > 1:
                epoch_batches = _stratified_minibatches(n, minibatch_size, group_ids_np)
            else:
                order = np.random.permutation(n)
                epoch_batches = [order[start : start + minibatch_size] for start in range(0, n, minibatch_size)]
            approx_kls: list[float] = []
            epoch_complete = True
            for idx in epoch_batches:
                states = torch.as_tensor(np.stack([self.buffer.states[i] for i in idx]), dtype=torch.float32, device=self.device)
                masks = torch.as_tensor(np.stack([self.buffer.masks[i] for i in idx]), dtype=torch.bool, device=self.device)
                actions = torch.as_tensor(actions_np[idx], dtype=torch.long, device=self.device)
                old_logprobs = torch.as_tensor(old_logprobs_np[idx], dtype=torch.float32, device=self.device)
                advantages = torch.as_tensor(advantages_np[idx], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(returns_np[idx], dtype=torch.float32, device=self.device)
                sample_weights = torch.as_tensor(weights_np[idx], dtype=torch.float32, device=self.device)
                sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)

                dist, values = self.policy(states, masks)
                logprobs = dist.log_prob(actions)
                entropy_values = dist.entropy()
                normalized_entropy_values = _normalized_masked_entropy(entropy_values, masks)
                entropy = (entropy_values * sample_weights).mean()
                normalized_entropy = (normalized_entropy_values * sample_weights).mean()
                log_ratios = logprobs - old_logprobs
                ratios = torch.exp(log_ratios)
                approx_kl = (((ratios - 1.0) - log_ratios) * sample_weights).mean()
                clip_fraction = (
                    (torch.abs(ratios - 1.0) > self.eps_clip).float() * sample_weights
                ).mean()

                if (
                    not self.full_batch_kl_stop
                    and self.target_kl is not None
                    and optimizer_steps > 0
                    and float(approx_kl.item()) > self.target_kl
                ):
                    kl_early_stop = True
                    epoch_complete = False
                    break

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                policy_loss = -(torch.min(surr1, surr2) * sample_weights).mean()
                value_loss = (((values - returns) ** 2) * sample_weights).mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * normalized_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                self._clip_policy_gradients()
                self.optimizer.step()
                optimizer_steps += 1
                minibatches_completed += 1
                samples_seen[idx] = True
                approx_kls.append(float(approx_kl.item()))
                last_metrics = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "normalized_entropy": float(normalized_entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                    "clip_fraction": float(clip_fraction.item()),
                    "advantage_mean": advantage_mean,
                    "advantage_std": advantage_std,
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
            if epoch_complete:
                epochs_completed += 1
            if kl_early_stop:
                break
            if self.full_batch_kl_stop:
                final_diagnostics = self._buffer_policy_diagnostics(
                    actions_np,
                    old_logprobs_np,
                    weights_np,
                    group_ids_np,
                )
                full_batch_kl_checks += 1
                kl_for_stop = max(
                    final_diagnostics["approx_kl"],
                    final_diagnostics["max_group_approx_kl"],
                )
                if self.target_kl is not None and kl_for_stop > self.target_kl:
                    kl_early_stop = True
                    break
            elif self.target_kl is not None and approx_kls and float(np.mean(approx_kls)) > self.target_kl:
                kl_early_stop = True
                break
        if progress is not None:
            progress.close()

        if final_diagnostics is None:
            final_diagnostics = self._buffer_policy_diagnostics(
                actions_np,
                old_logprobs_np,
                weights_np,
                group_ids_np,
            )
        last_metrics.update(final_diagnostics)
        group_coverages = [
            float(np.mean(samples_seen[group_ids_np == group_id]))
            for group_id in np.unique(group_ids_np)
        ]
        last_metrics["epochs_completed"] = float(epochs_completed)
        last_metrics["kl_early_stop"] = float(kl_early_stop)
        last_metrics["optimizer_steps"] = float(optimizer_steps)
        last_metrics["minibatches_completed"] = float(minibatches_completed)
        last_metrics["minibatches_planned"] = float(self.k_epochs * batches_per_epoch)
        last_metrics["samples_seen_fraction"] = float(np.mean(samples_seen))
        last_metrics["min_group_seen_fraction"] = float(min(group_coverages)) if group_coverages else 0.0
        last_metrics["full_batch_kl_checks"] = float(full_batch_kl_checks)
        self.buffer.clear()
        return last_metrics

    def update_actor(
        self,
        advantages: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        progress_label: str = "",
        progress_interval_seconds: float = 0.0,
    ) -> dict[str, float]:
        """Update only the actor using externally computed window advantages.

        Slow deployment is a composite action whose count and placement choices
        share one window-level return. Its actors therefore must not run GAE over
        those component choices as if they were consecutive environment steps.
        """

        n = len(self.buffer)
        if n == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
            }
        advantages_np = np.asarray(advantages, dtype=np.float32)
        if advantages_np.shape != (n,):
            raise ValueError(f"advantages must have shape ({n},), got {advantages_np.shape}")
        if sample_weights is None:
            weights_np = np.ones(n, dtype=np.float32)
        else:
            weights_np = np.asarray(sample_weights, dtype=np.float32)
            if weights_np.shape != (n,):
                raise ValueError(f"sample_weights must have shape ({n},), got {weights_np.shape}")
            if np.any(weights_np < 0.0) or not np.isfinite(weights_np).all():
                raise ValueError("sample_weights must be finite and non-negative")
            weights_np = weights_np / max(float(weights_np.mean()), 1e-8)

        actions_np = np.asarray(self.buffer.actions, dtype=np.int64)
        old_logprobs_np = np.asarray(self.buffer.logprobs, dtype=np.float32)
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
                states = torch.as_tensor(
                    np.stack([self.buffer.states[i] for i in idx]),
                    dtype=torch.float32,
                    device=self.device,
                )
                masks = torch.as_tensor(
                    np.stack([self.buffer.masks[i] for i in idx]),
                    dtype=torch.bool,
                    device=self.device,
                )
                actions = torch.as_tensor(actions_np[idx], dtype=torch.long, device=self.device)
                old_logprobs = torch.as_tensor(old_logprobs_np[idx], dtype=torch.float32, device=self.device)
                batch_advantages = torch.as_tensor(advantages_np[idx], dtype=torch.float32, device=self.device)
                batch_weights = torch.as_tensor(weights_np[idx], dtype=torch.float32, device=self.device)

                dist, _ = self.policy(states, masks)
                logprobs = dist.log_prob(actions)
                entropies = dist.entropy()
                normalized_entropies = _normalized_masked_entropy(entropies, masks)
                ratios = torch.exp(logprobs - old_logprobs)
                approx_kl = (old_logprobs - logprobs).mean()
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages
                denominator = batch_weights.sum().clamp_min(1e-8)
                policy_loss = -(torch.min(surr1, surr2) * batch_weights).sum() / denominator
                entropy = (entropies * batch_weights).sum() / denominator
                normalized_entropy = (
                    normalized_entropies * batch_weights
                ).sum() / denominator
                loss = policy_loss - self.entropy_coef * normalized_entropy

                self.optimizer.zero_grad()
                loss.backward()
                self._clip_policy_gradients()
                self.optimizer.step()
                approx_kls.append(float(approx_kl.item()))
                last_metrics = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": 0.0,
                    "entropy": float(entropy.item()),
                    "normalized_entropy": float(normalized_entropy.item()),
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

    def update_from_returns(
        self,
        returns: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        advantage_group_ids: np.ndarray | None = None,
        actor_use_value_baseline: bool = True,
        auxiliary_advantages: np.ndarray | None = None,
        auxiliary_advantage_coef: float = 0.0,
        progress_label: str = "",
        progress_interval_seconds: float = 0.0,
    ) -> dict[str, float]:
        """Run PPO from externally supplied Monte-Carlo returns.

        Composite slow-control actions are not consecutive environment steps,
        so GAE over their component choices is invalid.  Each component still
        has its own pre-action state and value prediction, however.  Training
        that value head against the shared window return provides a much more
        useful state-dependent baseline than assigning one window advantage to
        every count and placement decision.
        """

        n = len(self.buffer)
        if n == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "approx_kl": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "auxiliary_advantage_mean": 0.0,
                "auxiliary_advantage_std": 0.0,
                "auxiliary_advantage_coef": float(auxiliary_advantage_coef),
                "combined_advantage_std": 0.0,
                "explained_variance": 0.0,
                "post_explained_variance": 0.0,
            }
        if not np.isfinite(auxiliary_advantage_coef) or auxiliary_advantage_coef < 0.0:
            raise ValueError("auxiliary_advantage_coef must be finite and non-negative")
        returns_np = np.asarray(returns, dtype=np.float32)
        if returns_np.shape != (n,):
            raise ValueError(f"returns must have shape ({n},), got {returns_np.shape}")
        old_values_np = np.asarray(self.buffer.values, dtype=np.float32)
        if old_values_np.shape != (n,):
            raise ValueError("buffer value predictions do not match actions")
        if sample_weights is None:
            weights_np = np.ones(n, dtype=np.float32)
        else:
            weights_np = np.asarray(sample_weights, dtype=np.float32)
            if weights_np.shape != (n,):
                raise ValueError(f"sample_weights must have shape ({n},), got {weights_np.shape}")
            if np.any(weights_np < 0.0) or not np.isfinite(weights_np).all():
                raise ValueError("sample_weights must be finite and non-negative")
        weights_np = weights_np / max(float(weights_np.mean()), 1e-8)

        # A poor component critic can make an otherwise useful Monte-Carlo
        # signal noisier. Count can therefore train its actor from direct
        # returns centered within comparable service stages, while the value
        # head still learns the original absolute returns below.
        advantages_np = returns_np - old_values_np if actor_use_value_baseline else returns_np.copy()
        if advantage_group_ids is not None:
            group_ids_np = np.asarray(advantage_group_ids)
            if group_ids_np.shape != (n,):
                raise ValueError(f"advantage_group_ids must have shape ({n},), got {group_ids_np.shape}")
            advantages_np = _center_advantages_by_group(
                advantages_np,
                weights_np,
                group_ids_np,
            )
        advantage_mean = float(np.average(advantages_np, weights=weights_np))
        advantage_variance = float(np.average((advantages_np - advantage_mean) ** 2, weights=weights_np))
        advantage_std = float(np.sqrt(advantage_variance))
        if advantage_std > 1e-8:
            advantages_np = (advantages_np - advantage_mean) / (advantage_std + 1e-8)
        else:
            advantages_np = advantages_np - advantage_mean

        auxiliary_advantage_mean = 0.0
        auxiliary_advantage_std = 0.0
        if auxiliary_advantages is not None:
            auxiliary_np = np.asarray(auxiliary_advantages, dtype=np.float32)
            if auxiliary_np.shape != (n,):
                raise ValueError(f"auxiliary_advantages must have shape ({n},), got {auxiliary_np.shape}")
            if not np.isfinite(auxiliary_np).all():
                raise ValueError("auxiliary_advantages must be finite")
            auxiliary_advantage_mean = float(np.average(auxiliary_np, weights=weights_np))
            auxiliary_variance = float(
                np.average(
                    (auxiliary_np - auxiliary_advantage_mean) ** 2,
                    weights=weights_np,
                )
            )
            auxiliary_advantage_std = float(np.sqrt(auxiliary_variance))
            if auxiliary_advantage_std > 1e-8:
                auxiliary_np = (
                    auxiliary_np - auxiliary_advantage_mean
                ) / (auxiliary_advantage_std + 1e-8)
            else:
                auxiliary_np = auxiliary_np - auxiliary_advantage_mean
            advantages_np = advantages_np + float(auxiliary_advantage_coef) * auxiliary_np

        # Keep the overall PPO gradient scale stable after mixing local credit
        # with the Slow window-level residual. The coefficient still controls
        # their relative contribution before this final normalization.
        combined_advantage_mean = float(np.average(advantages_np, weights=weights_np))
        combined_advantage_variance = float(
            np.average(
                (advantages_np - combined_advantage_mean) ** 2,
                weights=weights_np,
            )
        )
        combined_advantage_std = float(np.sqrt(combined_advantage_variance))
        if combined_advantage_std > 1e-8:
            advantages_np = (
                advantages_np - combined_advantage_mean
            ) / (combined_advantage_std + 1e-8)
        else:
            advantages_np = advantages_np - combined_advantage_mean

        return_variance = float(np.average((returns_np - np.average(returns_np, weights=weights_np)) ** 2, weights=weights_np))
        residual_variance = float(
            np.average(
                ((returns_np - old_values_np) - np.average(returns_np - old_values_np, weights=weights_np)) ** 2,
                weights=weights_np,
            )
        )
        explained_variance = 0.0 if return_variance <= 1e-8 else 1.0 - residual_variance / return_variance

        actions_np = np.asarray(self.buffer.actions, dtype=np.int64)
        old_logprobs_np = np.asarray(self.buffer.logprobs, dtype=np.float32)
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

        last_metrics: dict[str, float] = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
            "auxiliary_advantage_mean": auxiliary_advantage_mean,
            "auxiliary_advantage_std": auxiliary_advantage_std,
            "auxiliary_advantage_coef": float(auxiliary_advantage_coef),
            "combined_advantage_std": combined_advantage_std,
        }
        optimizer_steps = 0
        epochs_completed = 0
        kl_early_stop = False
        for epoch_idx in range(self.k_epochs):
            order = np.random.permutation(n)
            approx_kls: list[float] = []
            for start in range(0, n, minibatch_size):
                idx = order[start : start + minibatch_size]
                states = torch.as_tensor(
                    np.stack([self.buffer.states[i] for i in idx]),
                    dtype=torch.float32,
                    device=self.device,
                )
                masks = torch.as_tensor(
                    np.stack([self.buffer.masks[i] for i in idx]),
                    dtype=torch.bool,
                    device=self.device,
                )
                actions = torch.as_tensor(actions_np[idx], dtype=torch.long, device=self.device)
                old_logprobs = torch.as_tensor(old_logprobs_np[idx], dtype=torch.float32, device=self.device)
                advantages = torch.as_tensor(advantages_np[idx], dtype=torch.float32, device=self.device)
                targets = torch.as_tensor(returns_np[idx], dtype=torch.float32, device=self.device)
                batch_weights = torch.as_tensor(weights_np[idx], dtype=torch.float32, device=self.device)
                denominator = batch_weights.sum().clamp_min(1e-8)

                dist, values = self.policy(states, masks)
                logprobs = dist.log_prob(actions)
                entropies = dist.entropy()
                normalized_entropies = _normalized_masked_entropy(entropies, masks)
                log_ratios = logprobs - old_logprobs
                ratios = torch.exp(log_ratios)
                approx_kl = (((ratios - 1.0) - log_ratios) * batch_weights).sum() / denominator
                clip_fraction = (
                    (torch.abs(ratios - 1.0) > self.eps_clip).float() * batch_weights
                ).sum() / denominator
                if self.target_kl is not None and optimizer_steps > 0 and float(approx_kl.item()) > self.target_kl:
                    kl_early_stop = True
                    break
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                policy_loss = -(torch.min(surr1, surr2) * batch_weights).sum() / denominator
                value_loss = (((values - targets) ** 2) * batch_weights).sum() / denominator
                entropy = (entropies * batch_weights).sum() / denominator
                normalized_entropy = (
                    normalized_entropies * batch_weights
                ).sum() / denominator
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * normalized_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                self._clip_policy_gradients()
                self.optimizer.step()
                optimizer_steps += 1
                approx_kls.append(float(approx_kl.item()))
                last_metrics = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "normalized_entropy": float(normalized_entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                    "clip_fraction": float(clip_fraction.item()),
                    "advantage_mean": advantage_mean,
                    "advantage_std": advantage_std,
                    "auxiliary_advantage_mean": auxiliary_advantage_mean,
                    "auxiliary_advantage_std": auxiliary_advantage_std,
                    "auxiliary_advantage_coef": float(auxiliary_advantage_coef),
                    "combined_advantage_std": combined_advantage_std,
                    "explained_variance": explained_variance,
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
            epochs_completed = epoch_idx + 1
            if kl_early_stop:
                break
            if self.target_kl is not None and approx_kls and float(np.mean(approx_kls)) > self.target_kl:
                kl_early_stop = True
                break
        if progress is not None:
            progress.close()
        post_values_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, minibatch_size):
                states = torch.as_tensor(
                    np.stack(self.buffer.states[start : start + minibatch_size]),
                    dtype=torch.float32,
                    device=self.device,
                )
                masks = torch.as_tensor(
                    np.stack(self.buffer.masks[start : start + minibatch_size]),
                    dtype=torch.bool,
                    device=self.device,
                )
                _, values = self.policy(states, masks)
                post_values_parts.append(values.detach().cpu().numpy())
        post_values_np = np.concatenate(post_values_parts).astype(np.float32, copy=False)
        post_residual = returns_np - post_values_np
        post_residual_variance = float(
            np.average(
                (post_residual - np.average(post_residual, weights=weights_np)) ** 2,
                weights=weights_np,
            )
        )
        last_metrics["post_explained_variance"] = (
            0.0 if return_variance <= 1e-8 else 1.0 - post_residual_variance / return_variance
        )
        last_metrics.update(self._buffer_policy_diagnostics(actions_np, old_logprobs_np, weights_np))
        last_metrics["epochs_completed"] = float(epochs_completed)
        last_metrics["kl_early_stop"] = float(kl_early_stop)
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
                self._clip_policy_gradients()
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
