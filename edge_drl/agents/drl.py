from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest
from edge_drl.models.ppo import PPOAgent


def slow_obs_dim(num_nodes: int, num_service_types: int) -> int:
    return 8 + num_nodes * 5 + num_nodes * num_service_types


def fast_obs_dim(num_nodes: int, policy_kind: str = "gat_node_scorer") -> int:
    dim = FAST_GLOBAL_DIM + num_nodes * FAST_NODE_FEATURE_DIM
    if policy_kind == "gat_node_scorer":
        dim += num_nodes * num_nodes * FAST_EDGE_FEATURE_DIM
    return dim


FAST_GLOBAL_DIM = 12
FAST_NODE_FEATURE_DIM = 6
FAST_EDGE_FEATURE_DIM = 3


class SlowWindowCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


@dataclass
class SlowDeploymentPPOAgent:
    num_nodes: int
    num_service_types: int
    max_service_stages: int
    replicas_per_stage: int = 5
    coverage_repair: bool = True
    lr: float = 3e-4
    k_epochs: int = 3
    entropy_coef: float = 0.001
    count_entropy_coef: float | None = None
    placement_entropy_coef: float | None = None
    value_coef: float = 0.5
    critic_lr: float | None = None
    critic_k_epochs: int = 4
    target_kl: float | None = 0.03
    minibatch_size: int = 2048
    device: str = "cpu"
    count_ppo: PPOAgent = field(init=False)
    placement_ppo: PPOAgent = field(init=False)
    ppo: PPOAgent = field(init=False)
    window_critic: SlowWindowCritic = field(init=False)
    critic_optimizer: torch.optim.Optimizer = field(init=False)
    pending_count_indices: list[int] = field(default_factory=list)
    pending_placement_indices: list[int] = field(default_factory=list)
    count_window_ids: list[int] = field(default_factory=list)
    placement_window_ids: list[int] = field(default_factory=list)
    window_states: list[np.ndarray] = field(default_factory=list)
    window_old_values: list[float] = field(default_factory=list)
    window_returns: list[float] = field(default_factory=list)
    pending_window_id: int | None = None

    def __post_init__(self) -> None:
        self.count_ppo = PPOAgent(
            obs_dim=slow_obs_dim(self.num_nodes, self.num_service_types),
            action_dim=self.replicas_per_stage,
            hidden_dim=128,
            lr=self.lr,
            gamma=0.99,
            k_epochs=self.k_epochs,
            entropy_coef=self.entropy_coef if self.count_entropy_coef is None else self.count_entropy_coef,
            value_coef=self.value_coef,
            target_kl=self.target_kl,
            minibatch_size=self.minibatch_size,
            device=self.device,
        )
        self.placement_ppo = PPOAgent(
            obs_dim=slow_obs_dim(self.num_nodes, self.num_service_types),
            action_dim=self.num_nodes,
            hidden_dim=128,
            lr=self.lr,
            gamma=0.99,
            k_epochs=self.k_epochs,
            entropy_coef=self.entropy_coef if self.placement_entropy_coef is None else self.placement_entropy_coef,
            value_coef=self.value_coef,
            target_kl=self.target_kl,
            minibatch_size=self.minibatch_size,
            device=self.device,
        )
        self.ppo = self.placement_ppo
        critic_device = torch.device(self.device)
        self.window_critic = SlowWindowCritic(
            slow_obs_dim(self.num_nodes, self.num_service_types),
            hidden_dim=128,
        ).to(critic_device)
        self.critic_optimizer = torch.optim.Adam(
            self.window_critic.parameters(),
            lr=self.lr if self.critic_lr is None else self.critic_lr,
        )

    def plan_deployment(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> np.ndarray:
        env._require_ready()
        assert env.scenario is not None
        current = env.deployment if env.deployment is not None else np.zeros(
            (self.num_service_types, self.max_service_stages, self.num_nodes), dtype=bool
        )
        deployment = np.zeros_like(current, dtype=bool)
        remaining_memory = env.service_memory_capacities()
        remaining_storage = env.service_storage_capacities()
        demand = self._node_service_demand(env)
        self.pending_count_indices.clear()
        self.pending_placement_indices.clear()
        window_id: int | None = None
        if record:
            if self.pending_window_id is not None:
                raise RuntimeError("previous slow deployment window has no completed return")
            window_state = self._build_window_state(env, demand, remaining_memory, remaining_storage)
            state_t = torch.as_tensor(window_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                old_value = float(self.window_critic(state_t).item())
            window_id = len(self.window_states)
            self.window_states.append(window_state)
            self.window_old_values.append(old_value)
            self.pending_window_id = window_id

        stage_sequence = [
            (service, stage)
            for service in env.scenario.services
            for stage in service.stages
        ]
        for stage_index, (service, stage) in enumerate(stage_sequence):
            future_stages = [future_stage for _, future_stage in stage_sequence[stage_index + 1 :]]
            reserved_memory = float(sum(future_stage.memory_gb for future_stage in future_stages))
            reserved_storage = float(sum(future_stage.storage_gb for future_stage in future_stages))
            count_state = self._build_state(
                env,
                demand,
                remaining_memory,
                remaining_storage,
                service.service_id,
                stage.stage_id,
                0,
            )
            count_mask = self._build_count_mask(
                stage,
                remaining_memory,
                remaining_storage,
                reserved_memory=reserved_memory,
                reserved_storage=reserved_storage,
            )
            if not count_mask.any():
                continue
            count_action, count_logprob, count_value = self.count_ppo.act(
                count_state,
                count_mask,
                deterministic=deterministic,
            )
            if not count_mask[count_action]:
                count_action = int(np.where(count_mask)[0][0])
            target_replicas = count_action + 1
            if record:
                self.count_ppo.buffer.states.append(count_state)
                self.count_ppo.buffer.masks.append(count_mask.astype(bool))
                self.count_ppo.buffer.actions.append(count_action)
                self.count_ppo.buffer.logprobs.append(count_logprob)
                self.count_ppo.buffer.values.append(count_value)
                self.pending_count_indices.append(len(self.count_ppo.buffer.actions) - 1)
                assert window_id is not None
                self.count_window_ids.append(window_id)

            for replica_idx in range(target_replicas):
                state = self._build_state(
                    env,
                    demand,
                    remaining_memory,
                    remaining_storage,
                    service.service_id,
                    stage.stage_id,
                    replica_idx,
                )
                already = deployment[service.service_id, stage.stage_id]
                feasible_new = (
                    (remaining_memory >= stage.memory_gb)
                    & (remaining_storage >= stage.storage_gb)
                    & ~already
                )
                mask = feasible_new
                if not mask.any():
                    break

                action, logprob, value = self.placement_ppo.act(state, mask, deterministic=deterministic)
                if not mask[action]:
                    action = int(np.where(mask)[0][0])

                if record:
                    self.placement_ppo.buffer.states.append(state)
                    self.placement_ppo.buffer.masks.append(mask.astype(bool))
                    self.placement_ppo.buffer.actions.append(action)
                    self.placement_ppo.buffer.logprobs.append(logprob)
                    self.placement_ppo.buffer.values.append(value)
                    self.pending_placement_indices.append(len(self.placement_ppo.buffer.actions) - 1)
                    assert window_id is not None
                    self.placement_window_ids.append(window_id)

                deployment[service.service_id, stage.stage_id, action] = True
                remaining_memory[action] -= stage.memory_gb
                remaining_storage[action] -= stage.storage_gb

        if self.coverage_repair:
            self._coverage_repair(env, deployment, remaining_memory, remaining_storage)

        feasible, reason = env.check_deployment_feasible(deployment)
        if not feasible:
            deployment = self._repair_with_current_or_full(env, deployment)
        return deployment

    def assign_pending_reward(self, reward: float, done: bool) -> None:
        del done
        if self.pending_window_id is None:
            return
        if self.pending_window_id != len(self.window_returns):
            raise RuntimeError("slow deployment windows must be completed in collection order")
        self.window_returns.append(float(reward))
        self.pending_window_id = None
        self.pending_count_indices.clear()
        self.pending_placement_indices.clear()

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        if self.pending_window_id is not None:
            raise ValueError("slow deployment buffer contains a window without a return")
        if not self.window_returns:
            return self.empty_update_metrics()
        if len(self.window_states) != len(self.window_returns):
            raise ValueError("slow deployment window states and returns are misaligned")

        returns = np.asarray(self.window_returns, dtype=np.float32)
        old_values = np.asarray(self.window_old_values, dtype=np.float32)
        window_advantages = returns - old_values
        advantage_mean = float(window_advantages.mean())
        advantage_std = float(window_advantages.std())
        if len(window_advantages) > 1 and advantage_std > 1e-8:
            normalized_advantages = (window_advantages - advantage_mean) / (advantage_std + 1e-8)
        else:
            normalized_advantages = window_advantages.copy()

        count_ids = np.asarray(self.count_window_ids, dtype=np.int64)
        placement_ids = np.asarray(self.placement_window_ids, dtype=np.int64)
        count_metrics = self.count_ppo.update_actor(
            normalized_advantages[count_ids],
            sample_weights=self._equal_window_action_weights(count_ids),
            progress_label=f"{progress_label} count",
            progress_interval_seconds=progress_interval_seconds,
        )
        placement_metrics = self.placement_ppo.update_actor(
            normalized_advantages[placement_ids],
            sample_weights=self._equal_window_action_weights(placement_ids),
            progress_label=f"{progress_label} placement",
            progress_interval_seconds=progress_interval_seconds,
        )
        critic_metrics = self._update_window_critic(returns)
        window_count = len(self.window_returns)
        self.count_window_ids.clear()
        self.placement_window_ids.clear()
        self.window_states.clear()
        self.window_old_values.clear()
        self.window_returns.clear()
        return {
            "loss": count_metrics["loss"] + placement_metrics["loss"] + self.value_coef * critic_metrics["value_loss"],
            "policy_loss": count_metrics["policy_loss"] + placement_metrics["policy_loss"],
            "value_loss": critic_metrics["value_loss"],
            "entropy": count_metrics["entropy"] + placement_metrics["entropy"],
            "approx_kl": max(count_metrics.get("approx_kl", 0.0), placement_metrics.get("approx_kl", 0.0)),
            "window_count": float(window_count),
            "window_return_mean": float(returns.mean()),
            "window_return_std": float(returns.std()),
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
            "critic_explained_variance": critic_metrics["explained_variance"],
            "count_loss": count_metrics["loss"],
            "count_policy_loss": count_metrics["policy_loss"],
            "count_value_loss": 0.0,
            "count_entropy": count_metrics["entropy"],
            "count_approx_kl": count_metrics.get("approx_kl", 0.0),
            "placement_loss": placement_metrics["loss"],
            "placement_policy_loss": placement_metrics["policy_loss"],
            "placement_value_loss": 0.0,
            "placement_entropy": placement_metrics["entropy"],
            "placement_approx_kl": placement_metrics.get("approx_kl", 0.0),
        }

    def empty_update_metrics(self) -> dict[str, float]:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "window_count": 0.0,
            "window_return_mean": 0.0,
            "window_return_std": 0.0,
            "advantage_mean": 0.0,
            "advantage_std": 0.0,
            "critic_explained_variance": 0.0,
            "count_loss": 0.0,
            "count_policy_loss": 0.0,
            "count_value_loss": 0.0,
            "count_entropy": 0.0,
            "count_approx_kl": 0.0,
            "placement_loss": 0.0,
            "placement_policy_loss": 0.0,
            "placement_value_loss": 0.0,
            "placement_entropy": 0.0,
            "placement_approx_kl": 0.0,
        }

    def _equal_window_action_weights(self, window_ids: np.ndarray) -> np.ndarray:
        if len(window_ids) == 0:
            return np.asarray([], dtype=np.float32)
        counts = np.bincount(window_ids, minlength=len(self.window_returns)).astype(np.float32)
        return np.asarray([1.0 / max(float(counts[idx]), 1.0) for idx in window_ids], dtype=np.float32)

    def _update_window_critic(self, returns: np.ndarray) -> dict[str, float]:
        states = torch.as_tensor(np.stack(self.window_states), dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        value_loss = 0.0
        for _ in range(max(self.critic_k_epochs, 1)):
            values = self.window_critic(states)
            loss = nn.functional.mse_loss(values, targets)
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.window_critic.parameters(), 0.5)
            self.critic_optimizer.step()
            value_loss = float(loss.item())
        with torch.no_grad():
            predictions = self.window_critic(states).detach().cpu().numpy()
        target_variance = float(np.var(returns))
        explained_variance = 0.0 if target_variance <= 1e-8 else 1.0 - float(np.var(returns - predictions)) / target_variance
        return {"value_loss": value_loss, "explained_variance": explained_variance}

    def _build_state(
        self,
        env: EdgeComputingEnv,
        demand: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        service_id: int,
        stage_id: int,
        replica_idx: int,
    ) -> np.ndarray:
        assert env.scenario is not None
        nodes = env.scenario.nodes
        max_mem = max(n.memory_gb for n in nodes)
        max_storage = max(n.storage_gb for n in nodes)
        max_compute = max(n.compute_gcycles_per_s for n in nodes)
        node_features = []
        for node in nodes:
            node_features.extend(
                [
                    remaining_memory[node.node_id] / max_mem,
                    remaining_storage[node.node_id] / max_storage,
                    node.compute_gcycles_per_s / max_compute,
                    env.node_compute_load[node.node_id],
                    demand[node.node_id, service_id] / max(demand[:, service_id].max(), 1e-9),
                ]
            )
        scalars = [
            service_id / max(self.num_service_types - 1, 1),
            stage_id / max(self.max_service_stages - 1, 1),
            replica_idx / max(self.replicas_per_stage - 1, 1),
            env.current_time_minute / max(env.config.episode_hours * 60, 1),
            len(env.scenario.services[service_id].stages) / self.max_service_stages,
            np.log1p(env._arrival_rate_per_minute()) / np.log1p(20_000.0),
            np.log1p(env._arrival_rate_per_minute() * env.config.deployment_interval_minutes)
            / np.log1p(5_000_000.0),
            1.0,
        ]
        return np.asarray(scalars + node_features + demand.reshape(-1).tolist(), dtype=np.float32)

    def _build_window_state(
        self,
        env: EdgeComputingEnv,
        demand: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
    ) -> np.ndarray:
        assert env.scenario is not None
        nodes = env.scenario.nodes
        memory_capacity = env.service_memory_capacities()
        storage_capacity = env.service_storage_capacities()
        max_compute = max(n.compute_gcycles_per_s for n in nodes)
        total_node_demand = demand.sum(axis=1)
        max_node_demand = max(float(total_node_demand.max()), 1e-9)
        node_features: list[float] = []
        for node in nodes:
            node_features.extend(
                [
                    remaining_memory[node.node_id] / max(memory_capacity[node.node_id], 1e-9),
                    remaining_storage[node.node_id] / max(storage_capacity[node.node_id], 1e-9),
                    node.compute_gcycles_per_s / max_compute,
                    env.node_compute_load[node.node_id],
                    total_node_demand[node.node_id] / max_node_demand,
                ]
            )
        current_replica_rate = 0.0
        if env.deployment is not None:
            current_replica_rate = float(env.deployment.mean())
        scalars = [
            env.current_time_minute / max(env.config.episode_hours * 60, 1),
            np.log1p(env._arrival_rate_per_minute()) / np.log1p(20_000.0),
            np.log1p(env._arrival_rate_per_minute() * env.config.deployment_interval_minutes)
            / np.log1p(5_000_000.0),
            current_replica_rate,
            float(np.mean(env.node_compute_load)),
            float(np.max(env.node_compute_load)),
            float(env.config.service_resource_fraction),
            1.0,
        ]
        return np.asarray(scalars + node_features + demand.reshape(-1).tolist(), dtype=np.float32)

    def _node_service_demand(self, env: EdgeComputingEnv) -> np.ndarray:
        assert env.scenario is not None
        demand = np.zeros((self.num_nodes, self.num_service_types), dtype=np.float32)
        for user in env.scenario.users:
            demand[user.home_node] += np.asarray(user.service_weights, dtype=np.float32)
        demand /= max(float(len(env.scenario.users)), 1.0)
        demand *= float(env._arrival_rate_per_minute())
        return np.log1p(demand) / np.log1p(500.0)

    def _build_count_mask(
        self,
        stage,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        *,
        reserved_memory: float = 0.0,
        reserved_storage: float = 0.0,
    ) -> np.ndarray:
        feasible_slots = int(np.count_nonzero((remaining_memory >= stage.memory_gb) & (remaining_storage >= stage.storage_gb)))
        memory_budget = max(float(remaining_memory.sum()) - reserved_memory, 0.0)
        storage_budget = max(float(remaining_storage.sum()) - reserved_storage, 0.0)
        memory_count = int(np.floor(memory_budget / max(float(stage.memory_gb), 1e-9)))
        storage_count = int(np.floor(storage_budget / max(float(stage.storage_gb), 1e-9)))
        max_count = min(self.replicas_per_stage, feasible_slots, memory_count, storage_count)
        mask = np.zeros(self.replicas_per_stage, dtype=bool)
        if max_count > 0:
            mask[:max_count] = True
        return mask

    def _repair_with_current_or_full(self, env: EdgeComputingEnv, deployment: np.ndarray) -> np.ndarray:
        feasible, _ = env.check_deployment_feasible(deployment)
        if feasible:
            return deployment
        if env.deployment is not None:
            feasible, _ = env.check_deployment_feasible(env.deployment)
            if feasible:
                return env.deployment.copy()
        raise RuntimeError("slow PPO produced infeasible deployment and no feasible fallback exists")

    def _coverage_repair(
        self,
        env: EdgeComputingEnv,
        deployment: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
    ) -> None:
        """Repair uncovered service stages without overriding learned replica counts."""

        assert env.scenario is not None
        demand = self._node_service_demand(env)
        for service in env.scenario.services:
            node_order = np.argsort(-demand[:, service.service_id])
            for stage in service.stages:
                if deployment[service.service_id, stage.stage_id].any():
                    continue
                for node_id in node_order:
                    if remaining_memory[node_id] < stage.memory_gb or remaining_storage[node_id] < stage.storage_gb:
                        continue
                    deployment[service.service_id, stage.stage_id, node_id] = True
                    remaining_memory[node_id] -= stage.memory_gb
                    remaining_storage[node_id] -= stage.storage_gb
                    break


@dataclass
class FastSchedulingPPOAgent:
    num_nodes: int
    max_service_stages: int
    policy_kind: str = "gat_node_scorer"
    lr: float = 3e-4
    k_epochs: int = 4
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    target_kl: float | None = 0.03
    minibatch_size: int = 512
    device: str = "cpu"
    ppo: PPOAgent = field(init=False)

    def __post_init__(self) -> None:
        self.ppo = PPOAgent(
            obs_dim=fast_obs_dim(self.num_nodes, self.policy_kind),
            action_dim=self.num_nodes,
            hidden_dim=128,
            lr=self.lr,
            gamma=0.99,
            k_epochs=self.k_epochs,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            target_kl=self.target_kl,
            minibatch_size=self.minibatch_size,
            policy_kind=self.policy_kind,
            global_dim=FAST_GLOBAL_DIM,
            node_feature_dim=FAST_NODE_FEATURE_DIM,
            edge_feature_dim=FAST_EDGE_FEATURE_DIM,
            num_nodes=self.num_nodes,
            device=self.device,
        )

    def schedule(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest | None = None,
        deterministic: bool = False,
        record: bool = True,
    ) -> list[int]:
        env._require_ready()
        if request is None:
            assert env.current_request is not None
            request = env.current_request

        stage_nodes: list[int] = []
        masks: list[np.ndarray] = []
        states: list[np.ndarray] = []
        actions: list[int] = []
        logprobs: list[float] = []
        values: list[float] = []

        for stage_id in range(len(request.stage_compute_gcycles)):
            state = self._build_state(env, request, stage_id, stage_nodes)
            mask = self._build_mask(env, request, stage_id, stage_nodes)
            action, logprob, value = self.ppo.act(state, mask, deterministic=deterministic)
            if not mask[action]:
                action = int(np.where(mask)[0][0])
            stage_nodes.append(action)
            states.append(state)
            masks.append(mask)
            actions.append(action)
            logprobs.append(logprob)
            values.append(value)

        if record:
            for state, mask, action, logprob, value in zip(states, masks, actions, logprobs, values):
                self.ppo.buffer.states.append(state)
                self.ppo.buffer.masks.append(mask.astype(bool))
                self.ppo.buffer.actions.append(action)
                self.ppo.buffer.logprobs.append(logprob)
                self.ppo.buffer.values.append(value)
        return stage_nodes

    def schedule_with_diagnostics(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest | None = None,
    ) -> tuple[list[int], list[dict[str, float | int]]]:
        env._require_ready()
        if request is None:
            assert env.current_request is not None
            request = env.current_request

        stage_nodes: list[int] = []
        diagnostics: list[dict[str, float | int]] = []
        for stage_id in range(len(request.stage_compute_gcycles)):
            state = self._build_state(env, request, stage_id, stage_nodes)
            mask = self._build_mask(env, request, stage_id, stage_nodes)
            stats = self.ppo.action_stats(state, mask)
            action = int(stats["action"])
            if not mask[action]:
                action = int(np.where(mask)[0][0])
                stats["action"] = action
            stage_nodes.append(action)
            diagnostics.append(stats)
        return stage_nodes, diagnostics

    def assign_last_schedule_reward(self, reward: float, stage_count: int, done: bool, weight: float = 1.0) -> None:
        for stage_idx in range(stage_count):
            self.ppo.buffer.rewards.append(float(reward))
            self.ppo.buffer.dones.append(bool(done or stage_idx == stage_count - 1))
            self.ppo.buffer.weights.append(float(weight))

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        return self.ppo.update(progress_label=progress_label, progress_interval_seconds=progress_interval_seconds)

    def _build_state(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        partial_nodes: list[int],
    ) -> np.ndarray:
        assert env.scenario is not None
        prev_node = request.home_node if not partial_nodes else partial_nodes[-1]
        nodes = env.scenario.nodes
        max_compute = max(n.compute_gcycles_per_s for n in nodes)
        max_bandwidth = np.nanmax(np.where(np.isfinite(env.scenario.bandwidth_mb_s), env.scenario.bandwidth_mb_s, 0.0))
        tick_request_count = float(sum(item.request_count for item in env.current_requests))
        tick_group_count = float(len(env.current_requests))
        tick_service_count = float(
            sum(item.request_count for item in env.current_requests if item.service_id == request.service_id)
        )
        tick_node_counts = np.zeros(self.num_nodes, dtype=np.float64)
        for item in env.current_requests:
            tick_node_counts[item.home_node] += float(item.request_count)
        node_features = []
        deployed = env.deployment[request.service_id, stage_id] if env.deployment is not None else np.zeros(self.num_nodes, dtype=bool)
        for node in nodes:
            bandwidth = env.scenario.bandwidth_mb_s[prev_node, node.node_id]
            if not np.isfinite(bandwidth):
                bandwidth = 0.0
            reachable = prev_node == node.node_id or env.scenario.adjacency[prev_node, node.node_id]
            node_features.extend(
                [
                    float(deployed[node.node_id]),
                    float(reachable),
                    node.compute_gcycles_per_s / max_compute,
                    env.node_compute_load[node.node_id],
                    bandwidth / max(max_bandwidth, 1e-9),
                    tick_node_counts[node.node_id] / max(tick_request_count, 1.0),
                ]
            )
        scalars = [
            request.service_id / max(env.config.num_service_types - 1, 1),
            stage_id / max(self.max_service_stages - 1, 1),
            prev_node / max(self.num_nodes - 1, 1),
            request.input_mb / 2.0,
            request.stage_compute_gcycles[stage_id] / 8.0,
            request.deadline_s / 0.6,
            request.request_count / 100.0,
            env.current_time_minute / max(env.config.episode_hours * 60, 1),
            len(request.stage_compute_gcycles) / self.max_service_stages,
            np.log1p(tick_request_count) / np.log1p(5_000.0),
            tick_group_count / max(env.config.num_edge_nodes * env.config.num_service_types, 1),
            tick_service_count / max(tick_request_count, 1.0),
        ]
        if self.policy_kind == "gat_node_scorer":
            node_features += self._build_edge_features(env)
        return np.asarray(scalars + node_features, dtype=np.float32)

    def _build_edge_features(self, env: EdgeComputingEnv) -> list[float]:
        assert env.scenario is not None
        bandwidth = env.scenario.bandwidth_mb_s
        propagation = env.scenario.propagation_ms
        max_bandwidth = np.nanmax(np.where(np.isfinite(bandwidth), bandwidth, 0.0))
        finite_propagation = np.where(np.isfinite(propagation), propagation, 0.0)
        max_propagation = max(float(finite_propagation.max()), 1e-9)
        features: list[float] = []
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                connected = bool(env.scenario.adjacency[src, dst])
                bw = bandwidth[src, dst] if np.isfinite(bandwidth[src, dst]) else max_bandwidth
                prop = propagation[src, dst] if np.isfinite(propagation[src, dst]) else max_propagation
                features.extend(
                    [
                        float(connected),
                        float(bw / max(max_bandwidth, 1e-9)),
                        float(prop / max_propagation),
                    ]
                )
        return features

    def _build_mask(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        partial_nodes: list[int],
    ) -> np.ndarray:
        assert env.scenario is not None
        mask = env.scheduler_candidate_mask(request)[stage_id].copy()
        prev_node = request.home_node if not partial_nodes else partial_nodes[-1]
        reachable = env.scenario.adjacency[prev_node].copy()
        reachable[prev_node] = True
        mask &= reachable
        if not mask.any():
            mask = env.scheduler_candidate_mask(request)[stage_id].copy()
        if not mask.any():
            mask[request.home_node] = True
        return mask.astype(bool)


@dataclass
class HierarchicalPPOAgent:
    slow_agent: SlowDeploymentPPOAgent
    fast_agent: FastSchedulingPPOAgent
    window_reward: float = 0.0
    window_steps: float = 0.0
    slow_reward_scale: float = 1.0
    slow_deployment_memory_coef: float = 0.03
    slow_deployment_storage_coef: float = 0.01
    slow_migration_coef: float = 0.0
    window_deployment_memory_fraction: float = 0.0
    window_deployment_storage_fraction: float = 0.0
    window_migration_fraction: float = 0.0
    last_slow_window_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        env: EdgeComputingEnv,
        device: str = "cpu",
        replicas_per_stage: int = 5,
        slow_lr: float = 3e-4,
        fast_lr: float = 3e-4,
        slow_k_epochs: int = 3,
        fast_k_epochs: int = 4,
        slow_entropy_coef: float = 0.001,
        slow_count_entropy_coef: float | None = None,
        slow_placement_entropy_coef: float | None = None,
        fast_entropy_coef: float = 0.0,
        slow_value_coef: float = 0.5,
        slow_critic_lr: float | None = None,
        slow_critic_k_epochs: int = 4,
        fast_value_coef: float = 0.5,
        slow_target_kl: float | None = 0.03,
        fast_target_kl: float | None = 0.03,
        slow_minibatch_size: int = 2048,
        fast_minibatch_size: int = 512,
        fast_policy_kind: str = "gat_node_scorer",
        slow_reward_scale: float = 1.0,
        slow_deployment_memory_coef: float = 0.03,
        slow_deployment_storage_coef: float = 0.01,
        slow_migration_coef: float = 0.0,
    ) -> "HierarchicalPPOAgent":
        return cls(
            slow_agent=SlowDeploymentPPOAgent(
                num_nodes=env.config.num_edge_nodes,
                num_service_types=env.config.num_service_types,
                max_service_stages=env.config.max_service_stages,
                replicas_per_stage=replicas_per_stage,
                lr=slow_lr,
                k_epochs=slow_k_epochs,
                entropy_coef=slow_entropy_coef,
                count_entropy_coef=slow_count_entropy_coef,
                placement_entropy_coef=slow_placement_entropy_coef,
                value_coef=slow_value_coef,
                critic_lr=slow_critic_lr,
                critic_k_epochs=slow_critic_k_epochs,
                target_kl=slow_target_kl,
                minibatch_size=slow_minibatch_size,
                device=device,
            ),
            fast_agent=FastSchedulingPPOAgent(
                num_nodes=env.config.num_edge_nodes,
                max_service_stages=env.config.max_service_stages,
                policy_kind=fast_policy_kind,
                lr=fast_lr,
                k_epochs=fast_k_epochs,
                entropy_coef=fast_entropy_coef,
                value_coef=fast_value_coef,
                target_kl=fast_target_kl,
                minibatch_size=fast_minibatch_size,
                device=device,
            ),
            slow_reward_scale=slow_reward_scale,
            slow_deployment_memory_coef=slow_deployment_memory_coef,
            slow_deployment_storage_coef=slow_deployment_storage_coef,
            slow_migration_coef=slow_migration_coef,
        )

    def maybe_update_deployment(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> None:
        if not env.needs_deployment_update:
            return
        if record:
            self.flush_slow_window_reward(done=False)
        deployment = self.slow_agent.plan_deployment(env, deterministic=deterministic, record=record)
        migration_count = env.apply_deployment(deployment)
        if record:
            memory_fraction, storage_fraction = self._deployment_resource_fractions(env)
            assert env.scenario is not None
            stage_count = sum(len(service.stages) for service in env.scenario.services)
            possible_placements = max(stage_count * env.config.num_edge_nodes, 1)
            self.window_deployment_memory_fraction = memory_fraction
            self.window_deployment_storage_fraction = storage_fraction
            self.window_migration_fraction = float(migration_count / possible_placements)

    def act(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> list[int]:
        self.maybe_update_deployment(env, deterministic=deterministic, record=record)
        return self.fast_agent.schedule(env, deterministic=deterministic, record=record)

    def act_batch(
        self,
        env: EdgeComputingEnv,
        deterministic: bool = False,
        record: bool = True,
    ) -> list[list[int]]:
        self.maybe_update_deployment(env, deterministic=deterministic, record=record)
        return [
            self.fast_agent.schedule(env, request=request, deterministic=deterministic, record=record)
            for request in env.current_requests
        ]

    def observe_step_reward(
        self,
        reward: float,
        stage_count: int,
        done: bool,
        weight: float = 1.0,
        slow_reward: float | None = None,
    ) -> None:
        self.fast_agent.assign_last_schedule_reward(reward, stage_count, done, weight=weight)
        self.window_reward += (reward if slow_reward is None else slow_reward) * weight
        self.window_steps += weight
        if done:
            self.flush_slow_window_reward(done=True)

    def flush_slow_window_reward(self, done: bool) -> None:
        latency_return = self.window_reward / float(self.window_steps) if self.window_steps > 0 else 0.0
        deployment_memory_cost = self.slow_deployment_memory_coef * self.window_deployment_memory_fraction
        deployment_storage_cost = self.slow_deployment_storage_coef * self.window_deployment_storage_fraction
        migration_cost = self.slow_migration_coef * self.window_migration_fraction
        operating_cost = self.slow_reward_scale * (
            deployment_memory_cost + deployment_storage_cost + migration_cost
        )
        window_return = latency_return - operating_cost
        self.slow_agent.assign_pending_reward(window_return, done=done)
        self.last_slow_window_metrics = {
            "slow_window_return": float(window_return),
            "slow_window_latency_return": float(latency_return),
            "slow_deployment_memory_cost": float(self.slow_reward_scale * deployment_memory_cost),
            "slow_deployment_storage_cost": float(self.slow_reward_scale * deployment_storage_cost),
            "slow_migration_cost": float(self.slow_reward_scale * migration_cost),
            "slow_deployment_memory_fraction": float(self.window_deployment_memory_fraction),
            "slow_deployment_storage_fraction": float(self.window_deployment_storage_fraction),
            "slow_migration_fraction": float(self.window_migration_fraction),
        }
        self.window_reward = 0.0
        self.window_steps = 0
        self.window_deployment_memory_fraction = 0.0
        self.window_deployment_storage_fraction = 0.0
        self.window_migration_fraction = 0.0

    def _deployment_resource_fractions(self, env: EdgeComputingEnv) -> tuple[float, float]:
        assert env.scenario is not None
        assert env.deployment is not None
        memory_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        storage_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        for service in env.scenario.services:
            for stage in service.stages:
                placed = env.deployment[service.service_id, stage.stage_id]
                memory_used += placed * stage.memory_gb
                storage_used += placed * stage.storage_gb
        memory_fraction = float(memory_used.sum() / max(float(env.service_memory_capacities().sum()), 1e-9))
        storage_fraction = float(storage_used.sum() / max(float(env.service_storage_capacities().sum()), 1e-9))
        return memory_fraction, storage_fraction

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, dict[str, float]]:
        return {
            "slow": self.update_slow(
                progress_label=f"{progress_label} slow PPO",
                progress_interval_seconds=progress_interval_seconds,
            ),
            "fast": self.update_fast(
                progress_label=f"{progress_label} fast PPO",
                progress_interval_seconds=progress_interval_seconds,
            ),
        }

    @property
    def completed_slow_windows(self) -> int:
        return len(self.slow_agent.window_returns)

    def update_slow(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        return self.slow_agent.update(
            progress_label=progress_label,
            progress_interval_seconds=progress_interval_seconds,
        )

    def update_fast(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        return self.fast_agent.update(
            progress_label=progress_label,
            progress_interval_seconds=progress_interval_seconds,
        )
