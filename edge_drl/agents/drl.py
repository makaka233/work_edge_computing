from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest
from edge_drl.models.ppo import PPOAgent


def slow_obs_dim(num_nodes: int, num_service_types: int) -> int:
    return 6 + num_nodes * 5 + num_nodes * num_service_types


def fast_obs_dim(num_nodes: int, policy_kind: str = "gat_node_scorer") -> int:
    dim = FAST_GLOBAL_DIM + num_nodes * FAST_NODE_FEATURE_DIM
    if policy_kind == "gat_node_scorer":
        dim += num_nodes * num_nodes * FAST_EDGE_FEATURE_DIM
    return dim


FAST_GLOBAL_DIM = 9
FAST_NODE_FEATURE_DIM = 5
FAST_EDGE_FEATURE_DIM = 3


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
    target_kl: float | None = 0.03
    minibatch_size: int = 2048
    device: str = "cpu"
    count_ppo: PPOAgent = field(init=False)
    placement_ppo: PPOAgent = field(init=False)
    ppo: PPOAgent = field(init=False)
    pending_count_indices: list[int] = field(default_factory=list)
    pending_placement_indices: list[int] = field(default_factory=list)

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

    def plan_deployment(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> np.ndarray:
        env._require_ready()
        assert env.scenario is not None
        current = env.deployment if env.deployment is not None else np.zeros(
            (self.num_service_types, self.max_service_stages, self.num_nodes), dtype=bool
        )
        deployment = np.zeros_like(current, dtype=bool)
        remaining_memory = np.array([n.memory_gb for n in env.scenario.nodes], dtype=np.float64)
        remaining_storage = np.array([n.storage_gb for n in env.scenario.nodes], dtype=np.float64)
        demand = self._node_service_demand(env)
        self.pending_count_indices.clear()
        self.pending_placement_indices.clear()

        for service in env.scenario.services:
            for stage in service.stages:
                count_state = self._build_state(env, demand, remaining_memory, remaining_storage, service.service_id, stage.stage_id, 0)
                count_mask = self._build_count_mask(stage, remaining_memory, remaining_storage)
                if not count_mask.any():
                    continue
                count_action, count_logprob, count_value = self.count_ppo.act(count_state, count_mask, deterministic=deterministic)
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

                for replica_idx in range(target_replicas):
                    state = self._build_state(env, demand, remaining_memory, remaining_storage, service.service_id, stage.stage_id, replica_idx)
                    already = deployment[service.service_id, stage.stage_id]
                    feasible_new = (remaining_memory >= stage.memory_gb) & (remaining_storage >= stage.storage_gb) & ~already
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
        self._assign_pending_reward(self.count_ppo, reward, done)
        self._assign_pending_reward(self.placement_ppo, reward, done)
        self.pending_count_indices.clear()
        self.pending_placement_indices.clear()

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        self.assign_pending_reward(0.0, True)
        count_metrics = self.count_ppo.update(
            progress_label=f"{progress_label} count",
            progress_interval_seconds=progress_interval_seconds,
        )
        placement_metrics = self.placement_ppo.update(
            progress_label=f"{progress_label} placement",
            progress_interval_seconds=progress_interval_seconds,
        )
        return {
            "loss": count_metrics["loss"] + placement_metrics["loss"],
            "policy_loss": count_metrics["policy_loss"] + placement_metrics["policy_loss"],
            "value_loss": count_metrics["value_loss"] + placement_metrics["value_loss"],
            "entropy": count_metrics["entropy"] + placement_metrics["entropy"],
            "approx_kl": max(count_metrics.get("approx_kl", 0.0), placement_metrics.get("approx_kl", 0.0)),
            "count_loss": count_metrics["loss"],
            "count_policy_loss": count_metrics["policy_loss"],
            "count_value_loss": count_metrics["value_loss"],
            "count_entropy": count_metrics["entropy"],
            "count_approx_kl": count_metrics.get("approx_kl", 0.0),
            "placement_loss": placement_metrics["loss"],
            "placement_policy_loss": placement_metrics["policy_loss"],
            "placement_value_loss": placement_metrics["value_loss"],
            "placement_entropy": placement_metrics["entropy"],
            "placement_approx_kl": placement_metrics.get("approx_kl", 0.0),
        }

    def _assign_pending_reward(self, ppo: PPOAgent, reward: float, done: bool) -> None:
        missing = len(ppo.buffer.actions) - len(ppo.buffer.rewards)
        for action_idx in range(missing):
            ppo.buffer.rewards.append(float(reward))
            ppo.buffer.dones.append(bool(done or action_idx == missing - 1))

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
            1.0,
        ]
        return np.asarray(scalars + node_features + demand.reshape(-1).tolist(), dtype=np.float32)

    def _node_service_demand(self, env: EdgeComputingEnv) -> np.ndarray:
        assert env.scenario is not None
        demand = np.zeros((self.num_nodes, self.num_service_types), dtype=np.float32)
        for user in env.scenario.users:
            demand[user.home_node] += np.asarray(user.service_weights, dtype=np.float32)
        demand /= max(demand.max(), 1e-9)
        return demand

    def _build_count_mask(
        self,
        stage,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
    ) -> np.ndarray:
        feasible_slots = int(np.count_nonzero((remaining_memory >= stage.memory_gb) & (remaining_storage >= stage.storage_gb)))
        max_count = min(self.replicas_per_stage, feasible_slots)
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

    def assign_last_schedule_reward(self, reward: float, stage_count: int, done: bool) -> None:
        for stage_idx in range(stage_count):
            self.ppo.buffer.rewards.append(float(reward))
            self.ppo.buffer.dones.append(bool(done or stage_idx == stage_count - 1))

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
        fast_value_coef: float = 0.5,
        slow_target_kl: float | None = 0.03,
        fast_target_kl: float | None = 0.03,
        slow_minibatch_size: int = 2048,
        fast_minibatch_size: int = 512,
        fast_policy_kind: str = "gat_node_scorer",
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
        )

    def maybe_update_deployment(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> None:
        if not env.needs_deployment_update:
            return
        if record:
            self.flush_slow_window_reward(done=False)
        deployment = self.slow_agent.plan_deployment(env, deterministic=deterministic, record=record)
        env.apply_deployment(deployment)

    def act(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> list[int]:
        self.maybe_update_deployment(env, deterministic=deterministic, record=record)
        return self.fast_agent.schedule(env, deterministic=deterministic, record=record)

    def observe_step_reward(self, reward: float, stage_count: int, done: bool, weight: float = 1.0) -> None:
        self.fast_agent.assign_last_schedule_reward(reward, stage_count, done)
        self.window_reward += reward * weight
        self.window_steps += weight
        if done:
            self.flush_slow_window_reward(done=True)

    def flush_slow_window_reward(self, done: bool) -> None:
        if self.window_steps <= 0:
            return
        averaged_reward = self.window_reward / float(self.window_steps)
        self.slow_agent.assign_pending_reward(averaged_reward, done=done)
        self.window_reward = 0.0
        self.window_steps = 0

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, dict[str, float]]:
        return {
            "slow": self.slow_agent.update(
                progress_label=f"{progress_label} slow PPO",
                progress_interval_seconds=progress_interval_seconds,
            ),
            "fast": self.fast_agent.update(
                progress_label=f"{progress_label} fast PPO",
                progress_interval_seconds=progress_interval_seconds,
            ),
        }
