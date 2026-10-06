"""Single-policy end-to-end PPO baseline.

Unlike the Proposed hierarchy, this agent has one actor-critic, one optimizer,
conditional action heads, and one temporal reward stream. A masked categorical
head selects deployment/routing nodes; two continuous priorities independently
control compute and incoming-link shares. Capacity normalization uses the
``learned-share`` allocator and does not call the analytical KKT solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest
from edge_drl.models.unified_ppo import UnifiedHybridPPO


UNIFIED_GLOBAL_DIM = 21
UNIFIED_NODE_FEATURE_DIM = 21
UNIFIED_DECISION_MODES = {"coverage": 0, "extra": 1, "schedule": 2}


def unified_obs_dim(num_nodes: int, num_services: int = 0, max_stages: int = 0) -> int:
    return (UNIFIED_GLOBAL_DIM + num_nodes * UNIFIED_NODE_FEATURE_DIM
            + num_services * max_stages * num_nodes + 3 * num_nodes * num_nodes
            + num_services * num_nodes + 2 * max_stages)


@dataclass
class UnifiedPPOAgent:
    """One PPO policy for deployment, routing, and resource priorities."""

    num_nodes: int
    num_service_types: int
    max_service_stages: int
    hidden_dim: int = 256
    lr: float = 1e-4
    environment_gamma: float = 0.95
    k_epochs: int = 4
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    target_kl: float | None = 0.015
    minibatch_size: int = 2048
    reward_scale: float = 1.0
    device: str = "cpu"
    ppo: UnifiedHybridPPO = field(init=False)
    last_resource_weights: list[list[float]] = field(default_factory=list, init=False)
    decision_counts: dict[str, int] = field(default_factory=dict, init=False)
    action_modes: list[str] = field(default_factory=list, init=False)
    action_returns: list[float | None] = field(default_factory=list, init=False)
    action_weights: list[float | None] = field(default_factory=list, init=False)
    window_start_index: int | None = field(default=None, init=False)
    window_steps: list[tuple[int, int, float]] = field(default_factory=list, init=False)
    action_increment_costs: dict[int, float] = field(default_factory=dict, init=False)
    ledger: dict[tuple, tuple[float, float]] = field(default_factory=dict, init=False)
    batch_total_count: float = field(default=1.0, init=False)
    batch_increment_cost: float = field(default=0.0, init=False)
    max_settlement_error: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if not 0.0 <= self.environment_gamma <= 1.0:
            raise ValueError("environment_gamma must be in [0, 1]")
        self.action_dim = self.num_nodes + 1
        self.stop_action = self.action_dim - 1
        self.ppo = UnifiedHybridPPO(
            obs_dim=unified_obs_dim(self.num_nodes, self.num_service_types, self.max_service_stages),
            num_nodes=self.num_nodes,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            k_epochs=self.k_epochs,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            target_kl=self.target_kl,
            minibatch_size=self.minibatch_size,
            device=self.device,
        )
        self.reset_decision_counts()

    @classmethod
    def from_env(cls, env: EdgeComputingEnv, **kwargs: Any) -> "UnifiedPPOAgent":
        return cls(
            num_nodes=env.config.num_edge_nodes,
            num_service_types=env.config.num_service_types,
            max_service_stages=env.config.max_service_stages,
            **kwargs,
        )

    def reset_decision_counts(self) -> None:
        self.decision_counts = {key: 0 for key in UNIFIED_DECISION_MODES}
        self.sampled_priorities = np.empty((0, 2))
        self.priority_sum = np.zeros(2, dtype=np.float64)
        self.priority_count = 0
        self.stop_action_count = 0
        self.deployment_replica_sum = 0.0
        self.deployment_window_count = 0

    def begin_window(self) -> None:
        if self.window_start_index is not None:
            raise RuntimeError("previous UnifiedPPO window has not been finalized")
        self.window_start_index = len(self.ppo.buffer.actions)
        self.window_steps = []

    def begin_environment_step(self) -> int:
        if self.window_start_index is None:
            self.begin_window()
        return len(self.ppo.buffer.actions)

    def finish_environment_step(
        self,
        start_index: int,
        reward: float,
        *,
        done: bool,
        group_infos: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record one real environment-step reward for later macro credit."""

        end_index = len(self.ppo.buffer.actions)
        if start_index < 0 or start_index > end_index:
            raise ValueError("invalid UnifiedPPO environment-step action range")
        del done
        if group_infos is not None:
            physical_cost = sum(float(item["latency_s"]) * float(item["request_count"])
                                for item in group_infos)
            represented_count = sum(float(item["request_count"]) for item in group_infos)
            physical_mean = physical_cost / max(represented_count, 1.0)
            ledger_mean = self.batch_increment_cost / max(self.batch_total_count, 1.0)
            error = abs(physical_mean - ledger_mean)
            self.max_settlement_error = max(self.max_settlement_error, error)
            if not np.isclose(physical_mean, ledger_mean, rtol=1e-6, atol=1e-6):
                raise RuntimeError(f"UnifiedPPO ledger disagrees with settlement: {error:.6g}s/request")
        self.window_steps.append((start_index, end_index, float(reward) * self.reward_scale))

    def finish_window(self) -> None:
        """Give factorized actions returns on the physical environment clock.

        All routing/resource factors in one settlement share that settlement's
        normalized discounted macro return. Deployment factors share the first
        macro return of the ten-minute window. Inverse group-size weights make
        a 200-factor joint action contribute the same total optimizer weight as
        any other environment step instead of dominating merely because it has
        more requests.
        """

        if self.window_start_index is None:
            raise RuntimeError("UnifiedPPO window was not started")
        if not self.window_steps:
            raise RuntimeError("UnifiedPPO window contains no environment settlements")

        macro_returns = np.zeros(len(self.window_steps), dtype=np.float64)
        numerator = 0.0
        denominator = 0.0
        for index in reversed(range(len(self.window_steps))):
            reward = self.window_steps[index][2]
            numerator = reward + self.environment_gamma * numerator
            denominator = 1.0 + self.environment_gamma * denominator
            macro_returns[index] = numerator / max(denominator, 1e-9)

        deployment_indices = [
            index
            for index in range(self.window_start_index, len(self.action_modes))
            if self.action_modes[index] in {"coverage", "extra"}
        ]
        if deployment_indices:
            weight = 1.0 / len(deployment_indices)
            for index in deployment_indices:
                self.action_returns[index] = float(macro_returns[0])
                self.action_weights[index] = weight

        for step_index, (start, end, _) in enumerate(self.window_steps):
            schedule_indices = [
                index
                for index in range(start, end)
                if self.action_modes[index] == "schedule"
            ]
            if not schedule_indices:
                continue
            weight = 1.0 / len(schedule_indices)
            suffix_cost = 0.0
            for index in reversed(schedule_indices):
                if index in self.action_increment_costs:
                    suffix_cost += self.action_increment_costs[index]
                    # Causal reward-to-go over the autoregressive joint action.
                    # Prior increments do not depend on this action and are omitted.
                    future_numerator = 0.0
                    future_denominator = 1.0
                    discount = self.environment_gamma
                    for _, _, future_reward in self.window_steps[step_index + 1:]:
                        future_numerator += discount * future_reward
                        future_denominator += discount
                        discount *= self.environment_gamma
                    self.action_returns[index] = float(
                        (-suffix_cost * self.reward_scale + future_numerator) / future_denominator)
                else:
                    self.action_returns[index] = float(macro_returns[step_index])
                self.action_weights[index] = weight

        unresolved = [
            index
            for index in range(self.window_start_index, len(self.action_modes))
            if self.action_returns[index] is None or self.action_weights[index] is None
        ]
        if unresolved:
            raise RuntimeError(
                f"UnifiedPPO macro credit left {len(unresolved)} actions unresolved"
            )
        self.window_start_index = None
        self.window_steps = []

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        if self.window_start_index is not None:
            raise RuntimeError("cannot update UnifiedPPO with an unfinished window")
        action_count = len(self.ppo.buffer.actions)
        if not (
            len(self.action_modes)
            == len(self.action_returns)
            == len(self.action_weights)
            == action_count
        ):
            raise RuntimeError("UnifiedPPO macro-credit arrays do not match its PPO buffer")
        if any(value is None for value in self.action_returns):
            raise RuntimeError("UnifiedPPO contains actions without macro returns")
        if any(value is None for value in self.action_weights):
            raise RuntimeError("UnifiedPPO contains actions without macro sample weights")
        returns = np.asarray(self.action_returns, dtype=np.float32)
        weights = np.asarray(self.action_weights, dtype=np.float32)
        return_mean = float(np.average(returns, weights=weights))
        return_std = float(
            np.sqrt(np.average((returns - return_mean) ** 2, weights=weights))
        )
        metrics = self.ppo.update_from_returns(
            returns,
            sample_weights=weights,
            progress_label=progress_label,
            progress_interval_seconds=progress_interval_seconds,
        )
        metrics.update(
            {
                "coverage_actions": float(self.decision_counts["coverage"]),
                "extra_actions": float(self.decision_counts["extra"]),
                "schedule_actions": float(self.decision_counts["schedule"]),
                "stop_actions": float(self.stop_action_count),
                "mean_deployment_replicas": (
                    self.deployment_replica_sum / max(self.deployment_window_count, 1)
                ),
                "macro_return_mean": return_mean,
                "macro_return_std": return_std,
                "macro_effective_groups": float(weights.sum()),
                "ledger_settlement_error_s": self.max_settlement_error,
                "compute_priority_mean": float(self.priority_sum[0] / self.priority_count) if self.priority_count else 1.0,
                "link_priority_mean": float(self.priority_sum[1] / self.priority_count) if self.priority_count else 1.0,
            }
        )
        self.action_modes.clear()
        self.action_returns.clear()
        self.action_weights.clear()
        self.action_increment_costs.clear()
        self.max_settlement_error = 0.0
        self.reset_decision_counts()
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "design": "single-policy-ledger-credit-v3",
            "policy": self.ppo.policy.state_dict(),
            "optimizer": self.ppo.optimizer.state_dict(),
            "environment_gamma": self.environment_gamma,
            "architecture": {
                "num_nodes": self.num_nodes,
                "num_service_types": self.num_service_types,
                "max_service_stages": self.max_service_stages,
                "hidden_dim": self.hidden_dim,
                "action_dim": self.action_dim,
                "obs_dim": unified_obs_dim(self.num_nodes, self.num_service_types, self.max_service_stages),
            },
        }

    def load_state_dict(self, state: dict[str, Any], *, load_optimizer: bool = True) -> None:
        if state.get("design") != "single-policy-ledger-credit-v3":
            raise ValueError("UnifiedPPO v3 requires a fresh run; older actor/credit layouts are incompatible")
        self.ppo.policy.load_state_dict(state["policy"])
        if load_optimizer and "optimizer" in state:
            self.ppo.optimizer.load_state_dict(state["optimizer"])

    def _record_action(
        self,
        *,
        state: np.ndarray,
        mask: np.ndarray,
        action: int,
        logprob: float,
        value: float,
        mode: str,
        env: EdgeComputingEnv,
    ) -> None:
        self.ppo.buffer.add(
            state=state,
            mask=mask,
            action=action,
            logprob=logprob,
            reward=0.0,
            done=False,
            value=value,
            weight=1.0,
            sample_group=float(
                env.config.demand_load_multiplier
                if env.config.demand_load_group is None
                else env.config.demand_load_group
            ),
        )
        self.ppo.priorities.append(self.ppo.last_priorities.copy())
        self.ppo.active.append(self.ppo.last_active.copy())
        self.action_modes.append(mode)
        self.action_returns.append(None)
        self.action_weights.append(None)
        if action == self.stop_action:
            self.stop_action_count += 1
        self.decision_counts[mode] += 1

    def _act(
        self,
        env: EdgeComputingEnv,
        *,
        mode: str,
        service_id: int,
        stage_id: int,
        stage_count: int,
        replica_index: int,
        deployment: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        candidate_nodes: np.ndarray,
        source_node: int | None,
        request: TaskRequest | None,
        batch_progress: float,
        allow_stop: bool,
        demand: np.ndarray,
        deterministic: bool,
        record: bool,
    ) -> tuple[int, np.ndarray]:
        state = self._build_state(
            env,
            mode=mode,
            service_id=service_id,
            stage_id=stage_id,
            stage_count=stage_count,
            replica_index=replica_index,
            deployment=deployment,
            remaining_memory=remaining_memory,
            remaining_storage=remaining_storage,
            candidate_nodes=candidate_nodes,
            source_node=source_node,
            request=request,
            batch_progress=batch_progress,
            demand=demand,
        )
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[:self.num_nodes] = candidate_nodes
        if allow_stop:
            mask[self.stop_action] = True
        if not mask.any():
            raise RuntimeError(f"UnifiedPPO has no valid {mode} action")
        action, logprob, value = self.ppo.act(state, mask, deterministic=deterministic)
        action = int(action)
        if not mask[action]:
            raise RuntimeError("UnifiedPPO sampled a masked action")
        if record:
            self._record_action(
                state=state,
                mask=mask,
                action=action,
                logprob=float(logprob),
                value=float(value),
                mode=mode,
                env=env,
            )
        if action == self.stop_action:
            return -1, np.ones(2, dtype=np.float32)
        # Log-uniform bounded scale; Beta learns continuous priorities, not bins.
        priorities = np.exp(np.log(0.1) + self.ppo.last_priorities * np.log(100.0))
        return action, priorities

    def plan_deployment(
        self,
        env: EdgeComputingEnv,
        *,
        deterministic: bool,
        record: bool,
    ) -> np.ndarray:
        env._require_ready()
        assert env.scenario is not None and env.deployment is not None
        # Deployment must not observe stale reservations from the last batch.
        self.ledger = {}
        deployment = np.zeros_like(env.deployment, dtype=bool)
        remaining_memory = env.service_memory_capacities().copy()
        remaining_storage = env.service_storage_capacities().copy()
        demand = self._node_service_demand(env)
        stages = [
            (service, stage)
            for service in env.scenario.services
            for stage in service.stages
        ]
        memory_capacity = np.maximum(env.service_memory_capacities(), 1e-9)
        storage_capacity = np.maximum(env.service_storage_capacities(), 1e-9)
        stages.sort(
            key=lambda item: (
                -max(
                    float(item[1].memory_gb / np.max(memory_capacity)),
                    float(item[1].storage_gb / np.max(storage_capacity)),
                ),
                int(item[0].service_id),
                int(item[1].stage_id),
            )
        )

        # First establish legal coverage entirely through the shared policy.
        for service, stage in stages:
            candidates = (
                (remaining_memory + 1e-9 >= float(stage.memory_gb))
                & (remaining_storage + 1e-9 >= float(stage.storage_gb))
            )
            if not candidates.any():
                raise RuntimeError(
                    f"UnifiedPPO cannot cover service {service.service_id} stage {stage.stage_id}"
                )
            node_id, _ = self._act(
                env,
                mode="coverage",
                service_id=int(service.service_id),
                stage_id=int(stage.stage_id),
                stage_count=len(service.stages),
                replica_index=0,
                deployment=deployment,
                remaining_memory=remaining_memory,
                remaining_storage=remaining_storage,
                candidate_nodes=candidates,
                source_node=None,
                request=None,
                batch_progress=0.0,
                allow_stop=False,
                demand=demand,
                deterministic=deterministic,
                record=record,
            )
            deployment[service.service_id, stage.stage_id, node_id] = True
            remaining_memory[node_id] -= float(stage.memory_gb)
            remaining_storage[node_id] -= float(stage.storage_gb)

        # The same policy decides how many additional replicas to create by
        # repeatedly selecting a node or STOP.  Coverage can no longer be lost.
        for service, stage in sorted(
            stages,
            key=lambda item: (int(item[0].service_id), int(item[1].stage_id)),
        ):
            for replica_index in range(1, self.num_nodes):
                candidates = (
                    (remaining_memory + 1e-9 >= float(stage.memory_gb))
                    & (remaining_storage + 1e-9 >= float(stage.storage_gb))
                    & ~deployment[service.service_id, stage.stage_id]
                )
                node_id, _ = self._act(
                    env,
                    mode="extra",
                    service_id=int(service.service_id),
                    stage_id=int(stage.stage_id),
                    stage_count=len(service.stages),
                    replica_index=replica_index,
                    deployment=deployment,
                    remaining_memory=remaining_memory,
                    remaining_storage=remaining_storage,
                    candidate_nodes=candidates,
                    source_node=None,
                    request=None,
                    batch_progress=0.0,
                    allow_stop=True,
                    demand=demand,
                    deterministic=deterministic,
                    record=record,
                )
                if node_id < 0:
                    break
                deployment[service.service_id, stage.stage_id, node_id] = True
                remaining_memory[node_id] -= float(stage.memory_gb)
                remaining_storage[node_id] -= float(stage.storage_gb)

        feasible, reason = env.check_deployment_feasible(deployment)
        if not feasible:
            raise RuntimeError(f"UnifiedPPO produced infeasible deployment: {reason}")
        if record:
            self.deployment_replica_sum += float(deployment.sum())
            self.deployment_window_count += 1
        return deployment

    def schedule_batch(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        *,
        deterministic: bool,
        record: bool,
    ) -> tuple[list[list[int]], list[list[float]]]:
        env._require_ready()
        assert env.scenario is not None and env.deployment is not None
        demand = self._node_service_demand(env)
        remaining_memory, remaining_storage = self._remaining_resources(env, env.deployment)
        schedules: list[list[int]] = []
        resource_weights: list[list[float]] = []
        self.ledger = {}
        self.batch_total_count = max(sum(float(r.request_count) for r in requests), 1.0)
        self.batch_increment_cost = 0.0
        self.sampled_priorities = []
        total_requests = max(len(requests), 1)
        for request_index, request in enumerate(requests):
            nodes: list[int] = []
            weights: list[float] = []
            source_node = int(request.home_node)
            candidate_matrix = env.scheduler_candidate_mask(request)
            for stage_id in range(len(request.stage_compute_gcycles)):
                candidates = candidate_matrix[stage_id].copy()
                for node_id in np.flatnonzero(candidates):
                    if env.shortest_path(source_node, int(node_id)) is None:
                        candidates[int(node_id)] = False
                if not candidates.any():
                    raise RuntimeError(
                        f"UnifiedPPO has no reachable deployment for service {request.service_id} stage {stage_id}"
                    )
                node_id, priorities = self._act(
                    env,
                    mode="schedule",
                    service_id=int(request.service_id),
                    stage_id=stage_id,
                    stage_count=len(request.stage_compute_gcycles),
                    replica_index=0,
                    deployment=env.deployment,
                    remaining_memory=remaining_memory,
                    remaining_storage=remaining_storage,
                    candidate_nodes=candidates,
                    source_node=source_node,
                    request=request,
                    batch_progress=float(request_index) / total_requests,
                    allow_stop=False,
                    demand=demand,
                    deterministic=deterministic,
                    record=record,
                )
                nodes.append(int(node_id))
                weights.append(tuple(float(p) for p in priorities))
                self.sampled_priorities.append(priorities)
                increment = self._add_stage_cost(env, request, stage_id, source_node, node_id, priorities)
                self.batch_increment_cost += increment
                if record:
                    self.action_increment_costs[len(self.action_modes) - 1] = increment / self.batch_total_count
                    self.priority_sum += priorities
                    self.priority_count += 1
                source_node = int(node_id)
            schedules.append(nodes)
            resource_weights.append(weights)
        self.last_resource_weights = resource_weights
        self.sampled_priorities = np.asarray(self.sampled_priorities).reshape(-1, 2)
        return schedules, resource_weights

    def _ledger_increment(self, key: tuple, count: float, work: float, priority: float, capacity: float) -> float:
        if work <= 0.0:
            return 0.0
        p, q = self.ledger.get(key, (0.0, 0.0))
        new_p, new_q = p + count * priority, q + count * work / priority
        self.ledger[key] = (new_p, new_q)
        return (new_p * new_q - p * q) / max(capacity, 1e-9)

    def _add_stage_cost(self, env, request, stage, source, node, priorities) -> float:
        """Exact increment of weighted serial learned-share latency (P*Q/C)."""
        count = float(request.request_count)
        capacity = env.scenario.nodes[node].compute_gcycles_per_s * np.clip(
            1 - 0.75 * env.node_compute_load[node], 0.1, 1.0)
        cost = self._ledger_increment(("compute", stage, node), count,
            request.stage_compute_gcycles[stage], float(priorities[0]), capacity)
        data = request.input_mb if stage == 0 else request.stage_output_mb[stage - 1]
        path = env.shortest_path(source, node)
        for a, b in zip(path, path[1:]):
            capacity = env.scenario.bandwidth_mb_s[a, b] * np.clip(1 - 0.75 * env.link_load[a, b], 0.1, 1.0)
            cost += self._ledger_increment(("link", stage, a, b), count, data, float(priorities[1]), capacity)
            cost += count * env.scenario.propagation_ms[a, b] / 1000.0
        if stage == 0:
            cost += count * (env.config.radio_rtt_ms / 1000.0 +
                            request.input_mb / max(env.config.wireless_uplink_mbps / 8.0, 1e-9))
        return float(cost)

    def _node_service_demand(self, env: EdgeComputingEnv) -> np.ndarray:
        assert env.scenario is not None
        if getattr(self, "_demand_scenario", None) is env.scenario:
            return self._demand_cache
        demand = np.zeros((self.num_nodes, self.num_service_types), dtype=np.float64)
        for user in env.scenario.users:
            demand[int(user.home_node)] += np.asarray(user.service_weights, dtype=np.float64)
        demand /= max(float(len(env.scenario.users)), 1.0)
        self._demand_scenario = env.scenario
        self._demand_cache = demand
        return demand

    def _remaining_resources(
        self,
        env: EdgeComputingEnv,
        deployment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert env.scenario is not None
        memory = env.service_memory_capacities().copy()
        storage = env.service_storage_capacities().copy()
        for service in env.scenario.services:
            for stage in service.stages:
                selected = deployment[service.service_id, stage.stage_id]
                memory -= selected * float(stage.memory_gb)
                storage -= selected * float(stage.storage_gb)
        return memory, storage

    def _build_state(
        self,
        env: EdgeComputingEnv,
        *,
        mode: str,
        service_id: int,
        stage_id: int,
        stage_count: int,
        replica_index: int,
        deployment: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        candidate_nodes: np.ndarray,
        source_node: int | None,
        request: TaskRequest | None,
        batch_progress: float,
        demand: np.ndarray,
    ) -> np.ndarray:
        assert env.scenario is not None
        mode_one_hot = np.zeros(3, dtype=np.float32)
        mode_one_hot[UNIFIED_DECISION_MODES[mode]] = 1.0
        max_compute = max(float(node.compute_gcycles_per_s) for node in env.scenario.nodes)
        max_memory = np.maximum(env.service_memory_capacities(), 1e-9)
        max_storage = np.maximum(env.service_storage_capacities(), 1e-9)
        finite_links = env.scenario.bandwidth_mb_s[np.isfinite(env.scenario.bandwidth_mb_s)]
        max_bandwidth = max(float(np.max(finite_links)) if finite_links.size else 1.0, 1e-9)
        max_stage_compute = max(
            float(stage.compute_gcycles_mean) * float(env.config.task_compute_scale)
            for service in env.scenario.services
            for stage in service.stages
        )
        max_data = max(
            [
                float(service.input_mb_mean) * float(env.config.task_data_scale)
                for service in env.scenario.services
            ]
            + [
                float(stage.output_mb_mean) * float(env.config.task_data_scale)
                for service in env.scenario.services
                for stage in service.stages
            ]
            + [1e-9]
        )
        max_deadline = max(float(service.deadline_s_mean) for service in env.scenario.services)
        if request is None:
            request_count = 0.0
            stage_compute = float(
                env.scenario.services[service_id].stages[stage_id].compute_gcycles_mean
                * env.config.task_compute_scale
            )
            data_mb = float(
                env.scenario.services[service_id].input_mb_mean
                if stage_id == 0
                else env.scenario.services[service_id].stages[stage_id - 1].output_mb_mean
            ) * float(env.config.task_data_scale)
            deadline = float(env.scenario.services[service_id].deadline_s_mean)
        else:
            request_count = float(request.request_count)
            stage_compute = float(request.stage_compute_gcycles[stage_id])
            data_mb = float(request.input_mb if stage_id == 0 else request.stage_output_mb[stage_id - 1])
            deadline = float(request.deadline_s)
        finite_loads = env.link_load[np.isfinite(env.link_load)]
        global_features = np.asarray(
            [
                *mode_one_hot,
                float(env.current_time_minute / max(env.config.episode_minutes, 1)),
                float(service_id / max(self.num_service_types - 1, 1)),
                float(stage_id / max(self.max_service_stages - 1, 1)),
                float(stage_count / max(self.max_service_stages, 1)),
                float(replica_index / max(self.num_nodes - 1, 1)),
                float(np.log1p(request_count) / 8.0),
                float(stage_compute / max(max_stage_compute, 1e-9)),
                float(data_mb / max(max_data, 1e-9)),
                float(deadline / max(max_deadline, 1e-9)),
                float(-1.0 if source_node is None else source_node / max(self.num_nodes - 1, 1)),
                float(np.mean(deployment)),
                float(np.mean(env.node_compute_load)),
                float(np.max(env.node_compute_load)),
                float(np.mean(finite_loads)) if finite_loads.size else 0.0,
                float(np.max(finite_loads)) if finite_loads.size else 0.0,
                float(np.sum(np.maximum(remaining_memory, 0.0)) / np.sum(max_memory)),
                float(np.sum(np.maximum(remaining_storage, 0.0)) / np.sum(max_storage)),
                float(np.clip(batch_progress, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        if len(global_features) != UNIFIED_GLOBAL_DIM:
            raise RuntimeError("UnifiedPPO global feature layout changed")

        service_demand = demand[:, service_id]
        demand_max = max(float(np.max(service_demand)), 1e-9)
        node_features: list[list[float]] = []
        for node in env.scenario.nodes:
            node_id = int(node.node_id)
            reachable = 1.0
            propagation_s = 0.0
            bottleneck = max_bandwidth
            hop_fraction = 0.0
            if source_node is not None:
                path = env.shortest_path(int(source_node), node_id)
                if path is None:
                    reachable = 0.0
                    bottleneck = 0.0
                    hop_fraction = 1.0
                elif len(path) > 1:
                    edges = list(zip(path, path[1:]))
                    propagation_s = sum(
                        float(env.scenario.propagation_ms[a, b]) / 1000.0
                        for a, b in edges
                    )
                    bottleneck = min(
                        float(env.scenario.bandwidth_mb_s[a, b])
                        * max(1.0 - 0.75 * float(env.link_load[a, b]), 0.10)
                        for a, b in edges
                    )
                    hop_fraction = len(edges) / max(self.num_nodes - 1, 1)
            node_features.append(
                [
                    float(node.compute_gcycles_per_s / max_compute),
                    float(
                        node.compute_gcycles_per_s
                        * max(1.0 - 0.75 * float(env.node_compute_load[node_id]), 0.10)
                        / max_compute
                    ),
                    float(env.node_compute_load[node_id]),
                    float(max(remaining_memory[node_id], 0.0) / max_memory[node_id]),
                    float(max(remaining_storage[node_id], 0.0) / max_storage[node_id]),
                    float(deployment[service_id, stage_id, node_id]),
                    float(np.any(deployment[service_id, :, node_id])),
                    float(source_node is not None and node_id == source_node),
                    float(service_demand[node_id] / demand_max),
                    reachable,
                    float(np.tanh(propagation_s / 0.05)),
                    float(bottleneck / max_bandwidth),
                    float(hop_fraction),
                    float(candidate_nodes[node_id]),
                    float(node_id / max(self.num_nodes - 1, 1)),
                    float(np.mean(deployment[:, :, node_id])),
                    *self._ledger_features(env, stage_id, source_node, node_id, stage_compute, data_mb),
                ]
            )
        bandwidth = np.where(np.isfinite(env.scenario.bandwidth_mb_s), env.scenario.bandwidth_mb_s, 0.) / max_bandwidth
        propagation = np.nan_to_num(env.scenario.propagation_ms, nan=0., posinf=0.) / 100.
        chain = np.zeros((2, self.max_service_stages))
        if request is not None:
            chain[0, :stage_count] = np.asarray(request.stage_compute_gcycles) / max(max_stage_compute, 1e-9)
            chain[1, :stage_count] = np.asarray(request.stage_output_mb) / max_data
        state = np.concatenate([global_features, np.asarray(node_features).reshape(-1),
                                deployment.reshape(-1), bandwidth.reshape(-1), propagation.reshape(-1),
                                np.nan_to_num(env.link_load).reshape(-1), demand.reshape(-1), chain.reshape(-1)]).astype(np.float32)
        expected = unified_obs_dim(self.num_nodes, self.num_service_types, self.max_service_stages)
        if state.shape != (expected,):
            raise RuntimeError(f"UnifiedPPO state shape {state.shape} != {(expected,)}")
        return state

    def _ledger_features(self, env, stage, source, node, compute, data):
        p, q = self.ledger.get(("compute", stage, node), (0.0, 0.0))
        capacity = env.scenario.nodes[node].compute_gcycles_per_s * np.clip(1 - 0.75 * env.node_compute_load[node], 0.1, 1.0)
        path = env.shortest_path(source, node) if source is not None else [node]
        link_p = link_q = link_cost = 0.0
        if path is not None:
            for a, b in zip(path, path[1:]):
                lp, lq = self.ledger.get(("link", stage, a, b), (0.0, 0.0))
                lc = env.scenario.bandwidth_mb_s[a, b] * np.clip(1 - 0.75 * env.link_load[a, b], 0.1, 1.0)
                link_p += lp
                link_q += lq
                link_cost += (lp * data + lq) / max(lc, 1e-9)
        return [float(np.log1p(p) / 8), float(np.log1p(q) / 8),
                float(np.log1p((p * compute + q) / max(capacity, 1e-9))),
                float(np.log1p(link_p + link_q) / 8), float(np.log1p(link_cost))]
