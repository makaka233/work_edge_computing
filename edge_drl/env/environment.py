from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from edge_drl.allocators.kkt import ComputeDemand, LinkDemand, allocate_compute_kkt, allocate_link_kkt
from edge_drl.env.scenario import EdgeScenario, TaskRequest, generate_grouped_request, generate_realistic_scenario, generate_request


@dataclass
class EdgeEnvConfig:
    seed: int = 2026
    physical_seed: int | None = None
    scenario_seed: int | None = None
    num_users: int = 12_000
    num_edge_nodes: int = 48
    num_service_types: int = 5
    max_service_stages: int = 3
    episode_hours: int = 24
    deployment_interval_minutes: int = 240
    mean_requests_per_minute: float | None = None
    active_user_ratio: float = 0.15
    active_user_request_rate_per_minute: float = 1.5
    traffic_scale: float = 1.0
    task_compute_scale: float = 1.0
    task_data_scale: float = 1.0
    node_compute_capacity_scale: float = 1.0
    wired_link_bandwidth_scale: float = 1.0
    request_aggregation_window_seconds: float = 10.0
    max_representative_groups_per_window: int | None = 16
    load_ewma_tau_minutes: float = 1.0
    wireless_uplink_mbps: float = 150.0
    radio_rtt_ms: float = 10.0
    load_penalty_weight: float = 0.08
    migration_cost_weight: float = 0.02
    invalid_action_penalty: float = 10.0

    def __post_init__(self) -> None:
        if not 10_000 <= self.num_users <= 15_000:
            raise ValueError("num_users must be in [10000, 15000].")
        if self.max_service_stages > 3:
            raise ValueError("max_service_stages must be <= 3.")
        if self.deployment_interval_minutes != 240:
            raise ValueError("slow deployment interval must be exactly 240 minutes.")
        if self.mean_requests_per_minute is not None and self.mean_requests_per_minute <= 0:
            raise ValueError("mean_requests_per_minute override must be positive.")
        if not 0.0 < self.active_user_ratio <= 1.0:
            raise ValueError("active_user_ratio must be in (0, 1].")
        if self.active_user_request_rate_per_minute <= 0:
            raise ValueError("active_user_request_rate_per_minute must be positive.")
        if self.traffic_scale <= 0:
            raise ValueError("traffic_scale must be positive.")
        if self.task_compute_scale <= 0:
            raise ValueError("task_compute_scale must be positive.")
        if self.task_data_scale <= 0:
            raise ValueError("task_data_scale must be positive.")
        if self.node_compute_capacity_scale <= 0:
            raise ValueError("node_compute_capacity_scale must be positive.")
        if self.wired_link_bandwidth_scale <= 0:
            raise ValueError("wired_link_bandwidth_scale must be positive.")
        if self.request_aggregation_window_seconds < 0:
            raise ValueError("request_aggregation_window_seconds must be non-negative.")
        if self.max_representative_groups_per_window is not None and self.max_representative_groups_per_window <= 0:
            raise ValueError("max_representative_groups_per_window must be positive or None.")
        if self.load_ewma_tau_minutes <= 0:
            raise ValueError("load_ewma_tau_minutes must be positive.")
        if self.wireless_uplink_mbps <= 0:
            raise ValueError("wireless_uplink_mbps must be positive.")
        if self.radio_rtt_ms < 0:
            raise ValueError("radio_rtt_ms must be non-negative.")


class EdgeComputingEnv:
    """Event-driven edge-computing environment.

    The design follows DRL-AC-Allocation's separation of environment state,
    candidate actions, masks, and training entry points, while adapting the
    problem to staged edge services and two time scales.
    """

    def __init__(self, config: EdgeEnvConfig | None = None):
        self.config = config or EdgeEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.scenario: EdgeScenario | None = None
        self.deployment: np.ndarray | None = None
        self.current_time_minute = 0.0
        self.next_deployment_update_minute = 0.0
        self.request_counter = 0
        self.current_request: TaskRequest | None = None
        self.pending_requests: list[TaskRequest] = []
        self.request_group_probabilities: np.ndarray | None = None
        self.node_compute_load = np.zeros(self.config.num_edge_nodes, dtype=np.float64)
        self.link_load = np.zeros((self.config.num_edge_nodes, self.config.num_edge_nodes), dtype=np.float64)
        self.last_load_update_minute = 0.0
        self.last_migration_cost = 0.0
        self.metrics: dict[str, float] = {}

    def reset(self) -> dict[str, Any]:
        self.rng = np.random.default_rng(self.config.seed)
        physical_rng = np.random.default_rng(
            self.config.seed if self.config.physical_seed is None else self.config.physical_seed
        )
        demand_rng = np.random.default_rng(
            self.config.seed if self.config.scenario_seed is None else self.config.scenario_seed
        )
        self.scenario = generate_realistic_scenario(
            rng=physical_rng,
            demand_rng=demand_rng,
            num_users=self.config.num_users,
            num_edge_nodes=self.config.num_edge_nodes,
            num_service_types=self.config.num_service_types,
            max_service_stages=self.config.max_service_stages,
            node_compute_capacity_scale=self.config.node_compute_capacity_scale,
            wired_link_bandwidth_scale=self.config.wired_link_bandwidth_scale,
        )
        self.deployment = np.zeros(
            (self.config.num_service_types, self.config.max_service_stages, self.config.num_edge_nodes),
            dtype=bool,
        )
        self.current_time_minute = 0.0
        self.next_deployment_update_minute = 0.0
        self.request_counter = 0
        self.pending_requests = []
        self.request_group_probabilities = self._build_request_group_probabilities()
        self.node_compute_load = np.zeros(self.config.num_edge_nodes, dtype=np.float64)
        self.link_load = np.zeros((self.config.num_edge_nodes, self.config.num_edge_nodes), dtype=np.float64)
        self.last_load_update_minute = 0.0
        self.last_migration_cost = 0.0
        self.metrics = {
            "requests": 0.0,
            "aggregate_events": 0.0,
            "invalid_actions": 0.0,
            "total_latency_s": 0.0,
            "valid_requests": 0.0,
            "total_valid_latency_s": 0.0,
            "total_penalty_latency_s": 0.0,
            "deadline_violations": 0.0,
            "deployment_updates": 0.0,
        }
        self.current_request = self._next_request()
        self.last_load_update_minute = self.current_time_minute
        return self.observe()

    @property
    def done(self) -> bool:
        return self.current_time_minute >= self.config.episode_hours * 60

    @property
    def needs_deployment_update(self) -> bool:
        return self.current_time_minute >= self.next_deployment_update_minute

    def observe(self) -> dict[str, Any]:
        self._require_ready()
        assert self.scenario is not None
        assert self.deployment is not None
        assert self.current_request is not None

        nodes = np.array(
            [
                [
                    n.memory_gb,
                    n.storage_gb,
                    n.compute_gcycles_per_s,
                    self.node_compute_load[n.node_id],
                    n.x_km,
                    n.y_km,
                ]
                for n in self.scenario.nodes
            ],
            dtype=np.float32,
        )
        request = self.current_request
        request_features = np.array(
            [
                request.home_node,
                request.service_id,
                request.request_count,
                request.input_mb,
                len(request.stage_compute_gcycles),
                request.deadline_s,
                self.current_time_minute % (24 * 60),
            ],
            dtype=np.float32,
        )
        return {
            "time_minute": self.current_time_minute,
            "needs_deployment_update": self.needs_deployment_update,
            "nodes": nodes,
            "deployment": self.deployment.copy(),
            "node_compute_load": self.node_compute_load.copy(),
            "link_load": self.link_load.copy(),
            "request": request,
            "request_features": request_features,
            "candidate_mask": self.scheduler_candidate_mask(request),
        }

    def apply_deployment(self, deployment: np.ndarray) -> float:
        self._require_ready()
        assert self.scenario is not None
        assert self.deployment is not None

        deployment = np.asarray(deployment, dtype=bool)
        if deployment.shape != self.deployment.shape:
            raise ValueError(f"deployment shape must be {self.deployment.shape}")
        feasible, reason = self.check_deployment_feasible(deployment)
        if not feasible:
            raise ValueError(reason)

        changed = np.logical_xor(self.deployment, deployment)
        migration_cost = float(changed.sum())
        self.deployment = deployment.copy()
        self.next_deployment_update_minute = self.current_time_minute + self.config.deployment_interval_minutes
        self.last_migration_cost = migration_cost
        self.metrics["deployment_updates"] += 1.0
        return migration_cost

    def step(self, stage_nodes: list[int] | tuple[int, ...] | np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._require_ready()
        assert self.current_request is not None

        info = self.evaluate_schedule(self.current_request, stage_nodes)
        request_count = float(self.current_request.request_count)
        migration_cost = self.last_migration_cost
        migration_penalty = self.config.migration_cost_weight * migration_cost
        self.last_migration_cost = 0.0

        reward = -info["latency_s"]
        reward -= self.config.load_penalty_weight * info["load_penalty"]
        reward -= migration_penalty
        if not info["valid"]:
            reward -= self.config.invalid_action_penalty
            self.metrics["invalid_actions"] += request_count

        self.metrics["requests"] += request_count
        self.metrics["aggregate_events"] += 1.0
        self.metrics["total_latency_s"] += float(info["latency_s"]) * request_count
        self.metrics["total_penalty_latency_s"] += float(info["penalty_latency_s"]) * request_count
        if info["valid"]:
            self.metrics["valid_requests"] += request_count
            self.metrics["total_valid_latency_s"] += float(info["physical_latency_s"]) * request_count
        if info["latency_s"] > self.current_request.deadline_s:
            self.metrics["deadline_violations"] += request_count

        self._update_dynamic_loads(info, request_count=request_count)
        info["migration_cost"] = migration_cost
        info["migration_penalty"] = migration_penalty
        info["request_count"] = request_count
        self.current_request = self._next_request()
        return self.observe(), float(reward), self.done, info

    def scheduler_candidate_mask(self, request: TaskRequest) -> np.ndarray:
        self._require_ready()
        assert self.deployment is not None
        mask = np.zeros((len(request.stage_compute_gcycles), self.config.num_edge_nodes), dtype=bool)
        for stage_id in range(len(request.stage_compute_gcycles)):
            mask[stage_id] = self.deployment[request.service_id, stage_id]
        return mask

    def check_deployment_feasible(self, deployment: np.ndarray) -> tuple[bool, str]:
        self._require_ready()
        assert self.scenario is not None

        memory = np.zeros(self.config.num_edge_nodes, dtype=np.float64)
        storage = np.zeros(self.config.num_edge_nodes, dtype=np.float64)
        for service in self.scenario.services:
            for stage in service.stages:
                placed = deployment[service.service_id, stage.stage_id]
                memory += placed * stage.memory_gb
                storage += placed * stage.storage_gb
        node_memory = np.array([n.memory_gb for n in self.scenario.nodes])
        node_storage = np.array([n.storage_gb for n in self.scenario.nodes])
        if np.any(memory > node_memory + 1e-9):
            return False, "deployment exceeds node memory"
        if np.any(storage > node_storage + 1e-9):
            return False, "deployment exceeds node storage"

        for service in self.scenario.services:
            for stage in service.stages:
                if not deployment[service.service_id, stage.stage_id].any():
                    return False, f"service {service.service_id} stage {stage.stage_id} has no replica"
        return True, "ok"

    def evaluate_schedule(self, request: TaskRequest, stage_nodes: list[int] | tuple[int, ...] | np.ndarray) -> dict[str, Any]:
        self._require_ready()
        assert self.scenario is not None
        assert self.deployment is not None

        nodes = [int(v) for v in stage_nodes]
        valid = True
        violations: list[str] = []
        if len(nodes) != len(request.stage_compute_gcycles):
            valid = False
            violations.append("stage node count mismatch")
            nodes = (nodes + [request.home_node] * len(request.stage_compute_gcycles))[: len(request.stage_compute_gcycles)]

        for stage_id, node_id in enumerate(nodes):
            if not 0 <= node_id < self.config.num_edge_nodes:
                valid = False
                violations.append(f"node {node_id} out of range")
                nodes[stage_id] = request.home_node
            elif not self.deployment[request.service_id, stage_id, node_id]:
                valid = False
                violations.append(f"service stage {stage_id} is not deployed on node {node_id}")

        link_demands: list[LinkDemand] = []
        if nodes[0] != request.home_node:
            link_demands.append(
                LinkDemand("ingress", request.home_node, nodes[0], request.input_mb)
            )
        for stage_id in range(len(nodes) - 1):
            if nodes[stage_id] != nodes[stage_id + 1]:
                link_demands.append(
                    LinkDemand(
                        f"stage-{stage_id}",
                        nodes[stage_id],
                        nodes[stage_id + 1],
                        request.stage_output_mb[stage_id],
                    )
                )

        for demand in link_demands:
            if not self.scenario.adjacency[demand.src_node, demand.dst_node]:
                valid = False
                violations.append(f"link {demand.src_node}->{demand.dst_node} is unavailable")

        compute_demands = [
            ComputeDemand(
                f"stage-{stage_id}",
                node_id,
                request.stage_compute_gcycles[stage_id],
            )
            for stage_id, node_id in enumerate(nodes)
        ]
        node_capacity = np.array([n.compute_gcycles_per_s for n in self.scenario.nodes], dtype=np.float64)
        node_capacity *= np.clip(1.0 - 0.75 * self.node_compute_load, 0.10, 1.0)
        link_capacity = self.scenario.bandwidth_mb_s.copy()
        finite = np.isfinite(link_capacity)
        link_capacity[finite] *= np.clip(1.0 - 0.75 * self.link_load[finite], 0.10, 1.0)

        _, compute_delays, compute_delay = allocate_compute_kkt(compute_demands, node_capacity)
        allocation_penalty_s = 0.0
        try:
            _, link_delays, link_delay = allocate_link_kkt(link_demands, link_capacity)
        except ValueError:
            valid = False
            link_delays = {}
            link_delay = 0.0
            allocation_penalty_s = self.config.invalid_action_penalty

        access_delay = self.config.radio_rtt_ms / 1000.0
        access_delay += request.input_mb / max(self.config.wireless_uplink_mbps / 8.0, 1e-9)
        propagation_delay = 0.0
        for demand in link_demands:
            if self.scenario.adjacency[demand.src_node, demand.dst_node]:
                propagation_delay += float(self.scenario.propagation_ms[demand.src_node, demand.dst_node]) / 1000.0

        physical_latency_s = access_delay + compute_delay + link_delay + propagation_delay
        penalty_latency_s = allocation_penalty_s
        if not valid:
            penalty_latency_s += self.config.invalid_action_penalty
        latency_s = physical_latency_s + penalty_latency_s

        load_penalty = float(
            sum(self.node_compute_load[node_id] for node_id in nodes)
            + sum(self.link_load[d.src_node, d.dst_node] for d in link_demands if self.scenario.adjacency[d.src_node, d.dst_node])
        )
        return {
            "valid": valid,
            "violations": violations,
            "stage_nodes": nodes,
            "compute_delay_s": compute_delay,
            "link_delay_s": link_delay,
            "access_delay_s": access_delay,
            "propagation_delay_s": propagation_delay,
            "physical_latency_s": physical_latency_s,
            "penalty_latency_s": penalty_latency_s,
            "latency_s": latency_s,
            "compute_delays": compute_delays,
            "link_delays": link_delays,
            "compute_demands": compute_demands,
            "link_demands": link_demands,
            "load_penalty": load_penalty,
        }

    def _next_request(self) -> TaskRequest:
        if self.pending_requests:
            return self.pending_requests.pop(0)

        aggregation_window_minutes = self.config.request_aggregation_window_seconds / 60.0
        if aggregation_window_minutes > 0.0:
            while not self.done:
                expected_requests = self._arrival_rate_per_minute() * aggregation_window_minutes
                self.current_time_minute += aggregation_window_minutes
                total_count = int(self.rng.poisson(expected_requests))
                if total_count <= 0:
                    continue
                self.pending_requests = self._generate_aggregated_requests(total_count)
                if self.pending_requests:
                    return self.pending_requests.pop(0)
            self.pending_requests = self._generate_aggregated_requests(1)
            return self.pending_requests.pop(0)

        interarrival = self.rng.exponential(1.0 / max(self._arrival_rate_per_minute(), 1e-6))
        self.current_time_minute += interarrival
        request = generate_request(
            rng=self.rng,
            request_id=self.request_counter,
            arrival_minute=self.current_time_minute,
            users=self.scenario.users if self.scenario else [],
            services=self.scenario.services if self.scenario else [],
            task_compute_scale=self.config.task_compute_scale,
            task_data_scale=self.config.task_data_scale,
        )
        self.request_counter += 1
        return request

    def _build_request_group_probabilities(self) -> np.ndarray:
        assert self.scenario is not None
        probabilities = np.zeros((self.config.num_edge_nodes, self.config.num_service_types), dtype=np.float64)
        for user in self.scenario.users:
            probabilities[user.home_node] += np.asarray(user.service_weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        return probabilities.reshape(-1)

    def _generate_aggregated_requests(self, total_count: int) -> list[TaskRequest]:
        assert self.scenario is not None
        assert self.request_group_probabilities is not None
        group_counts = self.rng.multinomial(total_count, self.request_group_probabilities)
        group_counts = self._sample_representative_group_counts(group_counts)
        requests: list[TaskRequest] = []
        for group_id, request_count in enumerate(group_counts):
            if request_count <= 0:
                continue
            home_node = group_id // self.config.num_service_types
            service_id = group_id % self.config.num_service_types
            requests.append(
                generate_grouped_request(
                    rng=self.rng,
                    request_id=self.request_counter,
                    arrival_minute=self.current_time_minute,
                    request_count=int(request_count),
                    user_id=-1,
                    home_node=int(home_node),
                    service_id=int(service_id),
                    services=self.scenario.services,
                    task_compute_scale=self.config.task_compute_scale,
                    task_data_scale=self.config.task_data_scale,
                )
            )
            self.request_counter += 1
        requests.sort(key=lambda request: (-request.request_count, request.home_node, request.service_id))
        return requests

    def _sample_representative_group_counts(self, group_counts: np.ndarray) -> np.ndarray:
        cap = self.config.max_representative_groups_per_window
        nonzero = np.flatnonzero(group_counts > 0)
        if cap is None or len(nonzero) <= cap:
            return group_counts

        nonzero_counts = group_counts[nonzero].astype(np.float64)
        order = np.argsort(-nonzero_counts)
        top_k = max(1, min(cap // 2, cap, len(nonzero)))
        top = nonzero[order[:top_k]]
        remaining = nonzero[order[top_k:]]
        remaining_slots = cap - len(top)

        if remaining_slots > 0 and len(remaining) > 0:
            remaining_counts = group_counts[remaining].astype(np.float64)
            probabilities = remaining_counts / remaining_counts.sum()
            sampled = self.rng.choice(
                remaining,
                size=min(remaining_slots, len(remaining)),
                replace=False,
                p=probabilities,
            )
            selected = np.concatenate([top, sampled])
        else:
            selected = top

        original_total = int(group_counts.sum())
        selected_total = float(group_counts[selected].sum())
        scaled = group_counts[selected].astype(np.float64) * (original_total / max(selected_total, 1.0))
        base = np.floor(scaled).astype(int)
        remainder = original_total - int(base.sum())
        if remainder > 0:
            fractional_order = np.argsort(-(scaled - base))
            for idx in fractional_order[:remainder]:
                base[idx] += 1

        sampled_counts = np.zeros_like(group_counts)
        sampled_counts[selected] = base
        return sampled_counts

    def _arrival_rate_per_minute(self) -> float:
        minute_of_day = self.current_time_minute % (24 * 60)
        morning_peak = np.exp(-0.5 * ((minute_of_day - 9 * 60) / 105.0) ** 2)
        lunch_peak = np.exp(-0.5 * ((minute_of_day - 13 * 60) / 90.0) ** 2)
        evening_peak = np.exp(-0.5 * ((minute_of_day - 19 * 60) / 135.0) ** 2)
        night_factor = 0.35 if minute_of_day < 6 * 60 else 1.0
        daily_factor = 0.58 + 0.58 * morning_peak + 0.25 * lunch_peak + 0.82 * evening_peak
        return self._base_arrival_rate_per_minute() * night_factor * daily_factor

    def _base_arrival_rate_per_minute(self) -> float:
        if self.config.mean_requests_per_minute is not None:
            return self.config.mean_requests_per_minute
        return (
            self.config.num_users
            * self.config.active_user_ratio
            * self.config.active_user_request_rate_per_minute
            * self.config.traffic_scale
        )

    def _update_dynamic_loads(self, info: dict[str, Any], *, request_count: float) -> None:
        elapsed_minutes = max(self.current_time_minute - self.last_load_update_minute, 0.0)
        decay = float(np.exp(-elapsed_minutes / self.config.load_ewma_tau_minutes))
        self.node_compute_load *= decay
        self.link_load *= decay
        self.last_load_update_minute = self.current_time_minute
        ewma_window_s = self.config.load_ewma_tau_minutes * 60.0
        for demand in info["compute_demands"]:
            node_capacity = self.scenario.nodes[demand.node_id].compute_gcycles_per_s if self.scenario else 1.0
            service_time_s = demand.compute_gcycles * request_count / max(node_capacity, 1e-9)
            increment = min(service_time_s / ewma_window_s, 1.0)
            self.node_compute_load[demand.node_id] = min(1.0, self.node_compute_load[demand.node_id] + increment)
        for demand in info["link_demands"]:
            if self.scenario is None or not self.scenario.adjacency[demand.src_node, demand.dst_node]:
                continue
            capacity = self.scenario.bandwidth_mb_s[demand.src_node, demand.dst_node]
            transfer_time_s = demand.data_mb * request_count / max(capacity, 1e-9)
            increment = min(transfer_time_s / ewma_window_s, 1.0)
            self.link_load[demand.src_node, demand.dst_node] = min(1.0, self.link_load[demand.src_node, demand.dst_node] + increment)

    def _require_ready(self) -> None:
        if self.scenario is None:
            raise RuntimeError("call reset() before using the environment")
