from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from edge_drl.allocators.kkt import ComputeDemand, LinkDemand, allocate_compute_kkt, allocate_link_kkt
from edge_drl.env.scenario import EdgeScenario, TaskRequest, generate_grouped_request, generate_realistic_scenario


def _daily_arrival_factor(minute_of_day: float) -> float:
    morning_peak = np.exp(-0.5 * ((minute_of_day - 9 * 60) / 105.0) ** 2)
    lunch_peak = np.exp(-0.5 * ((minute_of_day - 13 * 60) / 90.0) ** 2)
    evening_peak = np.exp(-0.5 * ((minute_of_day - 19 * 60) / 135.0) ** 2)
    night_factor = 0.35 if minute_of_day < 6 * 60 else 1.0
    return float(night_factor * (0.58 + 0.58 * morning_peak + 0.25 * lunch_peak + 0.82 * evening_peak))


# Keep stationary 4h episodes at the mean traffic level of the legacy daily profile.
_STATIONARY_ARRIVAL_FACTOR = float(np.mean([_daily_arrival_factor(float(minute)) for minute in range(24 * 60)]))


@dataclass
class EdgeEnvConfig:
    seed: int = 2026
    physical_seed: int | None = None
    scenario_seed: int | None = None
    num_users: int = 12_000
    num_edge_nodes: int = 48
    num_service_types: int = 5
    max_service_stages: int = 3
    episode_hours: int = 4
    deployment_interval_minutes: int = 10
    arrival_profile: str = "stationary"
    mean_requests_per_minute: float | None = None
    active_user_ratio: float = 0.15
    active_user_request_rate_per_minute: float = 1.5
    traffic_scale: float = 1.0
    demand_load_multiplier: float = 1.0
    task_compute_scale: float = 1.0
    task_data_scale: float = 1.0
    node_compute_capacity_scale: float = 1.0
    wired_link_bandwidth_scale: float = 1.0
    service_resource_fraction: float = 0.5
    request_aggregation_window_seconds: float = 1.0
    load_ewma_tau_minutes: float = 1.0
    wireless_uplink_mbps: float = 150.0
    radio_rtt_ms: float = 10.0
    load_penalty_weight: float = 0.08
    migration_cost_weight: float = 0.0
    invalid_action_penalty: float = 10.0

    def __post_init__(self) -> None:
        if not 10_000 <= self.num_users <= 15_000:
            raise ValueError("num_users must be in [10000, 15000].")
        if self.max_service_stages > 3:
            raise ValueError("max_service_stages must be <= 3.")
        if self.deployment_interval_minutes <= 0:
            raise ValueError("deployment_interval_minutes must be positive.")
        if self.episode_hours <= 0:
            raise ValueError("episode_hours must be positive.")
        if self.arrival_profile not in {"stationary", "daily"}:
            raise ValueError("arrival_profile must be 'stationary' or 'daily'.")
        if self.mean_requests_per_minute is not None and self.mean_requests_per_minute <= 0:
            raise ValueError("mean_requests_per_minute override must be positive.")
        if not 0.0 < self.active_user_ratio <= 1.0:
            raise ValueError("active_user_ratio must be in (0, 1].")
        if self.active_user_request_rate_per_minute <= 0:
            raise ValueError("active_user_request_rate_per_minute must be positive.")
        if self.traffic_scale <= 0:
            raise ValueError("traffic_scale must be positive.")
        if self.demand_load_multiplier <= 0:
            raise ValueError("demand_load_multiplier must be positive.")
        if self.task_compute_scale <= 0:
            raise ValueError("task_compute_scale must be positive.")
        if self.task_data_scale <= 0:
            raise ValueError("task_data_scale must be positive.")
        if self.node_compute_capacity_scale <= 0:
            raise ValueError("node_compute_capacity_scale must be positive.")
        if self.wired_link_bandwidth_scale <= 0:
            raise ValueError("wired_link_bandwidth_scale must be positive.")
        if not 0.0 < self.service_resource_fraction <= 1.0:
            raise ValueError("service_resource_fraction must be in (0, 1].")
        if not np.isclose(self.request_aggregation_window_seconds, 1.0):
            raise ValueError("request_aggregation_window_seconds must be exactly 1.0 so one env.step equals one second.")
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
        self.current_requests: list[TaskRequest] = []
        self.current_request: TaskRequest | None = None
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
        self.current_requests = []
        self.request_group_probabilities = self._build_request_group_probabilities()
        self.node_compute_load = np.zeros(self.config.num_edge_nodes, dtype=np.float64)
        self.link_load = np.zeros((self.config.num_edge_nodes, self.config.num_edge_nodes), dtype=np.float64)
        self.last_load_update_minute = 0.0
        self.last_migration_cost = 0.0
        self.metrics = {
            "requests": 0.0,
            "aggregate_events": 0.0,
            "time_steps": 0.0,
            "invalid_actions": 0.0,
            "total_latency_s": 0.0,
            "valid_requests": 0.0,
            "total_valid_latency_s": 0.0,
            "total_penalty_latency_s": 0.0,
            "deadline_violations": 0.0,
            "deployment_updates": 0.0,
        }
        self.current_requests = self._generate_current_second_requests()
        self.current_request = self.current_requests[0] if self.current_requests else None
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
        if request is None:
            request_features = np.zeros(7, dtype=np.float32)
        else:
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
            "requests": tuple(self.current_requests),
            "request_features": request_features,
            "candidate_mask": (
                self.scheduler_candidate_mask(request)
                if request is not None
                else np.zeros((0, self.config.num_edge_nodes), dtype=bool)
            ),
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

    def step(
        self,
        batch_stage_nodes: list[list[int] | tuple[int, ...] | np.ndarray],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Schedule and settle every aggregate request group in one simulated second."""

        self._require_ready()
        requests = list(self.current_requests)
        schedules = list(batch_stage_nodes)
        if len(schedules) != len(requests):
            raise ValueError(f"expected {len(requests)} group schedules, got {len(schedules)}")

        group_infos = self.evaluate_batch_schedules(requests, schedules)
        migration_cost = self.last_migration_cost
        migration_penalty = self.config.migration_cost_weight * migration_cost
        self.last_migration_cost = 0.0

        group_rewards: list[float] = []
        counts = np.asarray([request.request_count for request in requests], dtype=np.float64)
        for request, group_info in zip(requests, group_infos):
            request_count = float(request.request_count)
            reward = -float(group_info["latency_s"])
            reward -= self.config.load_penalty_weight * float(group_info["load_penalty"])
            if not group_info["valid"]:
                reward -= self.config.invalid_action_penalty
                self.metrics["invalid_actions"] += request_count
            group_info["reward"] = float(reward)
            group_info["request_count"] = request_count
            group_rewards.append(float(reward))

            self.metrics["requests"] += request_count
            self.metrics["total_latency_s"] += float(group_info["latency_s"]) * request_count
            self.metrics["total_penalty_latency_s"] += float(group_info["penalty_latency_s"]) * request_count
            if group_info["valid"]:
                self.metrics["valid_requests"] += request_count
                self.metrics["total_valid_latency_s"] += float(group_info["physical_latency_s"]) * request_count
            if float(group_info["latency_s"]) > request.deadline_s:
                self.metrics["deadline_violations"] += request_count

        self.metrics["aggregate_events"] += float(len(requests))
        self.metrics["time_steps"] += 1.0
        self._update_dynamic_loads_batch(group_infos, requests)

        total_count = float(counts.sum())
        weighted_reward = self._weighted_group_mean(group_rewards, counts)
        if total_count > 0.0:
            weighted_reward -= migration_penalty
        self.current_time_minute += self.config.request_aggregation_window_seconds / 60.0
        if self.done:
            self.current_requests = []
        else:
            self.current_requests = self._generate_current_second_requests()
        self.current_request = self.current_requests[0] if self.current_requests else None

        info = self._summarize_batch_info(group_infos, counts)
        info.update(
            {
                "group_infos": group_infos,
                "group_count": len(group_infos),
                "request_count": total_count,
                "migration_cost": migration_cost,
                "migration_penalty": migration_penalty,
            }
        )
        return self.observe(), float(weighted_reward), self.done, info

    def scheduler_candidate_mask(self, request: TaskRequest) -> np.ndarray:
        self._require_ready()
        assert self.deployment is not None
        mask = np.zeros((len(request.stage_compute_gcycles), self.config.num_edge_nodes), dtype=bool)
        for stage_id in range(len(request.stage_compute_gcycles)):
            mask[stage_id] = self.deployment[request.service_id, stage_id]
        return mask

    def service_memory_capacities(self) -> np.ndarray:
        self._require_ready()
        assert self.scenario is not None
        return np.asarray(
            [node.memory_gb * self.config.service_resource_fraction for node in self.scenario.nodes],
            dtype=np.float64,
        )

    def service_storage_capacities(self) -> np.ndarray:
        self._require_ready()
        assert self.scenario is not None
        return np.asarray(
            [node.storage_gb * self.config.service_resource_fraction for node in self.scenario.nodes],
            dtype=np.float64,
        )

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
        node_memory = self.service_memory_capacities()
        node_storage = self.service_storage_capacities()
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
        """Evaluate one group in isolation, mainly for greedy action search."""

        return self.evaluate_batch_schedules([request], [stage_nodes])[0]

    def evaluate_batch_schedules(
        self,
        requests: list[TaskRequest],
        batch_stage_nodes: list[list[int] | tuple[int, ...] | np.ndarray],
    ) -> list[dict[str, Any]]:
        """Jointly allocate compute and links for all request groups in one second."""

        self._require_ready()
        assert self.scenario is not None
        assert self.deployment is not None
        if len(requests) != len(batch_stage_nodes):
            raise ValueError("requests and batch_stage_nodes must have the same length")

        prepared: list[dict[str, Any]] = []
        all_compute_demands: list[ComputeDemand] = []
        all_link_demands: list[LinkDemand] = []
        for group_idx, (request, stage_nodes) in enumerate(zip(requests, batch_stage_nodes)):
            group = self._prepare_group_demands(group_idx, request, stage_nodes)
            prepared.append(group)
            all_compute_demands.extend(group["joint_compute_demands"])
            all_link_demands.extend(group["joint_link_demands"])

        node_capacity = np.array([n.compute_gcycles_per_s for n in self.scenario.nodes], dtype=np.float64)
        node_capacity *= np.clip(1.0 - 0.75 * self.node_compute_load, 0.10, 1.0)
        link_capacity = self.scenario.bandwidth_mb_s.copy()
        finite = np.isfinite(link_capacity)
        link_capacity[finite] *= np.clip(1.0 - 0.75 * self.link_load[finite], 0.10, 1.0)

        _, joint_compute_delays, _ = allocate_compute_kkt(all_compute_demands, node_capacity)
        link_allocation_failed = False
        try:
            _, joint_link_delays, _ = allocate_link_kkt(all_link_demands, link_capacity)
        except ValueError:
            joint_link_delays = {}
            link_allocation_failed = True

        infos: list[dict[str, Any]] = []
        for group in prepared:
            valid = bool(group["valid"]) and not link_allocation_failed
            violations = list(group["violations"])
            if link_allocation_failed:
                violations.append("joint link allocation failed")
            compute_delays = {
                demand.demand_id.split(":", 1)[1]: joint_compute_delays[demand.demand_id]
                for demand in group["joint_compute_demands"]
            }
            link_delays = {
                demand.demand_id.split(":", 1)[1]: joint_link_delays.get(demand.demand_id, 0.0)
                for demand in group["joint_link_demands"]
            }
            compute_delay = float(sum(compute_delays.values()))
            link_delay = float(sum(link_delays.values()))
            penalty_latency_s = 0.0 if valid else self.config.invalid_action_penalty
            physical_latency_s = (
                float(group["access_delay_s"])
                + compute_delay
                + link_delay
                + float(group["propagation_delay_s"])
            )
            infos.append(
                {
                    "valid": valid,
                    "violations": violations,
                    "stage_nodes": group["stage_nodes"],
                    "compute_delay_s": compute_delay,
                    "link_delay_s": link_delay,
                    "access_delay_s": float(group["access_delay_s"]),
                    "propagation_delay_s": float(group["propagation_delay_s"]),
                    "physical_latency_s": physical_latency_s,
                    "penalty_latency_s": penalty_latency_s,
                    "latency_s": physical_latency_s + penalty_latency_s,
                    "compute_delays": compute_delays,
                    "link_delays": link_delays,
                    "compute_demands": group["compute_demands"],
                    "link_demands": group["link_demands"],
                    "load_penalty": float(group["load_penalty"]),
                }
            )
        return infos

    def _prepare_group_demands(
        self,
        group_idx: int,
        request: TaskRequest,
        stage_nodes: list[int] | tuple[int, ...] | np.ndarray,
    ) -> dict[str, Any]:
        assert self.scenario is not None
        assert self.deployment is not None
        nodes = [int(v) for v in stage_nodes]
        valid = True
        violations: list[str] = []
        stage_count = len(request.stage_compute_gcycles)
        if len(nodes) != stage_count:
            valid = False
            violations.append("stage node count mismatch")
            nodes = (nodes + [request.home_node] * stage_count)[:stage_count]

        for stage_id, node_id in enumerate(nodes):
            if not 0 <= node_id < self.config.num_edge_nodes:
                valid = False
                violations.append(f"node {node_id} out of range")
                nodes[stage_id] = request.home_node
            elif not self.deployment[request.service_id, stage_id, node_id]:
                valid = False
                violations.append(f"service stage {stage_id} is not deployed on node {node_id}")

        compute_demands = [
            ComputeDemand(f"stage-{stage_id}", node_id, request.stage_compute_gcycles[stage_id])
            for stage_id, node_id in enumerate(nodes)
        ]
        link_demands: list[LinkDemand] = []
        if nodes and nodes[0] != request.home_node:
            link_demands.append(LinkDemand("ingress", request.home_node, nodes[0], request.input_mb))
        for stage_id in range(max(len(nodes) - 1, 0)):
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

        count = float(request.request_count)
        joint_compute_demands = [
            ComputeDemand(
                f"g{group_idx}:{demand.demand_id}",
                demand.node_id,
                demand.compute_gcycles,
                multiplicity=count,
            )
            for demand in compute_demands
        ]
        joint_link_demands = [
            LinkDemand(
                f"g{group_idx}:{demand.demand_id}",
                demand.src_node,
                demand.dst_node,
                demand.data_mb,
                multiplicity=count,
            )
            for demand in link_demands
            if self.scenario.adjacency[demand.src_node, demand.dst_node]
        ]
        access_delay = self.config.radio_rtt_ms / 1000.0
        access_delay += request.input_mb / max(self.config.wireless_uplink_mbps / 8.0, 1e-9)
        propagation_delay = float(
            sum(
                self.scenario.propagation_ms[demand.src_node, demand.dst_node] / 1000.0
                for demand in link_demands
                if self.scenario.adjacency[demand.src_node, demand.dst_node]
            )
        )
        load_penalty = float(
            sum(self.node_compute_load[node_id] for node_id in nodes)
            + sum(
                self.link_load[demand.src_node, demand.dst_node]
                for demand in link_demands
                if self.scenario.adjacency[demand.src_node, demand.dst_node]
            )
        )
        return {
            "valid": valid,
            "violations": violations,
            "stage_nodes": nodes,
            "compute_demands": compute_demands,
            "link_demands": link_demands,
            "joint_compute_demands": joint_compute_demands,
            "joint_link_demands": joint_link_demands,
            "access_delay_s": access_delay,
            "propagation_delay_s": propagation_delay,
            "load_penalty": load_penalty,
        }

    def _generate_current_second_requests(self) -> list[TaskRequest]:
        expected_requests = self._arrival_rate_per_minute() / 60.0
        total_count = int(self.rng.poisson(expected_requests))
        return self._generate_aggregated_requests(total_count)

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
        if total_count <= 0:
            return []
        group_counts = self.rng.multinomial(total_count, self.request_group_probabilities)
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

    def _arrival_rate_per_minute(self) -> float:
        if self.config.arrival_profile == "stationary":
            return self._base_arrival_rate_per_minute() * _STATIONARY_ARRIVAL_FACTOR
        minute_of_day = self.current_time_minute % (24 * 60)
        return self._base_arrival_rate_per_minute() * _daily_arrival_factor(minute_of_day)

    def _base_arrival_rate_per_minute(self) -> float:
        if self.config.mean_requests_per_minute is not None:
            return self.config.mean_requests_per_minute * self.config.demand_load_multiplier
        return (
            self.config.num_users
            * self.config.active_user_ratio
            * self.config.active_user_request_rate_per_minute
            * self.config.traffic_scale
            * self.config.demand_load_multiplier
        )

    def _update_dynamic_loads_batch(
        self,
        group_infos: list[dict[str, Any]],
        requests: list[TaskRequest],
    ) -> None:
        elapsed_minutes = self.config.request_aggregation_window_seconds / 60.0
        decay = float(np.exp(-elapsed_minutes / self.config.load_ewma_tau_minutes))
        self.node_compute_load *= decay
        self.link_load *= decay
        self.last_load_update_minute = self.current_time_minute + elapsed_minutes
        ewma_window_s = self.config.load_ewma_tau_minutes * 60.0
        for info, request in zip(group_infos, requests):
            request_count = float(request.request_count)
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
                self.link_load[demand.src_node, demand.dst_node] = min(
                    1.0,
                    self.link_load[demand.src_node, demand.dst_node] + increment,
                )

    @staticmethod
    def _weighted_group_mean(values: list[float], weights: np.ndarray) -> float:
        if not values or float(weights.sum()) <= 0.0:
            return 0.0
        return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))

    def _summarize_batch_info(self, group_infos: list[dict[str, Any]], counts: np.ndarray) -> dict[str, Any]:
        def mean(key: str) -> float:
            return self._weighted_group_mean([float(info[key]) for info in group_infos], counts)

        return {
            "valid": all(bool(info["valid"]) for info in group_infos),
            "violations": [violation for info in group_infos for violation in info["violations"]],
            "latency_s": mean("latency_s"),
            "physical_latency_s": mean("physical_latency_s"),
            "penalty_latency_s": mean("penalty_latency_s"),
            "compute_delay_s": mean("compute_delay_s"),
            "link_delay_s": mean("link_delay_s"),
            "access_delay_s": mean("access_delay_s"),
            "propagation_delay_s": mean("propagation_delay_s"),
            "load_penalty": mean("load_penalty"),
        }

    def _require_ready(self) -> None:
        if self.scenario is None:
            raise RuntimeError("call reset() before using the environment")
