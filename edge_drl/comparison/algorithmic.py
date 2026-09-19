from __future__ import annotations

import heapq
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from edge_drl.comparison.checkpoint import build_proposed_agent
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import Service, TaskRequest


class CausalKKTPathScheduler:
    """Route serial stages by the marginal cost of the shared KKT allocator.

    A batch ledger is separate for every causal compute and transfer phase.
    Given previous demands with total weighted square roots S and a new demand
    of size d, the predicted increase of total delay is
    ``((S + sqrt(d))**2 - S**2) / effective_capacity``. The environment still
    performs the authoritative joint KKT settlement after all paths are chosen.
    """

    def __init__(self) -> None:
        self._paths: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], float] | None] = {}

    def _path(
        self, env: EdgeComputingEnv, source: int, target: int
    ) -> tuple[tuple[tuple[int, int], ...], float] | None:
        key = (int(source), int(target))
        if key not in self._paths:
            route = env.shortest_path(*key)
            if route is None:
                self._paths[key] = None
            else:
                assert env.scenario is not None
                edges = tuple((a, b) for a, b in zip(route, route[1:]))
                propagation = sum(float(env.scenario.propagation_ms[a, b]) / 1000.0 for a, b in edges)
                self._paths[key] = edges, propagation
        return self._paths[key]

    @staticmethod
    def _increment(size: float, previous_sqrt: float, capacity: float, multiplicity: float) -> float:
        if size <= 0.0:
            return 0.0
        weighted_sqrt = multiplicity * np.sqrt(size)
        return float((2.0 * previous_sqrt * weighted_sqrt + weighted_sqrt**2) / max(capacity, 1e-9))

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        if env.scenario is None or env.deployment is None:
            raise RuntimeError("environment must be reset before scheduling")
        nodes = env.config.num_edge_nodes
        stages = env.config.max_service_stages
        compute_sqrt = np.zeros((stages, nodes), dtype=np.float64)
        link_sqrt = np.zeros((stages, nodes, nodes), dtype=np.float64)
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes])
        capacities *= np.clip(1.0 - 0.75 * env.node_compute_load, 0.10, 1.0)
        bandwidth = env.scenario.bandwidth_mb_s * np.clip(1.0 - 0.75 * env.link_load, 0.10, 1.0)

        def transfer(phase: int, source: int, target: int, size: float, count: float) -> float:
            path = self._path(env, source, target)
            if path is None:
                return float("inf")
            edges, propagation = path
            return count * propagation + sum(
                self._increment(size, link_sqrt[phase, a, b], bandwidth[a, b], count)
                for a, b in edges
            )

        schedules: list[list[int]] = []
        for request in requests:
            mask = env.scheduler_candidate_mask(request)
            candidates = [np.flatnonzero(row).tolist() for row in mask]
            if not candidates or any(not row for row in candidates):
                raise RuntimeError(f"no deployed node for service {request.service_id}")
            count = float(request.request_count)
            previous: dict[int, float] = {}
            back: list[dict[int, int]] = []
            for stage_id, allowed in enumerate(candidates):
                current: dict[int, float] = {}
                parents: dict[int, int] = {}
                for node_id in allowed:
                    work = float(request.stage_compute_gcycles[stage_id])
                    compute = self._increment(work, compute_sqrt[stage_id, node_id], capacities[node_id], count)
                    if stage_id == 0:
                        total = transfer(0, request.home_node, node_id, request.input_mb, count) + compute
                        parent = -1
                    else:
                        options = (
                            (
                                previous[parent_id]
                                + transfer(
                                    stage_id,
                                    parent_id,
                                    node_id,
                                    request.stage_output_mb[stage_id - 1],
                                    count,
                                )
                                + compute,
                                parent_id,
                            )
                            for parent_id in previous
                        )
                        total, parent = min(options, default=(float("inf"), -1))
                    if np.isfinite(total):
                        current[node_id] = total
                        parents[node_id] = parent
                if not current:
                    raise RuntimeError(f"no reachable serial path for service {request.service_id}")
                previous = current
                back.append(parents)
            chosen = min(previous, key=lambda node_id: (previous[node_id], node_id))
            schedule = [chosen]
            for stage_id in range(len(back) - 1, 0, -1):
                chosen = back[stage_id][chosen]
                schedule.append(chosen)
            schedule.reverse()
            schedules.append(schedule)
            for stage_id, node_id in enumerate(schedule):
                compute_sqrt[stage_id, node_id] += count * np.sqrt(max(request.stage_compute_gcycles[stage_id], 0.0))
                source = request.home_node if stage_id == 0 else schedule[stage_id - 1]
                size = request.input_mb if stage_id == 0 else request.stage_output_mb[stage_id - 1]
                path = self._path(env, source, node_id)
                assert path is not None
                for a, b in path[0]:
                    link_sqrt[stage_id, a, b] += count * np.sqrt(max(size, 0.0))
        return schedules


class DemandAwareMILPPlacement:
    """MILP mandatory placement followed by feasible marginal replica search."""

    def __init__(self, *, max_additions: int = 0, candidate_nodes: int = 0, solver_time_limit_s: float = 5.0) -> None:
        self.max_additions = max_additions
        self.candidate_nodes = candidate_nodes
        self.solver_time_limit_s = solver_time_limit_s

    @staticmethod
    def _demand(env: EdgeComputingEnv) -> np.ndarray:
        assert env.scenario is not None
        demand = np.zeros((env.config.num_edge_nodes, env.config.num_service_types), dtype=np.float64)
        for user in env.scenario.users:
            demand[user.home_node] += user.service_weights
        return demand / max(len(env.scenario.users), 1)

    @staticmethod
    def _transfer_matrices(env: EdgeComputingEnv) -> tuple[np.ndarray, np.ndarray]:
        assert env.scenario is not None
        nodes = env.config.num_edge_nodes
        propagation = np.full((nodes, nodes), np.inf, dtype=np.float64)
        inverse_bandwidth = np.full((nodes, nodes), np.inf, dtype=np.float64)
        for source in range(nodes):
            for target in range(nodes):
                route = env.shortest_path(source, target)
                if route is None:
                    continue
                edges = tuple(zip(route, route[1:]))
                propagation[source, target] = sum(
                    float(env.scenario.propagation_ms[a, b]) / 1000.0 for a, b in edges
                )
                inverse_bandwidth[source, target] = sum(
                    1.0 / max(
                        float(env.scenario.bandwidth_mb_s[a, b])
                        * max(1.0 - 0.75 * float(env.link_load[a, b]), 0.10),
                        1e-9,
                    )
                    for a, b in edges
                )
        return propagation, inverse_bandwidth

    @staticmethod
    def _service_cost(
        service: Service,
        selected: np.ndarray,
        demand: np.ndarray,
        node_capacities: np.ndarray,
        propagation: np.ndarray,
        inverse_bandwidth: np.ndarray,
        task_compute_scale: float,
        task_data_scale: float,
        arrival_rate_per_s: float,
    ) -> float:
        allowed = [np.flatnonzero(selected[stage.stage_id]) for stage in service.stages]
        if any(len(nodes) == 0 for nodes in allowed):
            return float("inf")
        totals = 0.0
        for home in np.flatnonzero(demand):
            first = allowed[0]
            cost = (
                propagation[home, first]
                + service.input_mb_mean * task_data_scale * inverse_bandwidth[home, first]
                + service.stages[0].compute_gcycles_mean * task_compute_scale / node_capacities[first]
            )
            for stage_id in range(1, len(allowed)):
                previous, current = allowed[stage_id - 1], allowed[stage_id]
                transfer = (
                    propagation[np.ix_(previous, current)]
                    + service.stages[stage_id - 1].output_mb_mean
                    * task_data_scale
                    * inverse_bandwidth[np.ix_(previous, current)]
                )
                cost = np.min(cost[:, None] + transfer, axis=0)
                cost += service.stages[stage_id].compute_gcycles_mean * task_compute_scale / node_capacities[current]
            totals += float(demand[home]) * float(np.min(cost))
        # When several requests of the same stage arrive in a second, the
        # KKT delay grows with their shared square-root workload. A pooled
        # capacity surrogate credits replicas even when the closest path for
        # a representative home has not changed yet.
        service_mass = float(demand.sum())
        for stage in service.stages:
            pooled_capacity = float(node_capacities[allowed[stage.stage_id]].sum())
            totals += (
                service_mass**2
                * arrival_rate_per_s
                * stage.compute_gcycles_mean
                * task_compute_scale
                / max(pooled_capacity, 1e-9)
            )
        return totals

    def plan(self, env: EdgeComputingEnv) -> tuple[np.ndarray, dict[str, Any]]:
        if env.scenario is None or env.deployment is None:
            raise RuntimeError("environment must be reset before planning")
        started = time.perf_counter()
        nodes = env.config.num_edge_nodes
        services = env.scenario.services
        stages = [(service, stage) for service in services for stage in service.stages]
        demand = self._demand(env)
        propagation, inverse_bandwidth = self._transfer_matrices(env)
        node_capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes])
        node_capacities *= np.clip(1.0 - 0.75 * env.node_compute_load, 0.10, 1.0)
        arrival_rate_per_s = env._arrival_rate_per_minute() / 60.0
        memory_caps = env.service_memory_capacities()
        storage_caps = env.service_storage_capacities()
        stage_cost = np.empty((len(stages), nodes), dtype=np.float64)
        for index, (service, stage) in enumerate(stages):
            volume = service.input_mb_mean if stage.stage_id == 0 else service.stages[stage.stage_id - 1].output_mb_mean
            transport = propagation + volume * env.config.task_data_scale * inverse_bandwidth
            weighted_access = demand[:, service.service_id] @ transport
            workload = stage.compute_gcycles_mean * env.config.task_compute_scale / node_capacities
            stage_cost[index] = weighted_access + demand[:, service.service_id].sum() * workload

        variables = len(stages) * nodes
        coefficients = stage_cost.reshape(-1)
        constraints = np.zeros((len(stages) + 2 * nodes, variables), dtype=np.float64)
        for index, (_, stage) in enumerate(stages):
            constraints[index, index * nodes : (index + 1) * nodes] = 1.0
            for node_id in range(nodes):
                column = index * nodes + node_id
                constraints[len(stages) + node_id, column] = stage.memory_gb
                constraints[len(stages) + nodes + node_id, column] = stage.storage_gb
        lower = np.concatenate((np.ones(len(stages)), np.zeros(2 * nodes)))
        upper = np.concatenate((np.ones(len(stages)), memory_caps, storage_caps))
        result = milp(
            c=coefficients,
            integrality=np.ones(variables, dtype=np.int8),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(constraints, lower, upper),
            options={"time_limit": max(self.solver_time_limit_s, 0.1), "mip_rel_gap": 0.01},
        )
        if result.x is None:
            raise RuntimeError(f"mandatory deployment MILP failed: {result.message}")
        deployment = np.zeros_like(env.deployment, dtype=bool)
        for index, (service, stage) in enumerate(stages):
            deployment[service.service_id, stage.stage_id] = result.x[index * nodes : (index + 1) * nodes] > 0.5
        feasible, reason = env.check_deployment_feasible(deployment)
        if not feasible:
            raise RuntimeError(f"mandatory deployment MILP returned infeasible plan: {reason}")

        memory_left = memory_caps.copy()
        storage_left = storage_caps.copy()
        for service, stage in stages:
            memory_left -= deployment[service.service_id, stage.stage_id] * stage.memory_gb
            storage_left -= deployment[service.service_id, stage.stage_id] * stage.storage_gb
        costs = {
            service.service_id: self._service_cost(
                service, deployment[service.service_id], demand[:, service.service_id],
                node_capacities, propagation, inverse_bandwidth,
                env.config.task_compute_scale, env.config.task_data_scale, arrival_rate_per_s,
            )
            for service in services
        }
        shortlist = [
            np.argsort(stage_cost[index])[: self.candidate_nodes]
            if self.candidate_nodes > 0 else np.argsort(stage_cost[index])
            for index in range(len(stages))
        ]
        stage_indices: dict[int, list[int]] = {}
        for index, (service, _) in enumerate(stages):
            stage_indices.setdefault(service.service_id, []).append(index)
        versions = {service.service_id: 0 for service in services}
        heap: list[tuple[float, int, int, int, float]] = []

        def refresh_service(service_id: int) -> None:
            version = versions[service_id]
            for index in stage_indices[service_id]:
                service, stage = stages[index]
                for node_id in shortlist[index]:
                    if deployment[service_id, stage.stage_id, node_id]:
                        continue
                    if memory_left[node_id] + 1e-9 < stage.memory_gb or storage_left[node_id] + 1e-9 < stage.storage_gb:
                        continue
                    deployment[service_id, stage.stage_id, node_id] = True
                    new_cost = self._service_cost(
                        service, deployment[service_id], demand[:, service_id],
                        node_capacities, propagation, inverse_bandwidth,
                        env.config.task_compute_scale, env.config.task_data_scale, arrival_rate_per_s,
                    )
                    deployment[service_id, stage.stage_id, node_id] = False
                    gain = costs[service_id] - new_cost
                    if gain <= 1e-12:
                        continue
                    resource_fraction = (
                        stage.memory_gb / max(memory_caps[node_id], 1e-9)
                        + stage.storage_gb / max(storage_caps[node_id], 1e-9)
                    )
                    priority = gain / max(resource_fraction, 1e-9)
                    heapq.heappush(heap, (-priority, index, int(node_id), version, new_cost))

        for service in services:
            refresh_service(service.service_id)
        additions = 0
        addition_limit = self.max_additions if self.max_additions > 0 else len(stages) * nodes
        while heap and additions < addition_limit:
            _, index, node_id, version, new_cost = heapq.heappop(heap)
            service, stage = stages[index]
            if version != versions[service.service_id] or deployment[service.service_id, stage.stage_id, node_id]:
                continue
            if memory_left[node_id] + 1e-9 < stage.memory_gb or storage_left[node_id] + 1e-9 < stage.storage_gb:
                continue
            deployment[service.service_id, stage.stage_id, node_id] = True
            memory_left[node_id] -= stage.memory_gb
            storage_left[node_id] -= stage.storage_gb
            costs[service.service_id] = new_cost
            additions += 1
            versions[service.service_id] += 1
            refresh_service(service.service_id)
        feasible, reason = env.check_deployment_feasible(deployment)
        if not feasible:
            raise RuntimeError(f"replica search returned infeasible plan: {reason}")
        return deployment, {
            "solver_status": str(result.message),
            "mandatory_objective": float(result.fun),
            "forecast_latency_cost": float(sum(costs.values())),
            "added_replicas": additions,
            "replicas": int(deployment.sum()),
            "planning_time_s": time.perf_counter() - started,
        }


class HybridScheme(BaseComparisonScheme):
    """Frozen Proposed Slow policy with the common algorithmic Fast scheduler."""

    name = "Hybrid"

    def __init__(
        self, env: EdgeComputingEnv, checkpoint_path: str | Path,
        checkpoint_args: dict[str, Any], *, device: str,
    ) -> None:
        super().__init__()
        self.agent, self.checkpoint_metadata = build_proposed_agent(
            env, checkpoint_path, checkpoint_args, device=device
        )
        self.scheduler = CausalKKTPathScheduler()

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        self.agent.maybe_update_deployment(env, deterministic=True, record=False)
        assert env.deployment is not None
        self.diagnostics.planning.append(
            {
                "time_minute": env.current_time_minute,
                "planner": "frozen_proposed_slow",
                "replicas": int(env.deployment.sum()),
            }
        )

    def schedule_batch(self, env: EdgeComputingEnv, requests: list[TaskRequest]) -> list[list[int]]:
        return self.scheduler.schedule_batch(env, requests)


class AlgorithmicScheme(BaseComparisonScheme):
    """Demand-aware integer deployment with the same Fast algorithm as Hybrid."""

    name = "Algorithmic"

    def __init__(self) -> None:
        super().__init__()
        self.planner = DemandAwareMILPPlacement()
        self.scheduler = CausalKKTPathScheduler()

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        deployment, details = self.planner.plan(env)
        env.apply_deployment(deployment)
        self.diagnostics.planning.append({"time_minute": env.current_time_minute, **details})

    def schedule_batch(self, env: EdgeComputingEnv, requests: list[TaskRequest]) -> list[list[int]]:
        return self.scheduler.schedule_batch(env, requests)
