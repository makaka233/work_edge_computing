from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.static_costs import all_pairs_static_delay, expected_home_service_demand
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


class MonolithicScheme(BaseComparisonScheme):
    """Optimization-backed monolithic-service adaptation of reference [2]."""

    name = "Monolithic"

    def __init__(self, solver_time_limit_s: float = 120.0) -> None:
        super().__init__()
        self.solver_time_limit_s = float(solver_time_limit_s)
        self.replica_nodes: dict[int, tuple[int, ...]] = {}

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        started = time.perf_counter()
        deployment, objective, status = self._solve_facility_location(env)
        env.apply_deployment(deployment)
        self.replica_nodes = {
            service.service_id: tuple(
                int(node) for node in np.flatnonzero(deployment[service.service_id, 0])
            )
            for service in env.scenario.services  # type: ignore[union-attr]
        }
        self.diagnostics.planning.append(
            {
                "time_minute": env.current_time_minute,
                "objective": objective,
                "solver_status": status,
                "planning_time_s": time.perf_counter() - started,
                "replicas": int(deployment[:, 0, :].sum()),
            }
        )

    def adapt_requests(self, requests: list[TaskRequest]) -> list[TaskRequest]:
        adapted: list[TaskRequest] = []
        for request in requests:
            stage_count = len(request.stage_compute_gcycles)
            adapted.append(
                replace(
                    request,
                    stage_compute_gcycles=(float(sum(request.stage_compute_gcycles)),)
                    + (0.0,) * (stage_count - 1),
                    stage_output_mb=(0.0,) * stage_count,
                )
            )
        return adapted

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        schedules: list[list[int]] = []
        for request in requests:
            candidates = self.replica_nodes.get(request.service_id, ())
            if not candidates:
                raise RuntimeError(f"Monolithic has no replica for service {request.service_id}")
            best = min(
                candidates,
                key=lambda node: (
                    env.evaluate_schedule(request, [node] * len(request.stage_compute_gcycles))["latency_s"]
                    + 0.05 * env.node_compute_load[node],
                    node,
                ),
            )
            schedules.append([int(best)] * len(request.stage_compute_gcycles))
        return schedules

    def _solve_facility_location(self, env: EdgeComputingEnv) -> tuple[np.ndarray, float, str]:
        env._require_ready()
        assert env.scenario is not None
        services = env.scenario.services
        node_count = env.config.num_edge_nodes
        service_count = env.config.num_service_types
        demand = expected_home_service_demand(env)
        ingress = all_pairs_static_delay(env)
        y_count = service_count * node_count
        assignment_offset = y_count
        variable_count = y_count + node_count * service_count * node_count

        def y(service: int, node: int) -> int:
            return service * node_count + node

        def a(home: int, service: int, node: int) -> int:
            return assignment_offset + (home * service_count + service) * node_count + node

        objective = np.zeros(variable_count, dtype=np.float64)
        for service in services:
            total_compute = sum(stage.compute_gcycles_mean for stage in service.stages)
            for node in range(node_count):
                objective[y(service.service_id, node)] = 1e-9
                compute = total_compute / max(
                    env.scenario.nodes[node].compute_gcycles_per_s
                    * (1.0 - 0.75 * env.node_compute_load[node]),
                    1e-9,
                )
                for home in range(node_count):
                    objective[a(home, service.service_id, node)] = demand[home, service.service_id] * (
                        ingress[home, node] + compute
                    )

        row_count = node_count * service_count + node_count * service_count * node_count + 2 * node_count
        matrix = lil_matrix((row_count, variable_count), dtype=np.float64)
        lower = np.full(row_count, -np.inf, dtype=np.float64)
        upper = np.full(row_count, np.inf, dtype=np.float64)
        row = 0
        for home in range(node_count):
            for service in range(service_count):
                for node in range(node_count):
                    matrix[row, a(home, service, node)] = 1.0
                lower[row] = upper[row] = 1.0
                row += 1
        for home in range(node_count):
            for service in range(service_count):
                for node in range(node_count):
                    matrix[row, a(home, service, node)] = 1.0
                    matrix[row, y(service, node)] = -1.0
                    upper[row] = 0.0
                    row += 1
        for node in range(node_count):
            for service in services:
                matrix[row, y(service.service_id, node)] = sum(s.memory_gb for s in service.stages)
            upper[row] = env.service_memory_capacities()[node]
            row += 1
        for node in range(node_count):
            for service in services:
                matrix[row, y(service.service_id, node)] = sum(s.storage_gb for s in service.stages)
            upper[row] = env.service_storage_capacities()[node]
            row += 1
        assert row == row_count
        result = milp(
            objective,
            integrality=np.ones(variable_count, dtype=np.int8),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={"time_limit": self.solver_time_limit_s},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"Monolithic facility-location solver failed: {result.message}")
        deployment = np.zeros(
            (service_count, env.config.max_service_stages, node_count), dtype=bool
        )
        for service in services:
            for node in range(node_count):
                if result.x[y(service.service_id, node)] >= 0.5:
                    for stage in service.stages:
                        deployment[service.service_id, stage.stage_id, node] = True
        feasible, reason = env.check_deployment_feasible(deployment)
        if not feasible:
            raise RuntimeError(f"Monolithic projected deployment infeasible: {reason}")
        return deployment, float(result.fun), str(result.message)
