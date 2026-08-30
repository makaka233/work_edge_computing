from __future__ import annotations

import time

import numpy as np
from ortools.sat.python import cp_model

from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.static_costs import all_pairs_static_delay, expected_home_service_demand
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


class SICPScheme(BaseComparisonScheme):
    """Adapted JPS-CP service-chain placement, without TSN gate scheduling."""

    name = "SICP"

    def __init__(self, solver_time_limit_s: float = 120.0, workers: int = 1) -> None:
        super().__init__()
        self.solver_time_limit_s = float(solver_time_limit_s)
        self.workers = int(workers)
        self.chains: dict[int, tuple[int, ...]] = {}

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        started = time.perf_counter()
        deployment, chains, objective, status = self._solve_cp_sat(env)
        env.apply_deployment(deployment)
        self.chains = chains
        self.diagnostics.planning.append(
            {
                "time_minute": env.current_time_minute,
                "objective": objective,
                "solver_status": status,
                "planning_time_s": time.perf_counter() - started,
                "replicas": int(deployment.sum()),
            }
        )

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        return [list(self.chains[request.service_id]) for request in requests]

    def _solve_cp_sat(
        self, env: EdgeComputingEnv
    ) -> tuple[np.ndarray, dict[int, tuple[int, ...]], float, str]:
        env._require_ready()
        assert env.scenario is not None
        model = cp_model.CpModel()
        nodes = range(env.config.num_edge_nodes)
        demand = expected_home_service_demand(env)
        base_routes = all_pairs_static_delay(env)
        scale = 1_000_000
        resource_scale = 1_000
        x: dict[tuple[int, int, int], cp_model.IntVar] = {}
        z: dict[tuple[int, int, int, int], cp_model.IntVar] = {}
        objective_terms: list[cp_model.LinearExpr] = []

        for service in env.scenario.services:
            service_rate = float(demand[:, service.service_id].sum())
            for stage in service.stages:
                stage_vars = []
                for node in nodes:
                    variable = model.new_bool_var(f"x_{service.service_id}_{stage.stage_id}_{node}")
                    x[service.service_id, stage.stage_id, node] = variable
                    stage_vars.append(variable)
                    compute_cost = service_rate * stage.compute_gcycles_mean / max(
                        env.scenario.nodes[node].compute_gcycles_per_s, 1e-12
                    )
                    if stage.stage_id == 0:
                        unreachable = any(
                            demand[home, service.service_id] > 0.0
                            and not np.isfinite(base_routes[home, node])
                            for home in nodes
                        )
                        if unreachable:
                            model.add(variable == 0)
                        ingress_cost = sum(
                            demand[home, service.service_id]
                            * (
                                base_routes[home, node]
                                + _static_transfer_only(env, home, node, service.input_mb_mean)
                            )
                            for home in nodes
                            if np.isfinite(base_routes[home, node])
                        )
                    else:
                        ingress_cost = 0.0
                    objective_terms.append(int(round((compute_cost + ingress_cost) * scale)) * variable)
                model.add_exactly_one(stage_vars)

            for stage_id in range(len(service.stages) - 1):
                data_mb = service.stages[stage_id].output_mb_mean
                for left in nodes:
                    for right in nodes:
                        variable = model.new_bool_var(f"z_{service.service_id}_{stage_id}_{left}_{right}")
                        z[service.service_id, stage_id, left, right] = variable
                        model.add(variable <= x[service.service_id, stage_id, left])
                        model.add(variable <= x[service.service_id, stage_id + 1, right])
                        model.add(
                            variable
                            >= x[service.service_id, stage_id, left]
                            + x[service.service_id, stage_id + 1, right]
                            - 1
                        )
                        if not np.isfinite(base_routes[left, right]):
                            model.add(variable == 0)
                            continue
                        cost = service_rate * (
                            base_routes[left, right] + _static_transfer_only(env, left, right, data_mb)
                        )
                        objective_terms.append(int(round(cost * scale)) * variable)

        memory_caps = env.service_memory_capacities()
        storage_caps = env.service_storage_capacities()
        for node in nodes:
            memory_terms = []
            storage_terms = []
            for service in env.scenario.services:
                for stage in service.stages:
                    variable = x[service.service_id, stage.stage_id, node]
                    memory_terms.append(int(round(stage.memory_gb * resource_scale)) * variable)
                    storage_terms.append(int(round(stage.storage_gb * resource_scale)) * variable)
            model.add(sum(memory_terms) <= int(np.floor(memory_caps[node] * resource_scale + 1e-9)))
            model.add(sum(storage_terms) <= int(np.floor(storage_caps[node] * resource_scale + 1e-9)))
        model.minimize(sum(objective_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.solver_time_limit_s
        solver.parameters.num_search_workers = self.workers
        solver.parameters.random_seed = 0
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"SICP CP-SAT failed with status {solver.status_name(status)}")
        deployment = np.zeros(
            (env.config.num_service_types, env.config.max_service_stages, env.config.num_edge_nodes),
            dtype=bool,
        )
        chains: dict[int, tuple[int, ...]] = {}
        for service in env.scenario.services:
            chain = []
            for stage in service.stages:
                selected = [node for node in nodes if solver.value(x[service.service_id, stage.stage_id, node])]
                if len(selected) != 1:
                    raise RuntimeError("SICP did not return exactly one node per stage")
                node = int(selected[0])
                deployment[service.service_id, stage.stage_id, node] = True
                chain.append(node)
            chains[service.service_id] = tuple(chain)
        feasible, reason = env.check_deployment_feasible(deployment)
        if not feasible:
            raise RuntimeError(f"SICP deployment infeasible: {reason}")
        return deployment, chains, float(solver.objective_value / scale), solver.status_name(status)


def _static_transfer_only(env: EdgeComputingEnv, src: int, dst: int, data_mb: float) -> float:
    if src == dst:
        return 0.0
    assert env.scenario is not None
    path = env.shortest_path(src, dst)
    if path is None:
        return float("inf")
    return float(
        sum(data_mb / max(env.scenario.bandwidth_mb_s[left, right], 1e-12) for left, right in zip(path, path[1:]))
    )
