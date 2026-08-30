from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.sparse import lil_matrix

from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.static_costs import all_pairs_static_delay, expected_home_service_demand
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


@dataclass(frozen=True)
class VirtualCoreModel:
    f_unit: float
    cores_per_node: np.ndarray
    service_rates: np.ndarray


def build_virtual_core_model(env: EdgeComputingEnv) -> VirtualCoreModel:
    env._require_ready()
    assert env.scenario is not None
    capacities = np.asarray(
        [node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64
    )
    f_unit = float(np.median(capacities) / 10.0)
    cores = np.maximum(1, np.floor(capacities / f_unit)).astype(np.int64)
    means = np.asarray(
        [stage.compute_gcycles_mean * env.config.task_compute_scale for service in env.scenario.services for stage in service.stages],
        dtype=np.float64,
    )
    return VirtualCoreModel(f_unit, cores, f_unit / np.maximum(means, 1e-12))


def algorithm1_rounding(n_star: np.ndarray) -> np.ndarray:
    """Literal per-node flag rounding from AES-JDR Algorithm 1."""

    values = np.asarray(n_star, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("n_star must have shape [microservice, node]")
    rounded = np.zeros_like(values, dtype=np.int64)
    for node in range(values.shape[1]):
        flag = 0.0
        for microservice in range(values.shape[0]):
            value = float(values[microservice, node])
            label = int(np.floor(value + 0.5))
            if label >= value:
                flag += label - value
            else:
                flag -= value - label
            if flag >= 0.999:
                flag -= 2.0 * (label - value)
                rounded[microservice, node] = int(np.floor(value))
            else:
                rounded[microservice, node] = label
    return rounded


def adaptive_scaled_probabilities(
    p_star: np.ndarray,
    n_star: np.ndarray,
    n_integer: np.ndarray,
    cores_per_node: np.ndarray,
) -> np.ndarray:
    p = np.asarray(p_star, dtype=np.float64)
    x = np.asarray(n_star, dtype=np.float64) / np.asarray(cores_per_node, dtype=np.float64)[None, :]
    deployed = np.asarray(n_integer) > 0
    adjusted = np.zeros_like(p)
    valid = deployed & (x > 1e-12)
    adjusted[valid] = p[valid] / x[valid]
    totals = adjusted.sum(axis=1)
    if np.any(totals <= 0.0):
        missing = np.flatnonzero(totals <= 0.0).tolist()
        raise RuntimeError(f"AES-JDR rounding left no routable deployment for microservices {missing}")
    adjusted /= totals[:, None]
    adjusted[~deployed] = 0.0
    return adjusted


class DMDRScheme(BaseComparisonScheme):
    """Native AES-JDR RDMP adaptation using integer multiplicity and sampled routing."""

    name = "DMDR"

    def __init__(
        self,
        *,
        routing_seed: int,
        stability_margin: float = 0.95,
        solver_max_iterations: int = 400,
        solver_tolerance: float = 1e-7,
        plan_cache: dict[int, dict[str, object]] | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < stability_margin < 1.0:
            raise ValueError("stability_margin must be in (0, 1)")
        self.routing_rng = np.random.default_rng(routing_seed)
        self.stability_margin = float(stability_margin)
        self.solver_max_iterations = int(solver_max_iterations)
        self.solver_tolerance = float(solver_tolerance)
        self.plan_cache = plan_cache
        self.stage_keys: list[tuple[int, int]] = []
        self.routing_probabilities: np.ndarray | None = None
        self.integer_multiplicity: np.ndarray | None = None
        self._key_to_index: dict[tuple[int, int], int] = {}

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        started = time.perf_counter()
        window_index = int(round(env.current_time_minute / env.config.deployment_interval_minutes))
        cached = None if self.plan_cache is None else self.plan_cache.get(window_index)
        if cached is None:
            deployment, diagnostics = self._solve_native_rdmp(env)
            if self.plan_cache is not None:
                self.plan_cache[window_index] = {
                    "deployment": deployment.copy(),
                    "diagnostics": deepcopy(diagnostics),
                    "stage_keys": list(self.stage_keys),
                    "routing_probabilities": self.routing_probabilities.copy(),
                    "integer_multiplicity": self.integer_multiplicity.copy(),
                }
        else:
            deployment = np.asarray(cached["deployment"], dtype=bool).copy()
            diagnostics = deepcopy(cached["diagnostics"])
            self.stage_keys = list(cached["stage_keys"])
            self._key_to_index = {key: index for index, key in enumerate(self.stage_keys)}
            self.routing_probabilities = np.asarray(cached["routing_probabilities"], dtype=np.float64).copy()
            self.integer_multiplicity = np.asarray(cached["integer_multiplicity"], dtype=np.int64).copy()
            diagnostics["plan_cache_hit"] = True
        env.apply_deployment(deployment)
        diagnostics["time_minute"] = env.current_time_minute
        diagnostics["planning_time_s"] = time.perf_counter() - started
        self.diagnostics.planning.append(diagnostics)

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        if self.routing_probabilities is None:
            raise RuntimeError("DMDR must be planned before routing")
        schedules: list[list[int]] = []
        for request in requests:
            nodes = []
            for stage_id in range(len(request.stage_compute_gcycles)):
                row = self._key_to_index[request.service_id, stage_id]
                nodes.append(
                    int(
                        self.routing_rng.choice(
                            env.config.num_edge_nodes,
                            p=self.routing_probabilities[row],
                        )
                    )
                )
            schedules.append(nodes)
        return schedules

    def _solve_native_rdmp(self, env: EdgeComputingEnv) -> tuple[np.ndarray, dict[str, object]]:
        env._require_ready()
        assert env.scenario is not None
        virtual = build_virtual_core_model(env)
        self.stage_keys = [
            (service.service_id, stage.stage_id)
            for service in env.scenario.services
            for stage in service.stages
        ]
        self._key_to_index = {key: index for index, key in enumerate(self.stage_keys)}
        microservice_count = len(self.stage_keys)
        node_count = env.config.num_edge_nodes
        value_count = microservice_count * node_count
        demand = expected_home_service_demand(env)
        lambdas = np.asarray(
            [demand[:, service_id].sum() for service_id, _ in self.stage_keys],
            dtype=np.float64,
        )
        static_delay = all_pairs_static_delay(env)

        linear_matrix, linear_lower, linear_upper = self._linear_constraints(
            lambdas, virtual.service_rates, virtual.cores_per_node
        )
        bounds = Bounds(
            np.zeros(2 * value_count, dtype=np.float64),
            np.concatenate(
                [
                    np.tile(virtual.cores_per_node, microservice_count),
                    np.ones(value_count, dtype=np.float64),
                ]
            ),
        )
        feasible = linprog(
            np.concatenate([np.ones(value_count), np.zeros(value_count)]),
            A_ub=linear_matrix[np.isneginf(linear_lower)].toarray(),
            b_ub=linear_upper[np.isneginf(linear_lower)],
            A_eq=linear_matrix[np.isclose(linear_lower, linear_upper)].toarray(),
            b_eq=linear_upper[np.isclose(linear_lower, linear_upper)],
            bounds=list(zip(bounds.lb, bounds.ub)),
            method="highs",
        )
        if not feasible.success or feasible.x is None:
            raise RuntimeError(f"DMDR relaxed RDMP is infeasible: {feasible.message}")

        def objective_and_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
            n = vector[:value_count].reshape(microservice_count, node_count)
            q = vector[value_count:].reshape(microservice_count, node_count)
            gradient_n = np.zeros_like(n)
            gradient_q = np.zeros_like(q)
            objective = 0.0
            for microservice in range(microservice_count):
                rate = lambdas[microservice]
                mu = virtual.service_rates[microservice]
                raw_denominator = n[microservice] * mu - rate * q[microservice]
                active = (n[microservice] > 1e-10) | (q[microservice] > 1e-12)
                denominator = np.maximum(raw_denominator, 1e-12)
                numerator = rate * q[microservice]
                objective += float(np.sum((numerator / denominator)[active]))
                gradient_n[microservice, active] -= numerator[active] * mu / denominator[active] ** 2
                gradient_q[microservice, active] += (
                    rate / denominator[active]
                    + numerator[active] * rate / denominator[active] ** 2
                )

            for service in env.scenario.services:
                first = self._key_to_index[service.service_id, 0]
                home_rate = demand[:, service.service_id]
                ingress_vector = home_rate @ static_delay
                objective += float(ingress_vector @ q[first])
                gradient_q[first] += ingress_vector
                service_rate = float(home_rate.sum())
                for stage_id in range(len(service.stages) - 1):
                    left = self._key_to_index[service.service_id, stage_id]
                    right = self._key_to_index[service.service_id, stage_id + 1]
                    transfer = static_delay.copy()
                    data_mb = service.stages[stage_id].output_mb_mean * env.config.task_data_scale
                    for src in range(node_count):
                        for dst in range(node_count):
                            if src == dst:
                                continue
                            path = env.shortest_path(src, dst)
                            assert path is not None
                            transfer[src, dst] += sum(
                                data_mb / max(env.scenario.bandwidth_mb_s[a, b], 1e-12)
                                for a, b in zip(path, path[1:])
                            )
                    objective += service_rate * float(q[left] @ transfer @ q[right])
                    gradient_q[left] += service_rate * transfer @ q[right]
                    gradient_q[right] += service_rate * transfer.T @ q[left]
            return objective, np.concatenate([gradient_n.ravel(), gradient_q.ravel()])

        equality = np.isclose(linear_lower, linear_upper)
        inequality = ~equality
        result = minimize(
            fun=lambda vector: objective_and_gradient(vector)[0],
            x0=feasible.x,
            jac=lambda vector: objective_and_gradient(vector)[1],
            method="SLSQP",
            bounds=bounds,
            constraints=[
                LinearConstraint(
                    linear_matrix[equality], linear_lower[equality], linear_upper[equality]
                ),
                LinearConstraint(
                    linear_matrix[inequality], linear_lower[inequality], linear_upper[inequality]
                ),
            ],
            options={
                "maxiter": self.solver_max_iterations,
                "ftol": self.solver_tolerance,
                "disp": False,
            },
        )
        residual = _linear_residual(result.x, linear_matrix, linear_lower, linear_upper)
        if not result.success or residual > 1e-5:
            raise RuntimeError(
                f"DMDR relaxed RDMP solver failed: {result.message}; residual={residual:.3e}"
            )
        n_star = result.x[:value_count].reshape(microservice_count, node_count)
        p_star = result.x[value_count:].reshape(microservice_count, node_count)
        n_integer = algorithm1_rounding(n_star)
        if np.any(n_integer.sum(axis=0) > virtual.cores_per_node):
            raise RuntimeError("AES-JDR Algorithm 1 rounding violated virtual-core capacity")
        routing = adaptive_scaled_probabilities(
            p_star, n_star, n_integer, virtual.cores_per_node
        )
        deployment = np.zeros(
            (env.config.num_service_types, env.config.max_service_stages, node_count), dtype=bool
        )
        for index, (service_id, stage_id) in enumerate(self.stage_keys):
            deployment[service_id, stage_id] = n_integer[index] > 0
        feasible_deployment, reason = env.check_deployment_feasible(deployment)
        if not feasible_deployment:
            raise RuntimeError(
                "DMDR native multiplicity projection violates adapted memory/storage constraint: "
                + reason
            )
        self.integer_multiplicity = n_integer
        self.routing_probabilities = routing
        entropy = -np.sum(routing * np.log(np.maximum(routing, 1e-12)), axis=1)
        native_success_weight = 0.0
        native_total_weight = 0.0
        for service in env.scenario.services:
            service_rate = float(demand[:, service.service_id].sum())
            if service_rate <= 0.0:
                continue
            service_delay = 0.0
            first = self._key_to_index[service.service_id, 0]
            service_delay += float((demand[:, service.service_id] @ static_delay @ p_star[first]) / service_rate)
            for stage in service.stages:
                index = self._key_to_index[service.service_id, stage.stage_id]
                denominator = np.maximum(
                    n_star[index] * virtual.service_rates[index]
                    - lambdas[index] * p_star[index],
                    1e-12,
                )
                service_delay += float(np.sum(p_star[index] / denominator))
                if stage.stage_id < len(service.stages) - 1:
                    next_index = self._key_to_index[service.service_id, stage.stage_id + 1]
                    service_delay += float(p_star[index] @ static_delay @ p_star[next_index])
            native_total_weight += service_rate
            if service_delay <= service.deadline_s_mean:
                native_success_weight += service_rate
        return deployment, {
            "solver": "scipy.optimize.SLSQP",
            "solver_status": str(result.message),
            "objective": float(result.fun),
            "native_RDMP_objective": float(result.fun),
            "native_success_rate": native_success_weight / max(native_total_weight, 1e-12),
            "success": bool(result.success),
            "constraint_residual": float(residual),
            "max_constraint_residual": float(residual),
            "f_unit": virtual.f_unit,
            "virtual_cores_total": int(virtual.cores_per_node.sum()),
            "virtual_cores_min": int(virtual.cores_per_node.min()),
            "virtual_cores_max": int(virtual.cores_per_node.max()),
            "service_rate_min": float(virtual.service_rates.min()),
            "service_rate_max": float(virtual.service_rates.max()),
            "native_instance_total": int(n_integer.sum()),
            "N_total": int(n_integer.sum()),
            "physical_replica_total": int(deployment.sum()),
            "routing_entropy_mean": float(entropy.mean()),
            "routing_entropy": float(entropy.mean()),
            "stability_margin": self.stability_margin,
            "memory_storage_constraint": "indicator(N_integer > 0), adapted to common simulator",
            "multiplicity_projection": "native N_integer is diagnostic; physical simulator uses bool(N_integer > 0)",
        }

    def _linear_constraints(
        self,
        lambdas: np.ndarray,
        service_rates: np.ndarray,
        cores: np.ndarray,
    ):
        microservice_count = len(lambdas)
        node_count = len(cores)
        value_count = microservice_count * node_count
        row_count = microservice_count + node_count + 2 * value_count
        matrix = lil_matrix((row_count, 2 * value_count), dtype=np.float64)
        lower = np.full(row_count, -np.inf, dtype=np.float64)
        upper = np.full(row_count, np.inf, dtype=np.float64)
        row = 0
        for microservice in range(microservice_count):
            start = value_count + microservice * node_count
            matrix[row, start : start + node_count] = 1.0
            lower[row] = upper[row] = 1.0
            row += 1
        for node in range(node_count):
            for microservice in range(microservice_count):
                matrix[row, microservice * node_count + node] = 1.0
            upper[row] = float(cores[node])
            row += 1
        for microservice in range(microservice_count):
            for node in range(node_count):
                n_index = microservice * node_count + node
                q_index = value_count + n_index
                matrix[row, q_index] = 1.0
                matrix[row, n_index] = -1.0 / float(cores[node])
                upper[row] = 0.0
                row += 1
        for microservice in range(microservice_count):
            for node in range(node_count):
                n_index = microservice * node_count + node
                q_index = value_count + n_index
                matrix[row, q_index] = float(lambdas[microservice])
                matrix[row, n_index] = -self.stability_margin * float(service_rates[microservice])
                upper[row] = 0.0
                row += 1
        return matrix.tocsr(), lower, upper


def _linear_residual(vector, matrix, lower, upper) -> float:
    values = np.asarray(matrix @ vector, dtype=np.float64)
    lower_violation = np.where(np.isfinite(lower), np.maximum(lower - values, 0.0), 0.0)
    upper_violation = np.where(np.isfinite(upper), np.maximum(values - upper, 0.0), 0.0)
    return float(max(lower_violation.max(initial=0.0), upper_violation.max(initial=0.0)))
