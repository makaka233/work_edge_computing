"""Versioned greedy DARP/DPMS schemes; no outcome-dependent parameters.

The previous implementations remain in heuristic.py and algorithmic.py.
These are project-defined alternatives, not reproductions of a published method.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from edge_drl.comparison.algorithmic import CausalKKTPathScheduler, DemandAwareMILPPlacement, DPMSScheme
from edge_drl.comparison.heuristic import DARPScheme


BASELINE_IMPLEMENTATIONS = ("legacy-v1", "demand-greedy-v2")
GREEDY_IMPLEMENTATION = "demand-greedy-v2"


class DemandGreedyPlacement:
    """Greedy coverage, then demand/count replica allocation and local placement.

    Coverage places the largest normalized resource footprints first. Node
    scores use demand-weighted ingress locality and standalone execution cost.
    Additional replicas are allocated by service demand / current stage count.
    All feasible nodes are scored; only physical RAM/SSD limits stop allocation.
    A zero-objective feasibility MILP repairs fragmented mandatory coverage.
    """

    def plan(self, env):
        started = time.perf_counter()
        assert env.scenario is not None and env.deployment is not None
        stages = [(service, stage) for service in env.scenario.services for stage in service.stages]
        n, k = env.config.num_edge_nodes, len(stages)
        demand = DemandAwareMILPPlacement._demand(env)
        propagation, inverse_bw = DemandAwareMILPPlacement._transfer_matrices(env)
        capacity = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=float)
        capacity *= np.clip(1.0 - 0.75 * env.node_compute_load, 0.1, 1.0)
        memory_cap = env.service_memory_capacities().copy()
        storage_cap = env.service_storage_capacities().copy()
        memory, storage = memory_cap.copy(), storage_cap.copy()
        deployment = np.zeros_like(env.deployment, dtype=bool)
        costs, heat, footprints = [], [], []
        for service, stage in stages:
            weights = demand[:, service.service_id]
            active = weights > 0
            mass = float(weights.sum())
            volume = (service.input_mb_mean if stage.stage_id == 0
                      else service.stages[stage.stage_id - 1].output_mb_mean)
            # Later-stage ingress is a locality proxy, not a chain forecast.
            access = propagation.copy()
            if volume > 0:
                access += volume * env.config.task_data_scale * inverse_bw
            local = weights[active] @ access[active] if active.any() else np.zeros(n)
            costs.append(local + mass * stage.compute_gcycles_mean * env.config.task_compute_scale / capacity)
            heat.append(mass)
            footprints.append(max(stage.memory_gb / max(float(memory_cap.max()), 1e-12),
                                  stage.storage_gb / max(float(storage_cap.max()), 1e-12)))
        costs = np.asarray(costs)

        def feasible(i):
            service, stage = stages[i]
            return np.flatnonzero(~deployment[service.service_id, stage.stage_id]
                                  & (memory + 1e-9 >= stage.memory_gb)
                                  & (storage + 1e-9 >= stage.storage_gb)
                                  & np.isfinite(costs[i]))

        def place(i, node):
            service, stage = stages[i]
            deployment[service.service_id, stage.stage_id, node] = True
            memory[node] -= stage.memory_gb
            storage[node] -= stage.storage_gb

        repaired = False
        solver_status = "not_needed"
        for i in sorted(range(k), key=lambda i: (-footprints[i], -heat[i], i)):
            allowed = feasible(i)
            if allowed.size:
                place(i, int(min(allowed, key=lambda node: (costs[i, node], int(node)))))
                continue
            # Start coverage from scratch; no latency objective or result tuning.
            matrix = np.zeros((k + 2*n, k*n))
            for j, (_, stage) in enumerate(stages):
                matrix[j, j*n:(j+1)*n] = 1
                for node in range(n):
                    matrix[k+node, j*n+node] = stage.memory_gb
                    matrix[k+n+node, j*n+node] = stage.storage_gb
            result = milp(
                c=np.zeros(k*n), integrality=np.ones(k*n),
                bounds=Bounds(0, np.isfinite(costs).reshape(-1).astype(float)),
                constraints=LinearConstraint(matrix, np.r_[np.ones(k), np.zeros(2*n)],
                                             np.r_[np.ones(k), memory_cap, storage_cap]),
                options={"time_limit": 5.0},
            )
            if result.x is None:
                raise RuntimeError(f"Greedy coverage feasibility repair failed: {result.message}")
            deployment.fill(False)
            memory[:], storage[:] = memory_cap, storage_cap
            for j in range(k):
                for node in np.flatnonzero(result.x[j*n:(j+1)*n] > 0.5):
                    place(j, int(node))
            repaired, solver_status = True, str(result.message)
            break

        additions = 0
        while True:
            choices = []
            for i, (service, stage) in enumerate(stages):
                allowed = feasible(i)
                if not allowed.size or heat[i] <= 0:
                    continue
                count = int(deployment[service.service_id, stage.stage_id].sum())
                node = int(min(allowed, key=lambda node: (costs[i, node], int(node))))
                choices.append((-heat[i] / max(count, 1), i, node))
            if not choices:
                break
            _, i, node = min(choices)
            place(i, node)
            additions += 1
        valid, reason = env.check_deployment_feasible(deployment)
        if not valid:
            raise RuntimeError(f"Greedy deployment invalid: {reason}")
        return deployment, {
            "planner": "demand_proportional_greedy_locality",
            "baseline_implementation": GREEDY_IMPLEMENTATION,
            "coverage_repaired": repaired, "solver_status": solver_status,
            "replicas": int(deployment.sum()), "added_replicas": additions,
            "planning_time_s": time.perf_counter() - started,
        }


class StageMarginalScheduler(CausalKKTPathScheduler):
    """Stage-local marginal-cost decisions, without whole-chain optimization.

    Uses the square-root KKT incremental ledger for compute/individual links.
    Link scores approximate shared NIC contention; authoritative settlement
    still enforces the unchanged coupled NIC constraints. Future stages only
    prune unreachable dead ends, never truncate the candidate set.
    """

    def schedule_batch(self, env, requests):
        assert env.scenario is not None and env.deployment is not None
        self._paths.clear()
        n, stages = env.config.num_edge_nodes, env.config.max_service_stages
        compute_sqrt = np.zeros((stages, n))
        link_sqrt = np.zeros((stages, n, n))
        capacity = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=float)
        capacity *= np.clip(1.0 - 0.75 * env.node_compute_load, 0.1, 1.0)
        bandwidth = env.scenario.bandwidth_mb_s * np.clip(1.0 - 0.75 * env.link_load, 0.1, 1.0)
        schedules = []
        for request in requests:
            candidates = [list(np.flatnonzero(row)) for row in env.scheduler_candidate_mask(request)]
            for phase in range(len(candidates) - 2, -1, -1):
                candidates[phase] = [node for node in candidates[phase]
                                     if any(self._path(env, node, dst) is not None for dst in candidates[phase+1])]
            source, path = request.home_node, []
            count = float(request.request_count)
            for phase, allowed in enumerate(candidates):
                size = request.input_mb if phase == 0 else request.stage_output_mb[phase-1]
                work = request.stage_compute_gcycles[phase]
                choices = []
                for node in allowed:
                    route = self._path(env, source, node)
                    if route is None:
                        continue
                    edges, propagation = route
                    transfer = count * propagation + sum(
                        self._increment(size, link_sqrt[phase, a, b], bandwidth[a, b], count)
                        for a, b in edges)
                    score = transfer + self._increment(work, compute_sqrt[phase, node], capacity[node], count)
                    choices.append((float(score), int(node), edges))
                if not choices:
                    raise RuntimeError(f"No reachable deployment for service {request.service_id}")
                _, node, edges = min(choices)
                compute_sqrt[phase, node] += count * np.sqrt(max(work, 0.0))
                for a, b in edges:
                    link_sqrt[phase, a, b] += count * np.sqrt(max(size, 0.0))
                path.append(node)
                source = node
            if not path:
                raise RuntimeError(f"No stages for service {request.service_id}")
            schedules.append(path)
        return schedules


class GreedyDARPScheme(DARPScheme):
    """DARP v2: greedy placement, frozen Proposed Fast, unchanged KKT."""

    planner_label = "darp_demand_greedy_v2_with_frozen_proposed_fast"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = DemandGreedyPlacement()


class GreedyDPMSScheme(DPMSScheme):
    """DPMS v2: the same greedy deployment and stage-local marginal scheduling."""

    def __init__(self):
        super().__init__()
        self.planner = DemandGreedyPlacement()
        self.scheduler = StageMarginalScheduler()
