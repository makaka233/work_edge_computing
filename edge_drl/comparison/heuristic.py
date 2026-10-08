"""Transparent demand/locality placement and online stage-completion heuristics.

These are project-specific heuristics, not reproductions of PRP or HEFT.
They never read a learned checkpoint, future trace, or evaluation outcome.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from edge_drl.comparison.algorithmic import DemandAwareMILPPlacement, CausalKKTPathScheduler
from edge_drl.comparison.checkpoint import build_proposed_agent
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


class DemandAwareReplicaPlacement:
    """Mandatory coverage followed by demand-proportional replica allocation.

Replica priority is expected work divided by current pooled compute capacity
and resource footprint. Locations minimize demand-weighted access plus local
execution time. All feasible nodes participate; there is no replica cap.
"""

    def plan(self, env):
        started = time.perf_counter()
        assert env.scenario is not None and env.deployment is not None
        stages = [(service, stage) for service in env.scenario.services for stage in service.stages]
        n = env.config.num_edge_nodes
        demand = DemandAwareMILPPlacement._demand(env)
        propagation, inverse_bw = DemandAwareMILPPlacement._transfer_matrices(env)
        capacity = np.array([node.compute_gcycles_per_s for node in env.scenario.nodes])
        capacity *= np.maximum(1.0 - 0.75 * env.node_compute_load, 0.10)
        memory = env.service_memory_capacities().copy()
        storage = env.service_storage_capacities().copy()
        costs = []
        work = []
        for service, stage in stages:
            weights = demand[:, service.service_id]
            mass = float(weights.sum())
            # A locality estimate from ingress demand, not a whole-chain optimizer.
            volume = service.input_mb_mean if stage.stage_id == 0 else service.stages[stage.stage_id - 1].output_mb_mean
            access = propagation + volume * env.config.task_data_scale * inverse_bw
            active = weights > 0
            weighted_access = weights[active] @ access[active] if active.any() else np.zeros(n)
            costs.append(weighted_access + mass * stage.compute_gcycles_mean * env.config.task_compute_scale / capacity)
            work.append(mass * stage.compute_gcycles_mean * env.config.task_compute_scale)
        costs = np.asarray(costs)
        # Solve coverage jointly so early popular stages cannot starve later ones.
        variables = len(stages) * n
        matrix = np.zeros((len(stages) + 2 * n, variables))
        for i, (_, stage) in enumerate(stages):
            matrix[i, i * n:(i + 1) * n] = 1.0
            for node in range(n):
                matrix[len(stages) + node, i * n + node] = stage.memory_gb
                matrix[len(stages) + n + node, i * n + node] = stage.storage_gb
        finite = np.isfinite(costs).reshape(-1)
        result = milp(
            c=np.where(np.isfinite(costs), costs, 0.0).reshape(-1),
            integrality=np.ones(variables), bounds=Bounds(0, finite.astype(float)),
            constraints=LinearConstraint(matrix, np.r_[np.ones(len(stages)), np.zeros(2*n)],
                                         np.r_[np.ones(len(stages)), memory, storage]),
            options={"time_limit": 5.0, "mip_rel_gap": 0.01},
        )
        if result.x is None:
            raise RuntimeError(f"Popularity coverage infeasible: {result.message}")
        deployment = np.zeros_like(env.deployment, dtype=bool)
        for i, (service, stage) in enumerate(stages):
            selected = result.x[i*n:(i+1)*n] > 0.5
            deployment[service.service_id, stage.stage_id] = selected
            memory -= selected * stage.memory_gb
            storage -= selected * stage.storage_gb
        additions = 0
        while True:
            options = []
            for i, (service, stage) in enumerate(stages):
                selected = deployment[service.service_id, stage.stage_id]
                feasible = np.flatnonzero(~selected & (memory + 1e-9 >= stage.memory_gb)
                                         & (storage + 1e-9 >= stage.storage_gb) & np.isfinite(costs[i]))
                if not feasible.size or work[i] <= 0:
                    continue
                node = min(feasible, key=lambda node: (costs[i, node], int(node)))
                footprint = (stage.memory_gb / max(env.service_memory_capacities()[node], 1e-9)
                             + stage.storage_gb / max(env.service_storage_capacities()[node], 1e-9))
                priority = work[i] / max(float(capacity[selected].sum()) * footprint, 1e-9)
                options.append((-priority, i, int(node)))
            if not options:
                break
            _, i, node = min(options)
            service, stage = stages[i]
            deployment[service.service_id, stage.stage_id, node] = True
            memory[node] -= stage.memory_gb
            storage[node] -= stage.storage_gb
            additions += 1
        valid, reason = env.check_deployment_feasible(deployment)
        if not valid:
            raise RuntimeError(f"Popularity placement invalid: {reason}")
        return deployment, {"planner": "popularity_capacity_locality", "solver_status": str(result.message),
                            "replicas": int(deployment.sum()), "added_replicas": additions,
                            "planning_time_s": time.perf_counter() - started}


class PopularityFirstFitPlacement:
    """Demand-only replica priority and fixed ascending-node First-Fit packing.

    Coverage comes first. If greedy coverage fragments capacity, a binary
    feasibility repair finds one replica per stage without latency objectives.
    Extra replicas use demand divided by the current replica count. There is
    no artificial replica cap and no compute, bandwidth or locality score.
    """

    planner_name = "popularity_first_fit"

    def _service_priorities(self, env):
        return DemandAwareMILPPlacement._demand(env).sum(axis=0)

    def plan(self, env):
        started = time.perf_counter()
        assert env.scenario is not None and env.deployment is not None
        stages = [(service, stage) for service in env.scenario.services for stage in service.stages]
        n = env.config.num_edge_nodes
        demand = self._service_priorities(env)
        order = sorted(range(len(stages)), key=lambda i: (
            -float(demand[stages[i][0].service_id]),
            stages[i][0].service_id, stages[i][1].stage_id,
        ))
        memory_cap = env.service_memory_capacities().copy()
        storage_cap = env.service_storage_capacities().copy()
        memory, storage = memory_cap.copy(), storage_cap.copy()
        deployment = np.zeros_like(env.deployment, dtype=bool)

        def first_fit(i):
            service, stage = stages[i]
            feasible = np.flatnonzero(
                ~deployment[service.service_id, stage.stage_id]
                & (memory + 1e-9 >= stage.memory_gb)
                & (storage + 1e-9 >= stage.storage_gb)
            )
            return int(feasible[0]) if feasible.size else None

        def place(i, node):
            service, stage = stages[i]
            deployment[service.service_id, stage.stage_id, node] = True
            memory[node] -= stage.memory_gb
            storage[node] -= stage.storage_gb

        repaired = False
        for i in order:
            node = first_fit(i)
            if node is None:
                # Pure coverage feasibility repair; not a latency optimizer.
                matrix = np.zeros((len(stages) + 2 * n, len(stages) * n))
                for j, (_, stage) in enumerate(stages):
                    matrix[j, j*n:(j+1)*n] = 1
                    for node_id in range(n):
                        matrix[len(stages)+node_id, j*n+node_id] = stage.memory_gb
                        matrix[len(stages)+n+node_id, j*n+node_id] = stage.storage_gb
                result = milp(
                    c=np.zeros(len(stages)*n), integrality=np.ones(len(stages)*n),
                    bounds=Bounds(0, 1),
                    constraints=LinearConstraint(
                        matrix, np.r_[np.ones(len(stages)), np.zeros(2*n)],
                        np.r_[np.ones(len(stages)), memory_cap, storage_cap],
                    ),
                    options={"time_limit": 5.0},
                )
                if result.x is None:
                    raise RuntimeError(f"First-Fit coverage repair failed: {result.message}")
                deployment.fill(False)
                memory[:], storage[:] = memory_cap, storage_cap
                for j in range(len(stages)):
                    for node_id in np.flatnonzero(result.x[j*n:(j+1)*n] > 0.5):
                        place(j, int(node_id))
                repaired = True
                break
            place(i, node)

        additions = 0
        while True:
            choices = []
            for i in order:
                service, stage = stages[i]
                node = first_fit(i)
                heat = float(demand[service.service_id])
                if node is not None and heat > 0:
                    count = int(deployment[service.service_id, stage.stage_id].sum())
                    choices.append((-heat / max(count, 1), i, node))
            if not choices:
                break
            _, i, node = min(choices)
            place(i, node)
            additions += 1
        valid, reason = env.check_deployment_feasible(deployment)
        if not valid:
            raise RuntimeError(f"First-Fit deployment invalid: {reason}")
        return deployment, {
            "planner": self.planner_name, "coverage_repaired": repaired,
            "replicas": int(deployment.sum()), "added_replicas": additions,
            "planning_time_s": time.perf_counter() - started,
        }


class UniformFirstFitPlacement(PopularityFirstFitPlacement):
    """Equal stage replica counts, with fixed-order First-Fit resource packing.

    Constant service priorities make each next replica go to the feasible
    stage with the fewest replicas, breaking ties by service/stage order.
    This planner does not read demand, compute capacities or network costs.
    """

    planner_name = "uniform_first_fit"

    def _service_priorities(self, env):
        return np.ones(max(service.service_id for service in env.scenario.services) + 1)


class StageCompletionScheduler(CausalKKTPathScheduler):
    """Myopic serial-stage earliest completion using an online work ledger.

The score is transfer + (reserved work + new work)/effective capacity.
This is a work-based prediction, not the simulator's KKT marginal cost and
not the original non-preemptive HEFT processor-time-table model.
"""

    def schedule_batch(self, env, requests):
        assert env.scenario is not None and env.deployment is not None
        self._paths.clear()
        n, stages = env.config.num_edge_nodes, env.config.max_service_stages
        compute_work = np.zeros((stages, n))
        link_work = np.zeros((stages, n, n))
        capacity = np.array([node.compute_gcycles_per_s for node in env.scenario.nodes])
        capacity *= np.maximum(1.0 - 0.75 * env.node_compute_load, 0.10)
        bandwidth = env.scenario.bandwidth_mb_s * np.maximum(1.0 - 0.75 * env.link_load, 0.10)
        schedules = []
        for request in requests:
            candidates = [list(np.flatnonzero(row)) for row in env.scheduler_candidate_mask(request)]
            # Reachability pruning only: do not make a locally cheap dead end.
            for stage in range(len(candidates) - 2, -1, -1):
                candidates[stage] = [node for node in candidates[stage]
                                     if any(self._path(env, node, dst) is not None for dst in candidates[stage+1])]
            path = []
            source = request.home_node
            count = float(request.request_count)
            for stage, allowed in enumerate(candidates):
                size = request.input_mb if stage == 0 else request.stage_output_mb[stage-1]
                work = count * request.stage_compute_gcycles[stage]
                choices = []
                for node in allowed:
                    route = self._path(env, source, node)
                    if route is None:
                        continue
                    edges, propagation = route
                    transfer = propagation + sum((link_work[stage, a, b] + count * size)
                                                 / max(bandwidth[a, b], 1e-9) for a, b in edges) if size > 0 else 0.0
                    score = transfer + (compute_work[stage, node] + work) / max(capacity[node], 1e-9)
                    choices.append((float(score), int(node), edges))
                if not choices:
                    raise RuntimeError(f"No reachable deployment for service {request.service_id}")
                _, node, edges = min(choices)
                compute_work[stage, node] += work
                for a, b in edges:
                    link_work[stage, a, b] += count * size
                path.append(node)
                source = node
            schedules.append(path)
        return schedules


class PopularityEFTScheme(BaseComparisonScheme):
    name = "PopularityEFT"

    def __init__(self):
        super().__init__()
        self.planner = DemandAwareReplicaPlacement()
        self.scheduler = StageCompletionScheduler()

    def maybe_plan(self, env):
        if env.needs_deployment_update:
            deployment, details = self.planner.plan(env)
            env.apply_deployment(deployment)
            self.diagnostics.planning.append({"time_minute": env.current_time_minute, **details})

    def schedule_batch(self, env, requests):
        return self.scheduler.schedule_batch(env, requests)


class NearestReplicaScheduler(CausalKKTPathScheduler):
    """Route each serial stage to its nearest reachable deployed replica.

    Distance is physical-path propagation delay. Decisions use neither current
    compute/link load nor other requests in the batch. This is a simple
    locality baseline, while still enforcing deployment and reachability.
    """

    def schedule_batch(self, env, requests):
        assert env.scenario is not None and env.deployment is not None
        self._paths.clear()
        schedules = []
        for request in requests:
            candidates = [list(np.flatnonzero(row)) for row in env.scheduler_candidate_mask(request)]
            for stage in range(len(candidates) - 2, -1, -1):
                candidates[stage] = [
                    node
                    for node in candidates[stage]
                    if any(self._path(env, node, target) is not None for target in candidates[stage + 1])
                ]
            source = request.home_node
            schedule = []
            for allowed in candidates:
                reachable = []
                for node in allowed:
                    route = self._path(env, source, node)
                    if route is not None:
                        reachable.append((float(route[1]), int(node)))
                if not reachable:
                    raise RuntimeError(f"No reachable deployment for service {request.service_id}")
                _, selected = min(reachable)
                schedule.append(selected)
                source = selected
            schedules.append(schedule)
        return schedules


class PopularityNearestScheme(BaseComparisonScheme):
    """Popularity/resource placement with static nearest-replica scheduling."""

    name = "PopularityNearest"

    def __init__(self):
        super().__init__()
        self.planner = DemandAwareReplicaPlacement()
        self.scheduler = NearestReplicaScheduler()

    def maybe_plan(self, env):
        if env.needs_deployment_update:
            deployment, details = self.planner.plan(env)
            env.apply_deployment(deployment)
            self.diagnostics.planning.append(
                {"time_minute": env.current_time_minute, **details, "planner": "popularity_nearest"}
            )

    def schedule_batch(self, env, requests):
        return self.scheduler.schedule_batch(env, requests)


class DARPScheme(BaseComparisonScheme):
    """Fixed algorithmic placement paired with the frozen Proposed Fast PPO.

    This is a deployment-only component comparison.  It changes exactly one
    decision layer relative to Proposed: Slow PPO is replaced by
    :class:`DemandAwareReplicaPlacement`. Scheduling and continuous KKT settlement
    remain identical to Proposed.
    """

    name = "Demand-Aware Replica Placement (DARP)"
    planner_label = "darp_with_frozen_proposed_fast"

    def __init__(
        self,
        env: EdgeComputingEnv,
        checkpoint_path,
        checkpoint_args,
        *,
        device: str,
        fast_deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.fast_deterministic = bool(fast_deterministic)
        self.planner = DemandAwareReplicaPlacement()
        self.agent, self.checkpoint_metadata = build_proposed_agent(
            env, checkpoint_path, checkpoint_args, device=device
        )

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if env.needs_deployment_update:
            deployment, details = self.planner.plan(env)
            env.apply_deployment(deployment)
            self.diagnostics.planning.append(
                {
                    "time_minute": env.current_time_minute,
                    **details,
                    "planner": self.planner_label,
                }
            )

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        if requests is not env.current_requests:
            env.current_requests = requests
            env.current_request = requests[0] if requests else None
        return self.agent.fast_agent.schedule_batch(
            env, requests, deterministic=self.fast_deterministic, record=False
        )


class FitDeployScheme(DARPScheme):
    """First-Fit placement with the same frozen Fast PPO and KKT settlement."""

    name = "FitDeploy"
    planner_label = "popularity_first_fit_with_frozen_proposed_fast"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = PopularityFirstFitPlacement()


class UniformFirstFitDeployScheme(DARPScheme):
    """Uniform First-Fit deployment with the frozen Proposed Fast PPO."""

    name = "UniformFirstFitDeploy"
    planner_label = "uniform_first_fit_with_frozen_proposed_fast"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = UniformFirstFitPlacement()


PopularityDeployScheme = DARPScheme
PopularityPlacement = DemandAwareReplicaPlacement
