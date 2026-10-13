"""Classic heterogeneous list-scheduling baseline for the Fast controller."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from edge_drl.comparison.checkpoint import build_proposed_agent
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.window_feedback import SlowWindowFeedbackMixin


class HEFTScheduler:
    """HEFT-style list scheduling for deployed serial service chains.

    Request groups form independent chain DAGs. Upward ranks use average static
    execution and communication costs. Ready tasks are considered by decreasing
    rank and assigned to the deployed node with the earliest finish time using
    an insertion-based processor calendar. The environment remains responsible
    for authoritative KKT settlement.
    """

    def __init__(self) -> None:
        self._route_cache = {}

    def _route(self, env, source: int, target: int):
        key = (int(source), int(target))
        if key not in self._route_cache:
            route = env.shortest_path(*key)
            if route is None:
                self._route_cache[key] = None
            else:
                edges = tuple(zip(route, route[1:]))
                propagation = sum(float(env.scenario.propagation_ms[a, b]) / 1000.0 for a, b in edges)
                inverse_bandwidth = sum(1.0 / max(float(env.scenario.bandwidth_mb_s[a, b]), 1e-9) for a, b in edges)
                self._route_cache[key] = (propagation, inverse_bandwidth)
        return self._route_cache[key]

    def _communication(self, env, source: int, target: int, data_mb: float, count: float) -> float:
        route = self._route(env, source, target)
        if route is None:
            return float("inf")
        propagation, inverse_bandwidth = route
        return float(propagation + count * max(data_mb, 0.0) * inverse_bandwidth)

    @staticmethod
    def _earliest_slot(intervals, ready: float, duration: float) -> tuple[float, float]:
        start = max(float(ready), 0.0)
        for occupied_start, occupied_finish in intervals:
            if start + duration <= occupied_start + 1e-12:
                break
            start = max(start, occupied_finish)
        return start, start + duration

    def schedule_batch(self, env, requests):
        if env.scenario is None or env.deployment is None:
            raise RuntimeError("environment must be reset before scheduling")
        self._route_cache.clear()
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64)
        candidates = []
        ranks = []
        for request in requests:
            allowed = [list(np.flatnonzero(row)) for row in env.scheduler_candidate_mask(request)]
            if not allowed or any(not nodes for nodes in allowed):
                raise RuntimeError(f"no deployed node for service {request.service_id}")
            candidates.append(allowed)
            stage_ranks = np.zeros(len(allowed), dtype=np.float64)
            for stage in range(len(allowed) - 1, -1, -1):
                count = float(request.request_count)
                work = count * float(request.stage_compute_gcycles[stage])
                average_compute = float(np.mean(work / np.maximum(capacities[allowed[stage]], 1e-9)))
                successor = 0.0
                if stage + 1 < len(allowed):
                    data = float(request.stage_output_mb[stage])
                    communication = [
                        self._communication(env, source, target, data, count)
                        for source in allowed[stage]
                        for target in allowed[stage + 1]
                        if source != target
                    ]
                    finite = [value for value in communication if np.isfinite(value)]
                    successor = (float(np.mean(finite)) if finite else 0.0) + stage_ranks[stage + 1]
                stage_ranks[stage] = average_compute + successor
            ranks.append(stage_ranks)

        calendars = [[] for _ in range(env.config.num_edge_nodes)]
        schedules = [[] for _ in requests]
        finishes = [[] for _ in requests]
        ready = [(-float(ranks[index][0]), index, 0) for index in range(len(requests))]
        import heapq
        heapq.heapify(ready)
        while ready:
            _, request_index, stage = heapq.heappop(ready)
            request = requests[request_index]
            count = float(request.request_count)
            duration_work = count * float(request.stage_compute_gcycles[stage])
            source = request.home_node if stage == 0 else schedules[request_index][stage - 1]
            parent_finish = 0.0 if stage == 0 else finishes[request_index][stage - 1]
            data = float(request.input_mb if stage == 0 else request.stage_output_mb[stage - 1])
            choices = []
            for node in candidates[request_index][stage]:
                communication = self._communication(env, source, node, data, count)
                if not np.isfinite(communication):
                    continue
                execution = duration_work / max(float(capacities[node]), 1e-9)
                start, finish = self._earliest_slot(
                    calendars[node], parent_finish + communication, execution
                )
                choices.append((finish, start, int(node)))
            if not choices:
                raise RuntimeError(f"no reachable HEFT assignment for service {request.service_id}")
            finish, start, node = min(choices)
            calendars[node].append((start, finish))
            calendars[node].sort()
            schedules[request_index].append(node)
            finishes[request_index].append(finish)
            next_stage = stage + 1
            if next_stage < len(candidates[request_index]):
                heapq.heappush(
                    ready,
                    (-float(ranks[request_index][next_stage]), request_index, next_stage),
                )
        return schedules


class HEFTScheme(SlowWindowFeedbackMixin, BaseComparisonScheme):
    """Frozen Proposed Slow policy with HEFT-style Fast scheduling."""

    name = "Heterogeneous Earliest Finish Time Scheduling (HEFT)"

    def __init__(self, env, checkpoint_path: str | Path, checkpoint_args: dict, *, device: str):
        super().__init__()
        self.agent, self.checkpoint_metadata = build_proposed_agent(
            env, checkpoint_path, checkpoint_args, device=device
        )
        self.scheduler = HEFTScheduler()
        self._initialize_window_feedback()

    def maybe_plan(self, env):
        if not env.needs_deployment_update:
            return
        self._before_slow_plan(env)
        self.agent.maybe_update_deployment(env, deterministic=True, record=False)
        self._capture_deployment_usage(env)
        self.diagnostics.planning.append(
            {"time_minute": env.current_time_minute, "planner": "frozen_proposed_slow_heft"}
        )

    def schedule_batch(self, env, requests):
        return self.scheduler.schedule_batch(env, requests)


HEFTFastScheme = HEFTScheme
