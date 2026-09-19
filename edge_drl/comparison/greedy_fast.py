from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from edge_drl.comparison.algorithmic import HybridScheme
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


class GreedyFastScheduler:
    """Choose the cheapest feasible node for each stage as it is reached.

    The local score estimates the request group's transfer and compute delay
    under current loads and work already assigned in this one-second batch.
    Unlike the KKT path scheduler, it neither optimizes a complete chain nor
    uses the KKT marginal-delay formula. Future stages are checked only for
    reachability, so the greedy choice cannot strand a request.
    """

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        if env.scenario is None or env.deployment is None:
            raise RuntimeError("environment must be reset before scheduling")
        if not requests:
            return []

        nodes = env.config.num_edge_nodes
        stages = env.config.max_service_stages
        compute_work = np.zeros((stages, nodes), dtype=np.float64)
        link_data = np.zeros((stages, nodes, nodes), dtype=np.float64)
        capacity = np.asarray(
            [node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64
        )
        capacity *= np.clip(1.0 - 0.75 * env.node_compute_load, 0.10, 1.0)
        bandwidth = env.scenario.bandwidth_mb_s * np.clip(
            1.0 - 0.75 * env.link_load, 0.10, 1.0
        )
        routes: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], float] | None] = {}

        def route(source: int, target: int) -> tuple[tuple[tuple[int, int], ...], float] | None:
            key = (int(source), int(target))
            if key not in routes:
                path = env.shortest_path(*key)
                if path is None:
                    routes[key] = None
                else:
                    edges = tuple(zip(path, path[1:]))
                    propagation = sum(
                        float(env.scenario.propagation_ms[a, b]) / 1000.0
                        for a, b in edges
                    )
                    routes[key] = edges, propagation
            return routes[key]

        schedules: list[list[int]] = []
        for request in requests:
            mask = env.scheduler_candidate_mask(request)
            candidates = [np.flatnonzero(row).tolist() for row in mask]
            if not candidates or any(not row for row in candidates):
                raise RuntimeError(f"no deployed node for service {request.service_id}")

            # The lookahead is a feasibility check, not a future-delay score.
            reachable = candidates[-1]
            feasible_by_stage = [list(row) for row in candidates]
            for stage_id in range(len(candidates) - 2, -1, -1):
                reachable = [
                    node_id
                    for node_id in candidates[stage_id]
                    if any(route(node_id, later) is not None for later in reachable)
                ]
                feasible_by_stage[stage_id] = reachable
            if not feasible_by_stage[0]:
                raise RuntimeError(f"no reachable stage chain for service {request.service_id}")

            count = float(request.request_count)
            previous = int(request.home_node)
            schedule: list[int] = []
            for stage_id, allowed in enumerate(feasible_by_stage):
                size = float(
                    request.input_mb
                    if stage_id == 0
                    else request.stage_output_mb[stage_id - 1]
                )
                work = float(request.stage_compute_gcycles[stage_id])

                def score(node_id: int) -> float:
                    chosen_route = route(previous, node_id)
                    if chosen_route is None:
                        return float("inf")
                    edges, propagation = chosen_route
                    transfer = propagation + sum(
                        (link_data[stage_id, a, b] + count * size)
                        / max(float(bandwidth[a, b]), 1e-9)
                        for a, b in edges
                    )
                    compute = (
                        compute_work[stage_id, node_id] + count * work
                    ) / max(float(capacity[node_id]), 1e-9)
                    return float(transfer + compute)

                node_id = min(allowed, key=lambda candidate: (score(candidate), candidate))
                chosen_route = route(previous, node_id)
                if chosen_route is None:
                    raise RuntimeError(f"no reachable stage chain for service {request.service_id}")
                for a, b in chosen_route[0]:
                    link_data[stage_id, a, b] += count * size
                compute_work[stage_id, node_id] += count * work
                schedule.append(int(node_id))
                previous = int(node_id)
            schedules.append(schedule)
        return schedules


class GreedyFastScheme(HybridScheme):
    """Frozen Proposed Slow deployment with a myopic, batch-aware Fast rule."""

    name = "GreedyFast"

    def __init__(
        self,
        env: EdgeComputingEnv,
        checkpoint_path: str | Path,
        checkpoint_args: dict[str, Any],
        *,
        device: str,
    ) -> None:
        super().__init__(env, checkpoint_path, checkpoint_args, device=device)
        self.scheduler = GreedyFastScheduler()
