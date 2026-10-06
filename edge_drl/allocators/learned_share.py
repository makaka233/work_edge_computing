"""Feasible resource normalization for end-to-end learned priorities.

The policy chooses one positive priority for every scheduled service stage.
This module only normalizes those choices to the available node/link capacity;
it does not solve an optimization problem and does not inspect workload sizes.
"""
from __future__ import annotations

import numpy as np

from edge_drl.allocators.kkt import ComputeDemand, LinkDemand


def _positive_weight(value: float) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("learned allocation weights must be finite and positive")
    return weight


def allocate_compute_learned_share(
    demands: list[ComputeDemand],
    node_capacities: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Allocate node capacity in proportion to policy priority and population."""

    allocations: dict[str, float] = {}
    delays: dict[str, float] = {}
    total_delay = 0.0
    by_node: dict[int, list[ComputeDemand]] = {}
    for demand in demands:
        if demand.compute_gcycles <= 0.0:
            allocations[demand.demand_id] = 0.0
            delays[demand.demand_id] = 0.0
        else:
            by_node.setdefault(demand.node_id, []).append(demand)
    for node_id, grouped in by_node.items():
        capacity = float(node_capacities[node_id])
        if capacity <= 0.0:
            raise ValueError(f"node {node_id} has non-positive compute capacity")
        priorities = np.asarray(
            [d.multiplicity * _positive_weight(d.allocation_weight) for d in grouped],
            dtype=np.float64,
        )
        denominator = float(priorities.sum())
        if denominator <= 0.0:
            raise ValueError("compute priority mass must be positive")
        for demand, priority in zip(grouped, priorities):
            group_rate = capacity * float(priority) / denominator
            delay = demand.compute_gcycles * demand.multiplicity / group_rate
            allocations[demand.demand_id] = group_rate
            delays[demand.demand_id] = delay
            total_delay += delay * demand.multiplicity
    return allocations, delays, total_delay


def allocate_link_learned_share(
    demands: list[LinkDemand],
    link_capacities: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Allocate link capacity in proportion to policy priority and population."""

    allocations: dict[str, float] = {}
    delays: dict[str, float] = {}
    total_delay = 0.0
    by_link: dict[tuple[int, int], list[LinkDemand]] = {}
    for demand in demands:
        if demand.src_node == demand.dst_node or demand.data_mb <= 0.0:
            allocations[demand.demand_id] = 0.0
            delays[demand.demand_id] = 0.0
        else:
            by_link.setdefault((demand.src_node, demand.dst_node), []).append(demand)
    for (src, dst), grouped in by_link.items():
        capacity = float(link_capacities[src, dst])
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError(f"link ({src}, {dst}) has non-positive bandwidth")
        priorities = np.asarray(
            [d.multiplicity * _positive_weight(d.allocation_weight) for d in grouped],
            dtype=np.float64,
        )
        denominator = float(priorities.sum())
        if denominator <= 0.0:
            raise ValueError("link priority mass must be positive")
        for demand, priority in zip(grouped, priorities):
            group_rate = capacity * float(priority) / denominator
            delay = demand.data_mb * demand.multiplicity / group_rate
            allocations[demand.demand_id] = group_rate
            delays[demand.demand_id] = delay
            total_delay += delay * demand.multiplicity
    return allocations, delays, total_delay
