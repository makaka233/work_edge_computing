"""Non-optimizing equal-share continuous resource allocation baselines."""
from __future__ import annotations

import numpy as np

from edge_drl.allocators.kkt import ComputeDemand, LinkDemand


def allocate_compute_equal_share(
    demands: list[ComputeDemand],
    node_capacities: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Give every individual request an equal share of its node's capacity."""

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
        population = float(sum(demand.multiplicity for demand in grouped))
        if population <= 0.0:
            raise ValueError("compute demand multiplicity must be positive")
        for demand in grouped:
            group_rate = capacity * float(demand.multiplicity) / population
            delay = demand.compute_gcycles * demand.multiplicity / group_rate
            allocations[demand.demand_id] = group_rate
            delays[demand.demand_id] = delay
            total_delay += delay * demand.multiplicity
    return allocations, delays, total_delay


def allocate_link_equal_share(
    demands: list[LinkDemand],
    link_capacities: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Give every individual active flow an equal share of link bandwidth."""

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
        population = float(sum(demand.multiplicity for demand in grouped))
        if population <= 0.0:
            raise ValueError("link demand multiplicity must be positive")
        for demand in grouped:
            group_rate = capacity * float(demand.multiplicity) / population
            delay = demand.data_mb * demand.multiplicity / group_rate
            allocations[demand.demand_id] = group_rate
            delays[demand.demand_id] = delay
            total_delay += delay * demand.multiplicity
    return allocations, delays, total_delay
