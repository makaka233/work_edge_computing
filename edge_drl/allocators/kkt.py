from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComputeDemand:
    demand_id: str
    node_id: int
    compute_gcycles: float
    multiplicity: float = 1.0
    # Causal position in a serial service chain.  The bare KKT allocator still
    # treats every demand passed to one call as concurrent; the environment
    # uses this field to make one call per chain phase.
    serial_phase: int = 0


@dataclass(frozen=True)
class LinkDemand:
    demand_id: str
    src_node: int
    dst_node: int
    data_mb: float
    multiplicity: float = 1.0
    serial_phase: int = 0


def allocate_compute_kkt(
    demands: list[ComputeDemand],
    node_capacities: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Allocate node compute by the KKT sqrt(C) rule.

    Returns per-demand allocated compute rate, per-demand delay, and total delay.
    Capacities are in Gcycles/s, demands are in Gcycles, delays are seconds.
    """

    allocations: dict[str, float] = {}
    delays: dict[str, float] = {}
    total_delay = 0.0
    by_node: dict[int, list[ComputeDemand]] = {}
    for demand in demands:
        if demand.compute_gcycles <= 0:
            allocations[demand.demand_id] = 0.0
            delays[demand.demand_id] = 0.0
            continue
        by_node.setdefault(demand.node_id, []).append(demand)

    for node_id, node_demands in by_node.items():
        capacity = float(node_capacities[node_id])
        if capacity <= 0:
            raise ValueError(f"node {node_id} has non-positive compute capacity")
        weighted_sqrt_loads = np.asarray(
            [d.multiplicity * np.sqrt(d.compute_gcycles) for d in node_demands],
            dtype=np.float64,
        )
        weighted_sqrt_sum = float(weighted_sqrt_loads.sum())
        for demand, weighted_sqrt_load in zip(node_demands, weighted_sqrt_loads):
            group_rate = capacity * float(weighted_sqrt_load) / weighted_sqrt_sum
            delay = demand.compute_gcycles * demand.multiplicity / group_rate
            allocations[demand.demand_id] = group_rate
            delays[demand.demand_id] = delay
            total_delay += delay * demand.multiplicity

    return allocations, delays, total_delay


def allocate_link_kkt(
    demands: list[LinkDemand],
    link_capacities: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Allocate link bandwidth by the KKT sqrt(D) rule."""

    allocations: dict[str, float] = {}
    delays: dict[str, float] = {}
    total_delay = 0.0
    by_link: dict[tuple[int, int], list[LinkDemand]] = {}
    for demand in demands:
        if demand.src_node == demand.dst_node or demand.data_mb <= 0:
            allocations[demand.demand_id] = 0.0
            delays[demand.demand_id] = 0.0
            continue
        key = (demand.src_node, demand.dst_node)
        by_link.setdefault(key, []).append(demand)

    for (src, dst), link_demands in by_link.items():
        capacity = float(link_capacities[src, dst])
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError(f"link ({src}, {dst}) has non-positive bandwidth")
        weighted_sqrt_loads = np.asarray(
            [d.multiplicity * np.sqrt(d.data_mb) for d in link_demands],
            dtype=np.float64,
        )
        weighted_sqrt_sum = float(weighted_sqrt_loads.sum())
        for demand, weighted_sqrt_load in zip(link_demands, weighted_sqrt_loads):
            group_rate = capacity * float(weighted_sqrt_load) / weighted_sqrt_sum
            delay = demand.data_mb * demand.multiplicity / group_rate
            allocations[demand.demand_id] = group_rate
            delays[demand.demand_id] = delay
            total_delay += delay * demand.multiplicity

    return allocations, delays, total_delay
