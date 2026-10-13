"""Exact feature acceleration; caches are observations, not action candidates.

All physical node pairs are retained. No service execution path is preselected.
Static graph snapshots detect in-place scenario changes; loads stay live.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RoutePlan:
    edges: np.ndarray
    valid_hops: np.ndarray
    reachable: np.ndarray
    propagation: np.ndarray

    def sum_edges(self, values: np.ndarray) -> np.ndarray:
        costs = np.where(self.valid_hops, values.reshape(-1)[self.edges], 0.0).sum(axis=-1)
        return np.where(self.reachable, costs, np.inf)


def graph_cache(env):
    scenario = env.scenario
    assert scenario is not None
    arrays = (scenario.adjacency, scenario.bandwidth_mb_s, scenario.propagation_ms)
    cached = getattr(env, "_rollout_graph_cache", None)
    if cached is not None and cached["scenario"] is scenario and all(
        np.array_equal(current, saved) for current, saved in zip(arrays, cached["snapshots"])
    ):
        return cached
    if cached is not None:
        # The router's own cache must also respect changed static edge costs.
        env._route_cache.clear()
    adjacency, bandwidth, propagation = arrays
    max_bandwidth = float(np.nanmax(np.where(np.isfinite(bandwidth), bandwidth, 0)))
    max_propagation = max(float(np.where(np.isfinite(propagation), propagation, 0).max()), 1e-9)
    features = np.stack([
        adjacency.astype(np.float64),
        np.where(np.isfinite(bandwidth), bandwidth, max_bandwidth) / max(max_bandwidth, 1e-9),
        np.where(np.isfinite(propagation), propagation, max_propagation) / max_propagation,
    ], axis=-1)
    cached = {"scenario": scenario, "snapshots": tuple(array.copy() for array in arrays),
              "edge_features": features, "routes": None}
    env._rollout_graph_cache = cached
    return cached


def edge_features(env, *, include_load=False):
    static = graph_cache(env)["edge_features"]
    if not include_load:
        return static.reshape(-1).tolist()
    load = np.clip(np.where(np.isfinite(env.link_load), env.link_load, 0), 0, 1)
    return np.concatenate([static, load[..., None]], axis=-1).reshape(-1).tolist()


def route_plan(env) -> RoutePlan:
    cache = graph_cache(env)
    if cache["routes"] is not None:
        return cache["routes"]
    n = env.config.num_edge_nodes
    paths = [[env.shortest_path(src, dst) for dst in range(n)] for src in range(n)]
    hops = max(1, max((len(path) - 1 for row in paths for path in row if path is not None), default=0))
    edges = np.zeros((n, n, hops), dtype=np.int64)
    valid = np.zeros_like(edges, dtype=bool)
    reachable = np.zeros((n, n), dtype=bool)
    for src, row in enumerate(paths):
        for dst, path in enumerate(row):
            if path is None:
                continue
            reachable[src, dst] = True
            for hop, (left, right) in enumerate(zip(path, path[1:])):
                edges[src, dst, hop] = left * n + right
                valid[src, dst, hop] = True
    plan = RoutePlan(edges, valid, reachable, np.zeros((n, n)))
    plan.propagation = plan.sum_edges(env.scenario.propagation_ms / 1000)
    cache["routes"] = plan
    return plan


def nic_features(env):
    capacities = env.node_nic_capacities()
    finite = np.isfinite(capacities)
    result = np.zeros((len(capacities), 2))
    if not finite.any():
        return result
    bandwidth = np.where(np.isfinite(env.scenario.bandwidth_mb_s), env.scenario.bandwidth_mb_s, 0)
    offered = env.link_load * bandwidth
    pressure = np.maximum(offered.sum(axis=1), offered.sum(axis=0))
    result[finite, 0] = capacities[finite] / max(float(capacities[finite].max()), 1e-9)
    result[finite, 1] = np.tanh(pressure[finite] / np.maximum(capacities[finite], 1e-9))
    return result
