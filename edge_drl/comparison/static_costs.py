from __future__ import annotations

import numpy as np

from edge_drl.env.environment import EdgeComputingEnv


def all_pairs_static_delay(env: EdgeComputingEnv, data_mb: float = 0.0) -> np.ndarray:
    env._require_ready()
    assert env.scenario is not None
    count = env.config.num_edge_nodes
    result = np.full((count, count), np.inf, dtype=np.float64)
    for src in range(count):
        for dst in range(count):
            path = env.shortest_path(src, dst)
            if path is None:
                continue
            delay = 0.0
            for left, right in zip(path, path[1:]):
                delay += float(env.scenario.propagation_ms[left, right]) / 1000.0
                delay += data_mb / max(float(env.scenario.bandwidth_mb_s[left, right]), 1e-12)
            result[src, dst] = delay
    return result


def expected_home_service_demand(env: EdgeComputingEnv) -> np.ndarray:
    """Expected requests/s by home node and service, without future trace access."""

    env._require_ready()
    assert env.scenario is not None
    demand = np.zeros((env.config.num_edge_nodes, env.config.num_service_types), dtype=np.float64)
    for user in env.scenario.users:
        demand[user.home_node] += np.asarray(user.service_weights, dtype=np.float64)
    rate_per_user_s = (
        env.config.active_user_ratio
        * env.config.active_user_request_rate_per_minute
        * env.config.traffic_scale
        * env.config.demand_load_multiplier
        / 60.0
    )
    if env.config.mean_requests_per_minute is not None:
        totals = demand.sum(axis=0)
        demand *= env.config.mean_requests_per_minute * env.config.demand_load_multiplier / 60.0 / max(float(totals.sum()), 1e-12)
    else:
        demand *= rate_per_user_s
    return demand
