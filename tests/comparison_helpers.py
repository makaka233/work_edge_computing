from __future__ import annotations

import numpy as np

from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.trace import ComparisonTrace, generate_comparison_trace
from edge_drl.env.environment import EdgeEnvConfig
from edge_drl.env.scenario import EdgeNode, EdgeScenario, Service, ServiceStage, User


def tiny_scenario() -> EdgeScenario:
    nodes = [
        EdgeNode(node, float(node), 0.0, 64.0, 256.0, 80.0 + 20.0 * node)
        for node in range(6)
    ]
    services = [
        Service(
            0,
            "two-stage",
            (
                ServiceStage(0, 0, 1.0, 2.0, 0.8, 0.08),
                ServiceStage(0, 1, 1.0, 2.0, 1.2, 0.01),
            ),
            0.2,
            0.3,
        ),
        Service(
            1,
            "three-stage",
            (
                ServiceStage(1, 0, 1.0, 2.0, 0.5, 0.05),
                ServiceStage(1, 1, 1.0, 2.0, 0.7, 0.03),
                ServiceStage(1, 2, 1.0, 2.0, 1.0, 0.01),
            ),
            0.15,
            0.3,
        ),
    ]
    users = [
        User(user, float(user % 6), 0.0, user % 6, (0.6, 0.4))
        for user in range(40)
    ]
    adjacency = np.ones((6, 6), dtype=bool)
    bandwidth = np.full((6, 6), 100.0, dtype=np.float64)
    np.fill_diagonal(bandwidth, np.inf)
    propagation = np.full((6, 6), 1.0, dtype=np.float64)
    np.fill_diagonal(propagation, 0.0)
    return EdgeScenario(nodes, users, services, adjacency, bandwidth, propagation)


def tiny_config(episode_minutes: int = 10) -> EdgeEnvConfig:
    return EdgeEnvConfig(
        seed=10,
        physical_seed=10,
        scenario_seed=20,
        num_users=10_000,
        num_edge_nodes=6,
        num_service_types=2,
        max_service_stages=3,
        episode_minutes=episode_minutes,
        deployment_interval_minutes=10,
        active_user_ratio=0.01,
        active_user_request_rate_per_minute=0.1,
        service_resource_fraction=1.0,
        topology_k_nearest=2,
    )


def tiny_trace(logical_steps: int = 600, request_seed: int = 30) -> ComparisonTrace:
    return generate_comparison_trace(
        scenario=tiny_scenario(),
        logical_steps=logical_steps,
        requests_per_minute=30.0,
        physical_seed=10,
        demand_seed=20,
        request_seed=request_seed,
        task_compute_scale=1.0,
        task_data_scale=1.0,
    )


def tiny_replay_env() -> TraceReplayEnv:
    return TraceReplayEnv(tiny_config(), tiny_scenario(), tiny_trace())
