from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EdgeNode:
    node_id: int
    x_km: float
    y_km: float
    memory_gb: float
    storage_gb: float
    compute_gcycles_per_s: float


@dataclass(frozen=True)
class ServiceStage:
    service_id: int
    stage_id: int
    memory_gb: float
    storage_gb: float
    compute_gcycles_mean: float
    output_mb_mean: float


@dataclass(frozen=True)
class Service:
    service_id: int
    name: str
    stages: tuple[ServiceStage, ...]
    input_mb_mean: float
    deadline_s_mean: float


@dataclass(frozen=True)
class User:
    user_id: int
    x_km: float
    y_km: float
    home_node: int
    service_weights: tuple[float, ...]


@dataclass(frozen=True)
class TaskRequest:
    request_id: int
    arrival_minute: float
    request_count: int
    user_id: int
    home_node: int
    service_id: int
    input_mb: float
    stage_compute_gcycles: tuple[float, ...]
    stage_output_mb: tuple[float, ...]
    deadline_s: float


@dataclass
class EdgeScenario:
    nodes: list[EdgeNode]
    users: list[User]
    services: list[Service]
    adjacency: np.ndarray
    bandwidth_mb_s: np.ndarray
    propagation_ms: np.ndarray


def generate_realistic_scenario(
    *,
    rng: np.random.Generator,
    demand_rng: np.random.Generator | None = None,
    num_users: int,
    num_edge_nodes: int,
    num_service_types: int,
    max_service_stages: int,
    node_compute_capacity_scale: float = 1.0,
    wired_link_bandwidth_scale: float = 1.0,
    topology_k_nearest: int = 6,
    deadline_scale: float = 1.0,
) -> EdgeScenario:
    """Generate a city-scale MEC scenario with realistic heterogeneity.

    The generator is synthetic but grounded in common MEC assumptions: clustered
    population density, heterogeneous regional edge nodes, sparse metro links,
    and a small catalogue of staged latency-sensitive services.
    """

    if not 10_000 <= num_users <= 15_000:
        raise ValueError("num_users must be in [10000, 15000].")
    if max_service_stages > 3:
        raise ValueError("max_service_stages must be <= 3.")
    if not 1 <= topology_k_nearest < num_edge_nodes:
        raise ValueError("topology_k_nearest must be in [1, num_edge_nodes).")
    if deadline_scale <= 0.0:
        raise ValueError("deadline_scale must be positive.")

    city_width_km = 36.0
    city_height_km = 28.0
    centers = np.array(
        [
            [18.0, 14.0],  # central business district
            [10.0, 19.0],  # residential cluster
            [26.0, 18.0],  # university / business park
            [12.0, 8.0],   # transport hub
            [29.0, 7.0],   # industrial district
        ],
        dtype=np.float64,
    )
    center_weights = np.array([0.30, 0.24, 0.18, 0.16, 0.12], dtype=np.float64)

    demand_rng = rng if demand_rng is None else demand_rng

    node_centers = rng.choice(len(centers), size=num_edge_nodes, p=center_weights)
    node_xy = centers[node_centers] + rng.normal(0.0, [3.2, 2.4], size=(num_edge_nodes, 2))
    node_xy[:, 0] = np.clip(node_xy[:, 0], 0.5, city_width_km - 0.5)
    node_xy[:, 1] = np.clip(node_xy[:, 1], 0.5, city_height_km - 0.5)

    nodes: list[EdgeNode] = []
    for node_id, (x_km, y_km) in enumerate(node_xy):
        tier = rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15])
        memory_gb = [64, 128, 256][tier] * rng.uniform(0.85, 1.20)
        storage_gb = [512, 1024, 2048][tier] * rng.uniform(0.85, 1.25)
        compute = [96, 192, 384][tier] * rng.uniform(0.80, 1.25) * node_compute_capacity_scale
        nodes.append(
            EdgeNode(
                node_id=node_id,
                x_km=float(x_km),
                y_km=float(y_km),
                memory_gb=float(memory_gb),
                storage_gb=float(storage_gb),
                compute_gcycles_per_s=float(compute),
            )
        )

    user_centers = demand_rng.choice(len(centers), size=num_users, p=center_weights)
    user_xy = centers[user_centers] + demand_rng.normal(0.0, [4.0, 3.0], size=(num_users, 2))
    user_xy[:, 0] = np.clip(user_xy[:, 0], 0.0, city_width_km)
    user_xy[:, 1] = np.clip(user_xy[:, 1], 0.0, city_height_km)
    distances = _pairwise_distance(user_xy, node_xy)
    home_nodes = distances.argmin(axis=1)

    services = _generate_services(
        rng=rng,
        num_service_types=num_service_types,
        max_service_stages=max_service_stages,
        deadline_scale=deadline_scale,
    )

    base_service_popularity = demand_rng.dirichlet(np.linspace(2.2, 0.8, num_service_types))
    users: list[User] = []
    for user_id, (x_km, y_km) in enumerate(user_xy):
        local_bias = demand_rng.dirichlet(np.ones(num_service_types) * 4.0)
        weights = 0.70 * base_service_popularity + 0.30 * local_bias
        weights = weights / weights.sum()
        users.append(
            User(
                user_id=user_id,
                x_km=float(x_km),
                y_km=float(y_km),
                home_node=int(home_nodes[user_id]),
                service_weights=tuple(float(v) for v in weights),
            )
        )

    node_dist = _pairwise_distance(node_xy, node_xy)
    adjacency = _build_sparse_metro_adjacency(node_dist, topology_k_nearest)

    bandwidth = np.zeros((num_edge_nodes, num_edge_nodes), dtype=np.float64)
    propagation = np.full((num_edge_nodes, num_edge_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(bandwidth, np.inf)
    np.fill_diagonal(propagation, 0.0)
    for i in range(num_edge_nodes):
        for j in range(i + 1, num_edge_nodes):
            if not adjacency[i, j]:
                continue
            distance = max(node_dist[i, j], 0.2)
            link_class = rng.choice(["bottleneck", "metro", "backbone"], p=[0.18, 0.62, 0.20])
            if link_class == "bottleneck":
                raw_bandwidth = rng.uniform(25.0, 90.0)
            elif link_class == "metro":
                raw_bandwidth = rng.uniform(120.0, 650.0)
            else:
                raw_bandwidth = rng.uniform(750.0, 1800.0)
            link_bandwidth = raw_bandwidth / (1.0 + 0.06 * distance) * wired_link_bandwidth_scale
            link_propagation = 0.35 * distance + rng.uniform(0.2, 1.5)
            bandwidth[i, j] = link_bandwidth
            bandwidth[j, i] = link_bandwidth
            propagation[i, j] = link_propagation
            propagation[j, i] = link_propagation

    return EdgeScenario(
        nodes=nodes,
        users=users,
        services=services,
        adjacency=adjacency,
        bandwidth_mb_s=bandwidth,
        propagation_ms=propagation,
    )


def _build_sparse_metro_adjacency(node_dist: np.ndarray, k_nearest: int) -> np.ndarray:
    """Build a connected sparse metro graph from geographic neighbors.

    Direct stage transfers are allowed only on physical edges, so the graph is
    explicitly connected after symmetric k-nearest-neighbor construction.  The
    repair adds the shortest bridge between disconnected components and keeps
    the topology sparse instead of silently reverting to a complete graph.
    """

    num_nodes = int(node_dist.shape[0])
    adjacency = np.eye(num_nodes, dtype=bool)
    for node_id in range(num_nodes):
        nearest = np.argsort(node_dist[node_id])[1 : k_nearest + 1]
        adjacency[node_id, nearest] = True
        adjacency[nearest, node_id] = True

    while True:
        components: list[list[int]] = []
        unseen = set(range(num_nodes))
        while unseen:
            root = unseen.pop()
            stack = [root]
            component = [root]
            while stack:
                current = stack.pop()
                neighbors = set(np.flatnonzero(adjacency[current]).tolist()) & unseen
                unseen.difference_update(neighbors)
                stack.extend(neighbors)
                component.extend(neighbors)
            components.append(component)
        if len(components) == 1:
            break

        base = np.asarray(components[0], dtype=np.int64)
        best: tuple[float, int, int] | None = None
        for other_component in components[1:]:
            other = np.asarray(other_component, dtype=np.int64)
            distances = node_dist[np.ix_(base, other)]
            flat_idx = int(np.argmin(distances))
            base_idx, other_idx = np.unravel_index(flat_idx, distances.shape)
            candidate = (float(distances[base_idx, other_idx]), int(base[base_idx]), int(other[other_idx]))
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        _, src, dst = best
        adjacency[src, dst] = True
        adjacency[dst, src] = True
    return adjacency


def generate_request(
    *,
    rng: np.random.Generator,
    request_id: int,
    arrival_minute: float,
    users: Sequence[User],
    services: Sequence[Service],
    request_count: int = 1,
    task_compute_scale: float = 1.0,
    task_data_scale: float = 1.0,
) -> TaskRequest:
    user = users[int(rng.integers(0, len(users)))]
    service_id = int(rng.choice(len(services), p=np.array(user.service_weights)))
    return generate_grouped_request(
        rng=rng,
        request_id=request_id,
        arrival_minute=arrival_minute,
        request_count=request_count,
        user_id=user.user_id,
        home_node=user.home_node,
        service_id=service_id,
        services=services,
        task_compute_scale=task_compute_scale,
        task_data_scale=task_data_scale,
    )


def generate_grouped_request(
    *,
    rng: np.random.Generator,
    request_id: int,
    arrival_minute: float,
    request_count: int,
    user_id: int,
    home_node: int,
    service_id: int,
    services: Sequence[Service],
    task_compute_scale: float = 1.0,
    task_data_scale: float = 1.0,
) -> TaskRequest:
    service = services[service_id]

    input_mb = _sample_lognormal_group_mean(
        rng,
        service.input_mb_mean * task_data_scale,
        0.35,
        request_count,
    )
    stage_compute = []
    stage_output = []
    for stage in service.stages:
        stage_compute.append(
            _sample_lognormal_group_mean(
                rng,
                stage.compute_gcycles_mean * task_compute_scale,
                0.30,
                request_count,
            )
        )
        stage_output.append(
            _sample_lognormal_group_mean(
                rng,
                stage.output_mb_mean * task_data_scale,
                0.35,
                request_count,
            )
        )

    deadline_s = _sample_lognormal_group_mean(rng, service.deadline_s_mean, 0.18, request_count)
    deadline_s = float(np.clip(deadline_s, 0.05, 0.60))
    return TaskRequest(
        request_id=request_id,
        arrival_minute=arrival_minute,
        request_count=int(request_count),
        user_id=user_id,
        home_node=home_node,
        service_id=service_id,
        input_mb=input_mb,
        stage_compute_gcycles=tuple(stage_compute),
        stage_output_mb=tuple(stage_output),
        deadline_s=float(deadline_s),
    )


def _sample_lognormal_group_mean(
    rng: np.random.Generator,
    median: float,
    sigma: float,
    count: int,
) -> float:
    """Sample a moment-matched mean for a group of iid lognormal tasks."""

    count = max(int(count), 1)
    mu = float(np.log(max(median, 1e-12)))
    single_mean = float(np.exp(mu + 0.5 * sigma**2))
    single_variance = float((np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2))
    mean_variance = single_variance / count
    group_sigma_sq = float(np.log1p(mean_variance / max(single_mean**2, 1e-24)))
    group_mu = float(np.log(single_mean) - 0.5 * group_sigma_sq)
    return float(rng.lognormal(group_mu, np.sqrt(group_sigma_sq)))


def _generate_services(
    *,
    rng: np.random.Generator,
    num_service_types: int,
    max_service_stages: int,
    deadline_scale: float = 1.0,
) -> list[Service]:
    services: list[Service] = []
    profiles = [
        {
            "name": "speech-recognition",
            "stage_compute": [0.65, 0.95],
            "stage_output": [0.035, 0.008],
            "input_mb": 0.18,
            "deadline_s": 0.12,
        },
        {
            "name": "ar-overlay",
            "stage_compute": [1.20, 1.65, 0.90],
            "stage_output": [0.10, 0.06, 0.025],
            "input_mb": 0.35,
            "deadline_s": 0.15,
        },
        {
            "name": "video-object-detection",
            "stage_compute": [2.20, 3.20],
            "stage_output": [0.18, 0.04],
            "input_mb": 0.80,
            "deadline_s": 0.28,
        },
        {
            "name": "industrial-inspection",
            "stage_compute": [1.80, 2.40, 1.20],
            "stage_output": [0.12, 0.04, 0.015],
            "input_mb": 0.55,
            "deadline_s": 0.22,
        },
        {
            "name": "traffic-perception",
            "stage_compute": [1.40, 2.10],
            "stage_output": [0.10, 0.025],
            "input_mb": 0.45,
            "deadline_s": 0.20,
        },
        {
            "name": "smart-retail-event",
            "stage_compute": [0.80, 1.10],
            "stage_output": [0.06, 0.015],
            "input_mb": 0.25,
            "deadline_s": 0.16,
        },
        {
            "name": "robot-control",
            "stage_compute": [0.45, 0.70],
            "stage_output": [0.025, 0.008],
            "input_mb": 0.08,
            "deadline_s": 0.08,
        },
        {
            "name": "medical-vital-anomaly",
            "stage_compute": [0.55, 0.85],
            "stage_output": [0.030, 0.006],
            "input_mb": 0.12,
            "deadline_s": 0.10,
        },
        {
            "name": "drone-inspection",
            "stage_compute": [1.60, 2.50, 1.10],
            "stage_output": [0.14, 0.05, 0.018],
            "input_mb": 0.65,
            "deadline_s": 0.24,
        },
        {
            "name": "connected-vehicle-planning",
            "stage_compute": [1.10, 1.45],
            "stage_output": [0.070, 0.018],
            "input_mb": 0.30,
            "deadline_s": 0.13,
        },
    ]
    for service_id in range(num_service_types):
        profile = profiles[service_id % len(profiles)]
        stage_count = min(len(profile["stage_compute"]), max_service_stages)
        stages = []
        for stage_id in range(stage_count):
            stages.append(
                ServiceStage(
                    service_id=service_id,
                    stage_id=stage_id,
                    memory_gb=float(rng.uniform(0.5, 4.0)),
                    storage_gb=float(rng.uniform(1.5, 12.0)),
                    compute_gcycles_mean=float(profile["stage_compute"][stage_id] * rng.uniform(0.85, 1.15)),
                    output_mb_mean=float(profile["stage_output"][stage_id] * rng.uniform(0.80, 1.20)),
                )
            )
        services.append(
            Service(
                service_id=service_id,
                name=str(profile["name"]),
                stages=tuple(stages),
                input_mb_mean=float(profile["input_mb"] * rng.uniform(0.85, 1.15)),
                deadline_s_mean=float(profile["deadline_s"] * deadline_scale * rng.uniform(0.90, 1.10)),
            )
        )
    return services


def _pairwise_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))
