from dataclasses import replace

import numpy as np

from edge_drl.agents.hierarchical import build_baseline_agent
from edge_drl.allocators.kkt import ComputeDemand, LinkDemand, allocate_compute_kkt, allocate_link_kkt
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from edge_drl.env.scenario import generate_realistic_scenario


def test_kkt_compute_sqrt_rule():
    demands = [
        ComputeDemand("a", 0, 4.0),
        ComputeDemand("b", 0, 9.0),
    ]
    allocations, delays, total = allocate_compute_kkt(demands, np.array([10.0]))
    assert round(allocations["a"], 6) == 4.0
    assert round(allocations["b"], 6) == 6.0
    assert round(delays["a"], 6) == 1.0
    assert round(delays["b"], 6) == 1.5
    assert round(total, 6) == 2.5


def test_kkt_link_sqrt_rule():
    demands = [
        LinkDemand("a", 0, 1, 25.0),
        LinkDemand("b", 0, 1, 100.0),
    ]
    bandwidth = np.zeros((2, 2), dtype=float)
    bandwidth[0, 1] = 30.0
    allocations, delays, total = allocate_link_kkt(demands, bandwidth)
    assert round(allocations["a"], 6) == 10.0
    assert round(allocations["b"], 6) == 20.0
    assert round(delays["a"], 6) == 2.5
    assert round(delays["b"], 6) == 5.0
    assert round(total, 6) == 7.5


def test_environment_constraints_and_rollout():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=7,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=2.0,
            request_aggregation_window_seconds=0.0,
        )
    )
    agent = build_baseline_agent()
    obs = env.reset()
    assert len(env.scenario.users) == 10_000
    assert max(len(service.stages) for service in env.scenario.services) <= 3
    assert env.config.deployment_interval_minutes == 240
    assert obs["needs_deployment_update"] is True

    agent.maybe_update_deployment(env)
    assert obs["deployment"].shape == env.deployment.shape
    feasible, reason = env.check_deployment_feasible(env.deployment)
    assert feasible, reason

    for _ in range(5):
        action = agent.act(env)
        obs, reward, done, info = env.step(action)
        assert info["valid"], info["violations"]
        assert np.isfinite(reward)
        assert info["latency_s"] >= 0
        if done:
            break


def test_migration_cost_is_charged_once():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=17,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=2.0,
            request_aggregation_window_seconds=0.0,
        )
    )
    agent = build_baseline_agent()
    env.reset()
    agent.maybe_update_deployment(env)
    assert env.last_migration_cost > 0

    first_action = agent.act(env)
    _, _, _, first_info = env.step(first_action)
    assert first_info["migration_cost"] > 0
    assert env.last_migration_cost == 0

    second_action = agent.act(env)
    _, _, _, second_info = env.step(second_action)
    assert second_info["migration_cost"] == 0


def test_scenario_seed_keeps_topology_fixed():
    config_a = EdgeEnvConfig(
        seed=101,
        scenario_seed=7,
        num_users=10_000,
        num_edge_nodes=16,
        num_service_types=3,
        episode_hours=1,
    )
    config_b = EdgeEnvConfig(
        seed=202,
        scenario_seed=7,
        num_users=10_000,
        num_edge_nodes=16,
        num_service_types=3,
        episode_hours=1,
    )
    env_a = EdgeComputingEnv(config_a)
    env_b = EdgeComputingEnv(config_b)
    env_a.reset()
    env_b.reset()
    xy_a = [(node.x_km, node.y_km) for node in env_a.scenario.nodes]
    xy_b = [(node.x_km, node.y_km) for node in env_b.scenario.nodes]
    assert xy_a == xy_b


def test_default_traffic_is_city_scale_for_ten_thousand_users():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=31,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=24,
        )
    )
    rates = []
    for minute in range(24 * 60):
        env.current_time_minute = float(minute)
        rates.append(env._arrival_rate_per_minute())

    avg_requests_per_second = float(np.mean(rates) / 60.0)
    peak_requests_per_second = float(np.max(rates) / 60.0)
    min_requests_per_second = float(np.min(rates) / 60.0)

    assert avg_requests_per_second > 10.0
    assert peak_requests_per_second > avg_requests_per_second
    assert min_requests_per_second > 1.0


def test_ten_service_catalog_is_realistic_and_staged():
    scenario = generate_realistic_scenario(
        rng=np.random.default_rng(51),
        num_users=10_000,
        num_edge_nodes=32,
        num_service_types=10,
        max_service_stages=3,
    )
    names = {service.name for service in scenario.services}

    assert len(scenario.services) == 10
    assert "drone-inspection" in names
    assert "connected-vehicle-planning" in names
    assert "medical-vital-anomaly" in names
    assert max(len(service.stages) for service in scenario.services) <= 3


def test_city_links_include_heterogeneous_bottlenecks():
    scenario = generate_realistic_scenario(
        rng=np.random.default_rng(53),
        num_users=10_000,
        num_edge_nodes=32,
        num_service_types=10,
        max_service_stages=3,
    )
    finite_bandwidth = scenario.bandwidth_mb_s[np.isfinite(scenario.bandwidth_mb_s) & (scenario.bandwidth_mb_s > 0)]

    assert scenario.adjacency.all()
    assert np.isfinite(scenario.bandwidth_mb_s[~np.eye(32, dtype=bool)]).all()
    assert np.isfinite(scenario.propagation_ms[~np.eye(32, dtype=bool)]).all()
    assert finite_bandwidth.min() < 100.0
    assert finite_bandwidth.max() > 700.0
    assert finite_bandwidth.max() / finite_bandwidth.min() > 10.0


def test_default_single_task_latency_is_mec_scale():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=41,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            request_aggregation_window_seconds=10.0,
            max_representative_groups_per_window=8,
        )
    )
    agent = build_baseline_agent()
    env.reset()
    agent.maybe_update_deployment(env)
    latencies = []
    weights = []
    while not env.done and env.metrics["requests"] < 20_000:
        action = agent.act(env)
        _, _, _, info = env.step(action)
        latencies.append(float(info["latency_s"]))
        weights.append(float(info["request_count"]))

    avg_latency_s = float(np.average(np.asarray(latencies), weights=np.asarray(weights)))
    p95_latency_s = float(np.percentile(latencies, 95))

    assert 0.02 <= avg_latency_s <= 0.30
    assert p95_latency_s <= 0.60


def test_request_aggregation_counts_underlying_requests():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=43,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=6_000.0,
            request_aggregation_window_seconds=1.0,
        )
    )
    agent = build_baseline_agent()
    env.reset()
    agent.maybe_update_deployment(env)
    request = env.current_request
    assert request is not None
    assert request.request_count > 1

    action = agent.act(env)
    _, _, _, info = env.step(action)

    assert info["request_count"] == request.request_count
    assert env.metrics["aggregate_events"] == 1
    assert env.metrics["requests"] == request.request_count


def test_aggregation_does_not_inflate_single_task_latency():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=45,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=1_800.0,
            request_aggregation_window_seconds=1.0,
        )
    )
    agent = build_baseline_agent()
    env.reset()
    agent.maybe_update_deployment(env)
    request = env.current_request
    assert request is not None

    single_request = replace(request, request_count=1)
    grouped_request = replace(request, request_count=100)
    action = agent.act(env)

    single_info = env.evaluate_schedule(single_request, action)
    grouped_info = env.evaluate_schedule(grouped_request, action)

    assert np.isclose(single_info["latency_s"], grouped_info["latency_s"])
    assert single_info["compute_demands"][0].compute_gcycles == grouped_info["compute_demands"][0].compute_gcycles


def test_representative_group_sampling_preserves_window_request_count():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=47,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=1_800.0,
            request_aggregation_window_seconds=10.0,
            max_representative_groups_per_window=4,
        )
    )
    env.reset()
    window_time = env.current_time_minute
    requests = [env.current_request, *env.pending_requests]
    assert len(requests) <= 4
    assert all(request.arrival_minute == window_time for request in requests)
    assert sum(request.request_count for request in requests) > 0
