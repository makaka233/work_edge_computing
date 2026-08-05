from dataclasses import replace

import numpy as np

from edge_drl.agents.hierarchical import build_baseline_agent
from edge_drl.allocators.kkt import ComputeDemand, LinkDemand, allocate_compute_kkt, allocate_link_kkt
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from edge_drl.env.scenario import generate_realistic_scenario


def test_default_episode_uses_one_second_steps_and_ten_minute_deployment():
    config = EdgeEnvConfig()
    env = EdgeComputingEnv(config)

    assert config.episode_hours == 4
    assert config.deployment_interval_minutes == 10
    assert config.request_aggregation_window_seconds == 1.0
    assert config.arrival_profile == "stationary"
    env.current_time_minute = 0.0
    initial_rate = env._arrival_rate_per_minute()
    env.current_time_minute = 180.0
    assert env._arrival_rate_per_minute() == initial_rate
    assert not env.done
    env.current_time_minute = 240.0
    assert env.done


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


def test_kkt_group_multiplicity_matches_individual_equal_tasks():
    grouped = [ComputeDemand("group", 0, 4.0, multiplicity=3.0)]
    individuals = [ComputeDemand(f"task-{idx}", 0, 4.0) for idx in range(3)]

    _, grouped_delays, grouped_total = allocate_compute_kkt(grouped, np.array([12.0]))
    _, individual_delays, individual_total = allocate_compute_kkt(individuals, np.array([12.0]))

    assert np.isclose(grouped_delays["group"], individual_delays["task-0"])
    assert np.isclose(grouped_total, individual_total)


def test_environment_constraints_and_rollout():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=7,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    agent = build_baseline_agent()
    agent.fast_policy.candidate_limit_per_stage = 2
    obs = env.reset()
    assert len(env.scenario.users) == 10_000
    assert max(len(service.stages) for service in env.scenario.services) <= 3
    assert env.config.deployment_interval_minutes == 10
    assert obs["needs_deployment_update"] is True

    agent.maybe_update_deployment(env)
    assert obs["deployment"].shape == env.deployment.shape
    feasible, reason = env.check_deployment_feasible(env.deployment)
    assert feasible, reason

    for _ in range(5):
        actions = agent.act_batch(env)
        obs, reward, done, info = env.step(actions)
        assert info["valid"], info["violations"]
        assert np.isfinite(reward)
        assert info["latency_s"] >= 0
        if done:
            break


def test_migration_change_is_logged_without_default_penalty():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=17,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    agent = build_baseline_agent()
    agent.fast_policy.candidate_limit_per_stage = 2
    env.reset()
    agent.maybe_update_deployment(env)
    assert env.last_migration_cost > 0

    first_actions = agent.act_batch(env)
    _, _, _, first_info = env.step(first_actions)
    assert first_info["migration_cost"] > 0
    assert first_info["migration_penalty"] == 0
    assert env.last_migration_cost == 0

    second_actions = agent.act_batch(env)
    _, _, _, second_info = env.step(second_actions)
    assert second_info["migration_cost"] == 0


def test_physical_seed_keeps_topology_fixed():
    config_a = EdgeEnvConfig(
        seed=101,
        physical_seed=7,
        scenario_seed=101,
        num_users=10_000,
        num_edge_nodes=16,
        num_service_types=3,
        episode_hours=1,
    )
    config_b = EdgeEnvConfig(
        seed=202,
        physical_seed=7,
        scenario_seed=202,
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


def test_demand_refresh_does_not_change_physical_network():
    base = dict(
        seed=101,
        physical_seed=2026,
        num_users=10_000,
        num_edge_nodes=16,
        num_service_types=3,
        episode_hours=1,
    )
    env_a = EdgeComputingEnv(EdgeEnvConfig(**base, scenario_seed=7))
    env_b = EdgeComputingEnv(EdgeEnvConfig(**base, scenario_seed=8))
    env_a.reset()
    env_b.reset()

    nodes_a = [
        (n.x_km, n.y_km, n.memory_gb, n.storage_gb, n.compute_gcycles_per_s)
        for n in env_a.scenario.nodes
    ]
    nodes_b = [
        (n.x_km, n.y_km, n.memory_gb, n.storage_gb, n.compute_gcycles_per_s)
        for n in env_b.scenario.nodes
    ]
    services_a = [
        (
            service.name,
            service.input_mb_mean,
            service.deadline_s_mean,
            tuple((s.memory_gb, s.storage_gb, s.compute_gcycles_mean, s.output_mb_mean) for s in service.stages),
        )
        for service in env_a.scenario.services
    ]
    services_b = [
        (
            service.name,
            service.input_mb_mean,
            service.deadline_s_mean,
            tuple((s.memory_gb, s.storage_gb, s.compute_gcycles_mean, s.output_mb_mean) for s in service.stages),
        )
        for service in env_b.scenario.services
    ]

    assert nodes_a == nodes_b
    assert services_a == services_b
    np.testing.assert_array_equal(env_a.scenario.adjacency, env_b.scenario.adjacency)
    np.testing.assert_allclose(env_a.scenario.bandwidth_mb_s, env_b.scenario.bandwidth_mb_s)
    np.testing.assert_allclose(env_a.scenario.propagation_ms, env_b.scenario.propagation_ms)

    home_a = np.bincount([u.home_node for u in env_a.scenario.users], minlength=16)
    home_b = np.bincount([u.home_node for u in env_b.scenario.users], minlength=16)
    assert not np.array_equal(home_a, home_b)


def test_default_traffic_is_city_scale_for_ten_thousand_users():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=31,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=24,
            arrival_profile="daily",
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
            request_aggregation_window_seconds=1.0,
        )
    )
    agent = build_baseline_agent()
    agent.fast_policy.candidate_limit_per_stage = 2
    env.reset()
    agent.maybe_update_deployment(env)
    latencies = []
    weights = []
    while not env.done and env.metrics["requests"] < 500:
        requests = list(env.current_requests)
        actions = agent.act_batch(env)
        _, _, _, info = env.step(actions)
        latencies.extend(float(group["latency_s"]) for group in info["group_infos"])
        weights.extend(float(request.request_count) for request in requests)

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
    agent.fast_policy.candidate_limit_per_stage = 2
    env.reset()
    agent.maybe_update_deployment(env)
    requests = list(env.current_requests)
    start_time = env.current_time_minute
    expected_count = sum(request.request_count for request in requests)
    assert expected_count > 1

    actions = agent.act_batch(env)
    _, _, _, info = env.step(actions)

    assert info["request_count"] == expected_count
    assert env.metrics["time_steps"] == 1
    assert env.metrics["aggregate_events"] == len(requests)
    assert env.metrics["requests"] == expected_count
    assert np.isclose(env.current_time_minute - start_time, 1.0 / 60.0)


def test_joint_settlement_is_independent_of_group_order():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=44,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=1_800.0,
        )
    )
    agent = build_baseline_agent()
    agent.fast_policy.candidate_limit_per_stage = 2
    env.reset()
    agent.maybe_update_deployment(env)
    requests = list(env.current_requests)
    actions = [agent.fast_policy.act(env, request) for request in requests]

    forward = env.evaluate_batch_schedules(requests, actions)
    reverse = env.evaluate_batch_schedules(list(reversed(requests)), list(reversed(actions)))
    forward_latency = {request.request_id: info["latency_s"] for request, info in zip(requests, forward)}
    reverse_latency = {
        request.request_id: info["latency_s"]
        for request, info in zip(reversed(requests), reverse)
    }

    assert forward_latency.keys() == reverse_latency.keys()
    for request_id in forward_latency:
        assert np.isclose(forward_latency[request_id], reverse_latency[request_id])


def test_joint_allocation_reflects_group_concurrency():
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
    action = agent.fast_policy.act(env, request)

    single_info = env.evaluate_schedule(single_request, action)
    grouped_info = env.evaluate_schedule(grouped_request, action)

    assert grouped_info["latency_s"] > single_info["latency_s"]
    assert single_info["compute_demands"][0].compute_gcycles == grouped_info["compute_demands"][0].compute_gcycles


def test_task_load_scales_request_compute_and_data():
    base_env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=46,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=6_000.0,
        )
    )
    heavy_env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=46,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=6_000.0,
            task_compute_scale=2.0,
            task_data_scale=3.0,
        )
    )
    base_env.reset()
    heavy_env.reset()
    base_request = base_env.current_request
    heavy_request = heavy_env.current_request
    assert base_request is not None
    assert heavy_request is not None

    assert heavy_request.service_id == base_request.service_id
    assert heavy_request.home_node == base_request.home_node
    assert heavy_request.input_mb > base_request.input_mb
    assert sum(heavy_request.stage_compute_gcycles) > sum(base_request.stage_compute_gcycles)
    assert sum(heavy_request.stage_output_mb) > sum(base_request.stage_output_mb)


def test_physical_capacity_scales_node_compute_and_wired_links():
    base_env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=48,
            physical_seed=48,
            scenario_seed=99,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
        )
    )
    constrained_env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=48,
            physical_seed=48,
            scenario_seed=99,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            node_compute_capacity_scale=0.5,
            wired_link_bandwidth_scale=0.25,
        )
    )
    base_env.reset()
    constrained_env.reset()
    assert base_env.scenario is not None
    assert constrained_env.scenario is not None

    base_compute = np.array([node.compute_gcycles_per_s for node in base_env.scenario.nodes])
    constrained_compute = np.array([node.compute_gcycles_per_s for node in constrained_env.scenario.nodes])
    np.testing.assert_allclose(constrained_compute, base_compute * 0.5)

    finite = np.isfinite(base_env.scenario.bandwidth_mb_s) & ~np.eye(16, dtype=bool)
    np.testing.assert_allclose(
        constrained_env.scenario.bandwidth_mb_s[finite],
        base_env.scenario.bandwidth_mb_s[finite] * 0.25,
    )


def test_service_resource_fraction_preserves_physical_nodes_but_limits_placement_capacity():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=49,
            physical_seed=49,
            scenario_seed=101,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            service_resource_fraction=0.4,
        )
    )
    env.reset()
    assert env.scenario is not None
    physical_memory = np.asarray([node.memory_gb for node in env.scenario.nodes])
    physical_storage = np.asarray([node.storage_gb for node in env.scenario.nodes])
    np.testing.assert_allclose(env.service_memory_capacities(), physical_memory * 0.4)
    np.testing.assert_allclose(env.service_storage_capacities(), physical_storage * 0.4)


def test_all_nonempty_node_service_groups_are_preserved():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=47,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=1_800.0,
            request_aggregation_window_seconds=1.0,
        )
    )
    env.reset()
    window_time = env.current_time_minute
    requests = env.current_requests
    assert len(requests) > 4
    assert all(request.arrival_minute == window_time for request in requests)
    assert len({(request.home_node, request.service_id) for request in requests}) == len(requests)
    assert sum(request.request_count for request in requests) > 0
