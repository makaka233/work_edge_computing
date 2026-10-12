"""Opt-in environment regression; no training objectives or baselines change."""

from collections import Counter
from dataclasses import asdict, replace
import sys

import numpy as np
import pytest

from edge_drl.comparison.checkpoint import edge_config_from_checkpoint
from edge_drl.comparison.monolithic import collapse_scenario
from edge_drl.env import latency200
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from edge_drl.env.scenario import TaskRequest
from train_dual_ppo import build_env, build_training_env, parse_args


def candidate_config(**overrides):
    fields = dict(
        seed=2026, physical_seed=2026, scenario_seed=2026,
        num_users=12_000, num_edge_nodes=32, num_service_types=10,
        episode_minutes=60, deployment_interval_minutes=10,
        edge_node_profile=latency200.NODE_PROFILE,
        service_workload_profile=latency200.SERVICE_PROFILE,
        active_user_ratio=0.20, active_user_request_rate_per_minute=1.75,
        wired_link_bandwidth_scale=0.80, service_resource_fraction=0.60,
        wireless_uplink_mbps=300.0, radio_rtt_ms=10.0,
        deadline_scale=2.75, sampled_load_update_mode="interval-average",
    )
    fields.update(overrides)
    return EdgeEnvConfig(**fields)


@pytest.fixture(scope="module")
def candidate():
    env = EdgeComputingEnv(candidate_config())
    env.reset()
    return env


def test_preset_reaches_both_training_environment_builders(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dual_ppo.py", "--pressure-profile", "mec-latency200"])
    args = parse_args()
    for env in (
        build_env(args),
        build_training_env(args, rollout_idx=0, episode_idx=0),
    ):
        config = env.config
        assert (config.num_users, config.num_edge_nodes, config.num_service_types) == (12000, 32, 10)
        assert (config.episode_minutes, config.deployment_interval_minutes) == (60, 10)
        assert config.edge_node_profile == latency200.NODE_PROFILE
        assert config.service_workload_profile == latency200.SERVICE_PROFILE
        assert (config.wireless_uplink_mbps, config.radio_rtt_ms) == (300.0, 10.0)
        assert config.deadline_scale == 2.75
        assert config.task_compute_scale == config.task_data_scale == 1.0
        assert config.compute_allocation_mode == config.link_allocation_mode == "kkt"


def test_explicit_flags_override_candidate_without_changing_legacy_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_dual_ppo.py", "--pressure-profile", "mec-latency200",
        "--wireless-uplink-mbps", "200", "--num-edge-nodes", "40",
        "--load-multipliers", "1.0",
    ])
    args = parse_args()
    assert args.wireless_uplink_mbps == 200.0
    assert args.num_edge_nodes == 40
    assert args.load_multipliers == "1.0"
    assert not args.load_strata
    monkeypatch.setattr(sys, "argv", ["train_dual_ppo.py"])
    legacy = parse_args()
    assert legacy.edge_node_profile == "synthetic-tiered"
    assert legacy.service_workload_profile == "legacy-random"
    assert legacy.wireless_uplink_mbps == 150.0


def test_service_values_units_and_single_resource_allowance(candidate):
    services = candidate.scenario.services
    assert [len(service.stages) for service in services] == [2] + [3] * 9
    expected_inputs = [.352, .300, .152, 1.049, .300, .250, .210, .120, .148, .801]
    np.testing.assert_allclose([row["input_mb"] for row in latency200.SERVICE_PROFILES], expected_inputs)
    # 72 Gcycle/s reference budgets are normalization, not physical GHz/TOPS.
    budgets_ms = [[5, 300], [4, 6, 3], [4, 18, 4], [6, 40, 3], [4, 18, 6],
                  [4, 18, 5], [8, 15, 5], [4, 8, 2], [8, 65, 6], [5, 40, 10]]
    for service, profile, budgets in zip(services, latency200.SERVICE_PROFILES, budgets_ms):
        assert service.name == profile["name"]
        np.testing.assert_allclose(profile["stage_compute"], np.asarray(budgets) * .072)
        assert .85 <= service.input_mb_mean / profile["input_mb"] <= 1.15
        assert .90 <= service.deadline_s_mean / (profile["deadline_s"] * 2.75) <= 1.10
        for stage, cost, output, memory, storage in zip(
            service.stages, profile["stage_compute"], profile["stage_output"],
            profile["stage_memory"], profile["stage_storage"],
        ):
            assert .85 <= stage.compute_gcycles_mean / cost <= 1.15
            assert .80 <= stage.output_mb_mean / output <= 1.20
            assert .92 <= stage.memory_gb / (memory * 1.15) <= 1.08
            assert .90 <= stage.storage_gb / (storage * 1.50) <= 1.10


def test_grounded_hardware_nic_and_propagation(candidate):
    scenario = candidate.scenario
    assert Counter(node.hardware_type for node in scenario.nodes) == dict(zip(latency200.NODE_TYPES, [17, 10, 5]))
    for node in scenario.nodes:
        tier = latency200.NODE_TYPES.index(node.hardware_type)
        assert .90 <= node.compute_gcycles_per_s / latency200.NODE_COMPUTE_GCYCLES_S[tier] <= 1.10
        assert .95 <= node.memory_gb / latency200.NODE_MEMORY_GB[tier] <= 1.05
        assert .90 <= node.storage_gb / latency200.NODE_STORAGE_GB[tier] <= 1.10
    reached = {0}
    for _ in scenario.nodes:
        reached.update(np.flatnonzero(scenario.adjacency[list(reached)].any(axis=0)).tolist())
    assert len(reached) == 32
    for i, j in zip(*np.where(np.triu(scenario.adjacency, k=1))):
        left, right = scenario.nodes[i], scenario.nodes[j]
        bandwidth = scenario.bandwidth_mb_s[i, j]
        assert 0 < bandwidth <= .80 * min(left.network_uplink_mb_s, right.network_uplink_mb_s)
        distance = max(np.hypot(left.x_km - right.x_km, left.y_km - right.y_km), .2)
        residual = scenario.propagation_ms[i, j] - .005 * distance
        assert .05 <= residual <= .20


def test_seeded_geography_demand_and_deadlines_are_not_changed(candidate):
    old = EdgeComputingEnv(candidate_config(
        edge_node_profile="hardware-constrained", service_workload_profile="edge-ai-pipelines",
    ))
    old.reset()
    np.testing.assert_array_equal(old.scenario.adjacency, candidate.scenario.adjacency)
    assert old.scenario.users == candidate.scenario.users
    assert [(n.x_km, n.y_km) for n in old.scenario.nodes] == [(n.x_km, n.y_km) for n in candidate.scenario.nodes]
    assert [s.deadline_s_mean for s in old.scenario.services] == [s.deadline_s_mean for s in candidate.scenario.services]
    assert candidate._arrival_rate_per_minute() / 60 == pytest.approx(57.26826970904384)


def test_complete_catalogue_is_feasible_and_joint_latency_is_not_clipped():
    env = EdgeComputingEnv(candidate_config())
    env.reset()
    node_id = int(np.argmax(env.service_memory_capacities()))
    deployment = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        for stage in service.stages:
            deployment[service.service_id, stage.stage_id, node_id] = True
    feasible, reason = env.check_deployment_feasible(deployment)
    assert feasible, reason
    env.apply_deployment(deployment)
    requests = [TaskRequest(
        request_id=service.service_id, arrival_minute=0., request_count=service.service_id + 1,
        user_id=0, home_node=node_id, service_id=service.service_id,
        input_mb=service.input_mb_mean,
        stage_compute_gcycles=tuple(s.compute_gcycles_mean for s in service.stages),
        stage_output_mb=tuple(s.output_mb_mean for s in service.stages), deadline_s=.05,
    ) for service in env.scenario.services]
    schedules = [[node_id] * len(r.stage_compute_gcycles) for r in requests]
    infos = env.evaluate_batch_schedules(requests, schedules)
    assert all(info["valid"] and np.isfinite(info["latency_s"]) for info in infos)
    weights = np.asarray([r.request_count for r in requests])
    access = sum(r.request_count * (.010 + r.input_mb / 37.5) for r in requests)
    work = np.zeros(3)
    for request in requests:
        work[:len(request.stage_compute_gcycles)] += request.request_count * np.sqrt(request.stage_compute_gcycles)
    total_capacity = sum(n.compute_gcycles_per_s for n in env.scenario.nodes)
    lower = (access + (work ** 2).sum() / total_capacity) / weights.sum()
    actual = np.average([info["physical_latency_s"] for info in infos], weights=weights)
    assert actual >= lower
    capacity = env.scenario.nodes[node_id].compute_gcycles_per_s
    assert actual == pytest.approx((access + (work ** 2).sum() / capacity) / weights.sum())
    heavy = replace(requests[0], request_count=1, stage_compute_gcycles=(1e3, 1e3))
    heavy_info = env.evaluate_schedule(heavy, schedules[0])
    assert heavy_info["valid"]
    assert heavy_info["physical_latency_s"] > .60
    assert heavy_info["latency_s"] == heavy_info["physical_latency_s"]


def test_checkpoint_config_and_monolithic_inherit_same_physical_resources(candidate):
    copied = edge_config_from_checkpoint(
        asdict(candidate.config), episode_minutes=60,
        environment_seed=2027, demand_seed=2033, demand_load_multiplier=1.0,
    )
    assert copied.edge_node_profile == latency200.NODE_PROFILE
    assert copied.service_workload_profile == latency200.SERVICE_PROFILE
    assert copied.wireless_uplink_mbps == 300
    assert copied.physical_seed == 2026
    assert copied.scenario_seed == 2033
    collapsed = collapse_scenario(candidate.scenario)
    assert collapsed.nodes == candidate.scenario.nodes
    np.testing.assert_array_equal(collapsed.bandwidth_mb_s, candidate.scenario.bandwidth_mb_s)
    for base, whole in zip(candidate.scenario.services, collapsed.services):
        assert len(whole.stages) == 1
        assert whole.input_mb_mean == base.input_mb_mean
        assert whole.stages[0].compute_gcycles_mean == sum(s.compute_gcycles_mean for s in base.stages)
        assert whole.stages[0].memory_gb == sum(s.memory_gb for s in base.stages)
        assert whole.stages[0].storage_gb == sum(s.storage_gb for s in base.stages)
