import numpy as np
import pytest

from edge_drl.comparison.dmdr import (
    DMDRScheme,
    adaptive_scaled_probabilities,
    algorithm1_rounding,
    build_virtual_core_model,
    project_resource_feasible_deployment,
)
from tests.comparison_helpers import tiny_replay_env


def test_algorithm1_rounding_is_flag_based_and_capacity_safe_example() -> None:
    values = np.asarray([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]])
    rounded = algorithm1_rounding(values)
    assert rounded.tolist() == [[1, 0], [1, 0], [0, 0]]
    assert not np.array_equal(rounded, np.rint(values).astype(int))


def test_adaptive_scaling_routes_only_to_deployed_nodes() -> None:
    p = np.asarray([[0.2, 0.3, 0.5]])
    n = np.asarray([[1.0, 2.0, 1.0]])
    n_int = np.asarray([[1, 0, 1]])
    adjusted = adaptive_scaled_probabilities(p, n, n_int, np.asarray([10, 10, 10]))
    assert adjusted.sum(axis=1) == np.asarray([1.0])
    assert adjusted[0, 1] == 0.0


def test_dmdr_native_solution_logs_virtual_cores_and_samples_deployed_nodes() -> None:
    env = tiny_replay_env()
    env.reset()
    virtual = build_virtual_core_model(env)
    assert virtual.f_unit == np.median([80.0, 100.0, 120.0, 140.0, 160.0, 180.0]) / 10.0
    scheme = DMDRScheme(routing_seed=55, solver_max_iterations=100)
    scheme.maybe_plan(env)
    assert scheme.integer_multiplicity is not None
    assert np.issubdtype(scheme.integer_multiplicity.dtype, np.integer)
    assert np.all(scheme.integer_multiplicity.sum(axis=0) <= virtual.cores_per_node)
    assert scheme.routing_probabilities is not None
    assert np.all(scheme.routing_probabilities >= 0.0)
    assert np.allclose(scheme.routing_probabilities.sum(axis=1), 1.0)
    assert np.all(scheme.routing_probabilities[scheme.integer_multiplicity <= 0] == 0.0)
    feasible, reason = env.check_deployment_feasible(env.deployment)
    assert feasible, reason
    schedules = scheme.schedule_batch(env, list(env.current_requests))
    for request, schedule in zip(env.current_requests, schedules):
        for stage_id, node in enumerate(schedule):
            assert env.deployment[request.service_id, stage_id, node]  # type: ignore[index]
    diagnostic = scheme.diagnostics.planning[-1]
    assert diagnostic["native_instance_total"] >= diagnostic["physical_replica_total"]
    assert diagnostic["constraint_residual"] <= 1e-5


def test_dmdr_sampler_empirical_frequency_tracks_probability() -> None:
    env = tiny_replay_env()
    env.reset()
    scheme = DMDRScheme(routing_seed=123)
    scheme.stage_keys = [(service.service_id, stage.stage_id) for service in env.scenario.services for stage in service.stages]
    scheme._key_to_index = {key: index for index, key in enumerate(scheme.stage_keys)}
    probabilities = np.zeros((len(scheme.stage_keys), env.config.num_edge_nodes))
    probabilities[:, 0] = 0.25
    probabilities[:, 1] = 0.75
    scheme.routing_probabilities = probabilities
    request = next(request for slot in env._comparison_trace.slots for request in slot if request.service_id == 0)
    repeated = [request] * 20_000
    schedules = scheme.schedule_batch(env, repeated)
    first_stage_nodes = np.asarray([schedule[0] for schedule in schedules])
    assert np.mean(first_stage_nodes == 1) == pytest.approx(0.75, abs=0.02)


def test_resource_projection_removes_overcommit_without_adding_native_support() -> None:
    env = tiny_replay_env()
    env.reset()
    stage_keys = [
        (service.service_id, stage.stage_id)
        for service in env.scenario.services
        for stage in service.stages
    ]
    native = np.ones((len(stage_keys), env.config.num_edge_nodes), dtype=np.int64)
    weights = np.full_like(native, 1.0 / env.config.num_edge_nodes, dtype=np.float64)
    env.config.service_resource_fraction = 0.04

    deployment, projected, diagnostics = project_resource_feasible_deployment(
        env, stage_keys, native, native.astype(np.float64), weights, weights
    )

    feasible, reason = env.check_deployment_feasible(deployment)
    assert feasible, reason
    assert diagnostics["resource_projection_repaired"] is True
    assert diagnostics["resource_projection_removed_pairs"] > 0
    assert np.all(projected <= native)
    assert np.all(projected.sum(axis=1) >= 1)


def test_resource_projection_can_restore_relaxed_support_after_rounding() -> None:
    env = tiny_replay_env()
    env.reset()
    stage_keys = [
        (service.service_id, stage.stage_id)
        for service in env.scenario.services
        for stage in service.stages
    ]
    stage_count = len(stage_keys)
    node_count = env.config.num_edge_nodes
    native = np.zeros((stage_count, node_count), dtype=np.int64)
    native[:, 0] = 1
    relaxed = np.full((stage_count, node_count), 0.1, dtype=np.float64)
    probabilities = np.full((stage_count, node_count), 1.0 / node_count, dtype=np.float64)
    native_routing = np.zeros_like(probabilities)
    native_routing[:, 0] = 1.0
    env.config.service_resource_fraction = 0.04

    deployment, projected, diagnostics = project_resource_feasible_deployment(
        env,
        stage_keys,
        native,
        relaxed,
        probabilities,
        native_routing,
    )

    feasible, reason = env.check_deployment_feasible(deployment)
    assert feasible, reason
    assert diagnostics["resource_projection_used_relaxed_fallback"] is True
    assert diagnostics["resource_projection_restored_pairs"] > 0
    assert np.all(projected.sum(axis=1) >= 1)
