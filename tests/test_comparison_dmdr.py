import numpy as np
import pytest

from edge_drl.comparison.dmdr import (
    DMDRScheme,
    adaptive_scaled_probabilities,
    algorithm1_rounding,
    build_virtual_core_model,
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
