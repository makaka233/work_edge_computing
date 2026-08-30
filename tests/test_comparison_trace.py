from dataclasses import FrozenInstanceError

import pytest

from edge_drl.comparison.scenario_transforms import transform_trace
from edge_drl.comparison.types import ExperimentPoint
from tests.comparison_helpers import tiny_config, tiny_replay_env, tiny_scenario, tiny_trace
from edge_drl.comparison.replay_env import TraceReplayEnv


def test_trace_is_deterministic_complete_and_immutable() -> None:
    first = tiny_trace(request_seed=31)
    second = tiny_trace(request_seed=31)
    different = tiny_trace(request_seed=32)
    assert first.logical_steps == 600
    assert first.slots == second.slots
    assert first.trace_hash == second.trace_hash
    assert first.trace_hash != different.trace_hash
    with pytest.raises(FrozenInstanceError):
        first.request_seed = 99  # type: ignore[misc]


def test_trace_transform_preserves_required_quantities() -> None:
    source = tiny_trace(request_seed=33)
    hetero = transform_trace(source, ExperimentPoint("stage_heterogeneity", 8.0, "H=8"))
    data = transform_trace(source, ExperimentPoint("intermediate_data", 2.0, "data=2"))
    original_requests = [request for slot in source.slots for request in slot]
    hetero_requests = [request for slot in hetero.slots for request in slot]
    data_requests = [request for slot in data.slots for request in slot]
    assert all(
        sum(before.stage_compute_gcycles) == pytest.approx(sum(after.stage_compute_gcycles), abs=1e-12)
        for before, after in zip(original_requests, hetero_requests)
    )
    assert all(
        after.stage_output_mb[-1] == before.stage_output_mb[-1]
        for before, after in zip(original_requests, data_requests)
    )
    assert hetero.trace_hash != source.trace_hash
    assert data.trace_hash != source.trace_hash


def test_four_replay_environments_share_hash_and_never_temporally_sample() -> None:
    trace = tiny_trace(request_seed=34)
    environments = [TraceReplayEnv(tiny_config(), tiny_scenario(), trace) for _ in range(4)]
    assert {env._comparison_trace.trace_hash for env in environments} == {trace.trace_hash}
    for env in environments:
        env.reset()
        deployment = __import__("numpy").ones(
            (env.config.num_service_types, env.config.max_service_stages, env.config.num_edge_nodes),
            dtype=bool,
        )
        for service in env.scenario.services:
            deployment[service.service_id, len(service.stages) :] = False
        env.apply_deployment(deployment)
        schedules = [
            [request.home_node] * len(request.stage_compute_gcycles)
            for request in env.current_requests
        ]
        env.step(schedules, represented_seconds=1.0)
        assert env.metrics["settlement_steps"] == 1.0
        assert env.metrics["time_steps"] == 1.0
