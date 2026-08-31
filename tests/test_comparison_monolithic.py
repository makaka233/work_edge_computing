import numpy as np

from edge_drl.comparison.monolithic import (
    adapt_request_for_monolithic_evaluation,
    collapse_scenario,
    collapse_trace,
    expand_monolithic_schedule,
)
from tests.comparison_helpers import tiny_replay_env


def test_monolithic_collapses_each_service_to_one_aggregated_stage() -> None:
    env = tiny_replay_env()
    env.reset()
    mono = collapse_scenario(env.scenario)
    for original, aggregated in zip(env.scenario.services, mono.services):
        assert len(aggregated.stages) == 1
        assert aggregated.stages[0].compute_gcycles_mean == sum(
            stage.compute_gcycles_mean for stage in original.stages
        )
        assert aggregated.stages[0].memory_gb == sum(stage.memory_gb for stage in original.stages)
        assert aggregated.stages[0].storage_gb == sum(stage.storage_gb for stage in original.stages)
        assert aggregated.stages[0].output_mb_mean == 0.0
    collapsed = collapse_trace(env._comparison_trace)
    for slot in collapsed.slots:
        for request in slot:
            assert len(request.stage_compute_gcycles) == 1
            assert request.stage_output_mb == (0.0,)


def test_monolithic_evaluation_preserves_stage_shape_and_shared_node_projection() -> None:
    env = tiny_replay_env()
    env.reset()
    original = next(request for slot in env._comparison_trace.slots for request in slot)
    adapted = adapt_request_for_monolithic_evaluation(original)
    assert len(adapted.stage_compute_gcycles) == len(original.stage_compute_gcycles)
    assert adapted.stage_compute_gcycles[0] == sum(original.stage_compute_gcycles)
    assert adapted.stage_compute_gcycles[1:] == (0.0,) * (len(original.stage_compute_gcycles) - 1)
    assert adapted.stage_output_mb == (0.0,) * len(original.stage_output_mb)
    assert expand_monolithic_schedule([4], len(adapted.stage_compute_gcycles)) == [4, 4]
