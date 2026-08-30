import numpy as np

from edge_drl.comparison.monolithic import collapse_scenario, collapse_trace
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
