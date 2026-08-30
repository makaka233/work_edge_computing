import numpy as np

from edge_drl.comparison.monolithic import MonolithicScheme
from tests.comparison_helpers import tiny_replay_env


def test_monolithic_projects_whole_service_and_zeroes_internal_transfer() -> None:
    env = tiny_replay_env()
    env.reset()
    scheme = MonolithicScheme(solver_time_limit_s=10.0)
    scheme.maybe_plan(env)
    assert env.deployment is not None
    for service in env.scenario.services:  # type: ignore[union-attr]
        stage_masks = [env.deployment[service.service_id, stage.stage_id] for stage in service.stages]
        assert all(np.array_equal(stage_masks[0], mask) for mask in stage_masks[1:])
        assert stage_masks[0].any()
    requests = scheme.adapt_requests(list(env.current_requests))
    assert all(request.stage_output_mb == (0.0,) * len(request.stage_output_mb) for request in requests)
    assert all(sum(request.stage_compute_gcycles[1:]) == 0.0 for request in requests)
    schedules = scheme.schedule_batch(env, requests)
    assert all(len(set(schedule)) == 1 for schedule in schedules)
    infos = env.evaluate_batch_schedules(requests, schedules)
    assert all(not info["link_demands"] for info in infos)
    assert all(len([d for d in info["compute_demands"] if d.compute_gcycles > 0.0]) == 1 for info in infos)
    for original, adapted, info in zip(env.current_requests, requests, infos):
        positive = [d for d in info["compute_demands"] if d.compute_gcycles > 0.0]
        assert positive[0].compute_gcycles == sum(original.stage_compute_gcycles)
        assert adapted.input_mb == original.input_mb
