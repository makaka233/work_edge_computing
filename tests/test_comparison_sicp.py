import numpy as np

from edge_drl.comparison.sicp import SICPScheme
from tests.comparison_helpers import tiny_replay_env


def test_sicp_has_one_replica_per_stage_and_fixed_window_chain() -> None:
    env = tiny_replay_env()
    env.reset()
    scheme = SICPScheme(solver_time_limit_s=10.0, workers=1)
    scheme.maybe_plan(env)
    assert env.deployment is not None
    for service in env.scenario.services:  # type: ignore[union-attr]
        for stage in service.stages:
            assert int(env.deployment[service.service_id, stage.stage_id].sum()) == 1
    before = dict(scheme.chains)
    schedules = scheme.schedule_batch(env, list(env.current_requests))
    assert all(tuple(schedule) == before[request.service_id] for request, schedule in zip(env.current_requests, schedules))
    env.current_time_minute = 5.0
    scheme.maybe_plan(env)
    assert scheme.chains == before
    feasible, reason = env.check_deployment_feasible(env.deployment)
    assert feasible, reason
