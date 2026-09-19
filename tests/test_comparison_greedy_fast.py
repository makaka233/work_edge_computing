from dataclasses import replace

import numpy as np

from edge_drl.comparison.greedy_fast import GreedyFastScheduler, GreedyFastScheme
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.types import ExperimentPoint
from tests.comparison_helpers import tiny_replay_env


def _full_deployment(env) -> None:
    placement = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        placement[service.service_id, : len(service.stages)] = True
    env.apply_deployment(placement)


def test_greedy_fast_schedules_valid_serial_paths_and_reserves_batch_work() -> None:
    env = tiny_replay_env()
    env.reset()
    _full_deployment(env)
    request = next(request for slot in env._comparison_trace.slots for request in slot)
    request = replace(
        request,
        home_node=0,
        input_mb=0.0,
        stage_compute_gcycles=(20.0,) * len(request.stage_compute_gcycles),
        stage_output_mb=(0.0,) * len(request.stage_output_mb),
        request_count=1,
    )

    scheduler = GreedyFastScheduler()
    schedules = scheduler.schedule_batch(env, [request, request])

    assert len(schedules) == 2
    assert schedules[0][0] != schedules[1][0]
    assert all(len(path) == len(request.stage_compute_gcycles) for path in schedules)
    assert all(info["valid"] for info in env.evaluate_batch_schedules([request, request], schedules))
    assert scheduler.schedule_batch(env, [request, request]) == schedules


def test_greedy_fast_reuses_frozen_proposed_slow_and_settles_episode(monkeypatch) -> None:
    calls = []

    class SlowOnlyAgent:
        def maybe_update_deployment(self, env, *, deterministic, record):
            calls.append((deterministic, record))
            _full_deployment(env)

    monkeypatch.setattr(
        "edge_drl.comparison.algorithmic.build_proposed_agent",
        lambda *args, **kwargs: (SlowOnlyAgent(), {}),
    )
    env = tiny_replay_env()
    scheme = GreedyFastScheme(env, "unused.pt", {}, device="cpu")
    assert isinstance(scheme.scheduler, GreedyFastScheduler)

    result = evaluate_scheme_episode(
        scheme=scheme,
        env=env,
        point=ExperimentPoint("request_load", 1.0, "nominal"),
        eval_seed=20,
        routing_repeat=0,
        sampled_seconds_per_window=6,
    )

    assert calls == [(True, False)]
    assert result.invalid_action_rate == 0.0
    assert result.settlement_steps == 6
    assert result.logical_steps == 600
