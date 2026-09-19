from dataclasses import replace

import numpy as np

from edge_drl.comparison.algorithmic import (
    AlgorithmicScheme,
    CausalKKTPathScheduler,
    DemandAwareMILPPlacement,
    HybridScheme,
)
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.types import ExperimentPoint
from tests.comparison_helpers import tiny_replay_env


def test_algorithmic_deployment_and_serial_routing_settle_without_invalid_actions() -> None:
    env = tiny_replay_env()
    env.reset()
    scheme = AlgorithmicScheme()
    scheme.maybe_plan(env)
    feasible, reason = env.check_deployment_feasible(env.deployment)
    assert feasible, reason
    assert env.metrics["deployment_updates"] == 1
    assert scheme.diagnostics.planning[0]["replicas"] >= 5

    requests = list(env.current_requests)
    schedules = scheme.schedule_batch(env, requests)
    assert len(schedules) == len(requests)
    assert all(len(path) == len(request.stage_compute_gcycles) for request, path in zip(requests, schedules))
    _, _, _, info = env.step(schedules)
    assert all(group["valid"] for group in info["group_infos"])


def test_algorithmic_planner_repeats_the_same_feasible_decision() -> None:
    env = tiny_replay_env()
    env.reset()
    planner = DemandAwareMILPPlacement()
    first, first_details = planner.plan(env)
    second, second_details = planner.plan(env)
    assert np.array_equal(first, second)
    assert first_details["added_replicas"] == second_details["added_replicas"]


def test_algorithmic_scheduler_accounts_for_existing_batch_work() -> None:
    env = tiny_replay_env()
    env.reset()
    deployment = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        deployment[service.service_id, : len(service.stages)] = True
    env.apply_deployment(deployment)
    request = next(request for slot in env._comparison_trace.slots for request in slot)
    request = replace(
        request,
        home_node=0,
        input_mb=0.0,
        stage_compute_gcycles=(20.0,) * len(request.stage_compute_gcycles),
        stage_output_mb=(0.0,) * len(request.stage_output_mb),
    )
    scheduler = CausalKKTPathScheduler()
    paths = scheduler.schedule_batch(env, [request, request])
    assert paths[0][0] != paths[1][0]
    assert all(info["valid"] for info in env.evaluate_batch_schedules([request, request], paths))


def test_hybrid_uses_proposed_slow_and_the_common_algorithmic_fast(monkeypatch) -> None:
    env = tiny_replay_env()
    env.reset()
    calls = []

    class SlowOnlyAgent:
        def maybe_update_deployment(self, current_env, *, deterministic, record):
            calls.append((deterministic, record))
            placement = np.zeros_like(current_env.deployment)
            for service in current_env.scenario.services:
                placement[service.service_id, : len(service.stages)] = True
            current_env.apply_deployment(placement)

    monkeypatch.setattr(
        "edge_drl.comparison.algorithmic.build_proposed_agent",
        lambda *args, **kwargs: (SlowOnlyAgent(), {}),
    )
    scheme = HybridScheme(env, "unused.pt", {}, device="cpu")
    scheme.maybe_plan(env)
    assert calls == [(True, False)]
    assert isinstance(scheme.scheduler, CausalKKTPathScheduler)
    paths = scheme.schedule_batch(env, list(env.current_requests))
    assert all(info["valid"] for info in env.evaluate_batch_schedules(list(env.current_requests), paths))


def test_algorithmic_scheme_completes_sampled_episode() -> None:
    env = tiny_replay_env()
    result = evaluate_scheme_episode(
        scheme=AlgorithmicScheme(),
        env=env,
        point=ExperimentPoint("request_load", 1.0, "nominal"),
        eval_seed=20,
        routing_repeat=0,
        sampled_seconds_per_window=6,
    )
    assert result.invalid_action_rate == 0.0
    assert result.settlement_steps == 6
    assert result.logical_steps == 600
