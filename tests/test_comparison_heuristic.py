from dataclasses import replace
import numpy as np
from edge_drl.comparison.heuristic import (
    NearestReplicaScheduler,
    PopularityDeployScheme,
    PopularityEFTScheme,
    PopularityNearestScheme,
    StageCompletionScheduler,
)
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.types import ExperimentPoint
from tests.comparison_helpers import tiny_replay_env


def test_popularity_placement_is_feasible_repeatable_and_does_not_mutate_env():
    env = tiny_replay_env()
    env.reset()
    before = env.deployment.copy()
    scheme = PopularityEFTScheme()
    first, _ = scheme.planner.plan(env)
    second, _ = scheme.planner.plan(env)
    assert np.array_equal(first, second)
    assert np.array_equal(env.deployment, before)
    assert env.check_deployment_feasible(first)[0]


def test_completion_scheduler_accounts_for_batch_work():
    env = tiny_replay_env()
    env.reset()
    deployment = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        deployment[service.service_id, :len(service.stages)] = True
    env.apply_deployment(deployment)
    request = next(r for slot in env._comparison_trace.slots for r in slot)
    request = replace(request, input_mb=0.0,
                      stage_compute_gcycles=(20.0,) * len(request.stage_compute_gcycles),
                      stage_output_mb=(0.0,) * len(request.stage_output_mb))
    paths = StageCompletionScheduler().schedule_batch(env, [request, request])
    assert paths[0][0] != paths[1][0]
    assert all(info['valid'] for info in env.evaluate_batch_schedules([request, request], paths))


def test_popularity_eft_uses_public_settlement():
    result = evaluate_scheme_episode(scheme=PopularityEFTScheme(), env=tiny_replay_env(),
                                    point=ExperimentPoint('request_load', 1.0, 'nominal'),
                                    eval_seed=20, routing_repeat=0, sampled_seconds_per_window=6)
    assert result.invalid_action_rate == 0
    assert result.settlement_steps == 6
    assert result.logical_steps == 600


def test_nearest_scheduler_is_deterministic_and_valid():
    env = tiny_replay_env()
    env.reset()
    deployment = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        deployment[service.service_id, :len(service.stages)] = True
    env.apply_deployment(deployment)
    requests = list(env.current_requests)
    scheduler = NearestReplicaScheduler()
    first = scheduler.schedule_batch(env, requests)
    second = scheduler.schedule_batch(env, requests)
    assert first == second
    assert all(info['valid'] for info in env.evaluate_batch_schedules(requests, first))


def test_popularity_nearest_uses_public_settlement():
    result = evaluate_scheme_episode(scheme=PopularityNearestScheme(), env=tiny_replay_env(),
                                    point=ExperimentPoint('request_load', 1.0, 'nominal'),
                                    eval_seed=20, routing_repeat=0, sampled_seconds_per_window=6)
    assert result.invalid_action_rate == 0
    assert result.settlement_steps == 6


def test_popularity_deploy_replaces_only_planner(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    calls = []

    class FakeFast:
        def schedule_batch(self, received_env, requests, *, deterministic, record):
            calls.append((received_env, requests, deterministic, record))
            return NearestReplicaScheduler().schedule_batch(received_env, requests)

    class FakeAgent:
        fast_agent = FakeFast()

    monkeypatch.setattr(
        "edge_drl.comparison.heuristic.build_proposed_agent",
        lambda *args, **kwargs: (FakeAgent(), {"source": "test"}),
    )
    scheme = PopularityDeployScheme(env, "unused.pt", {}, device="cpu")
    scheme.maybe_plan(env)
    schedules = scheme.schedule_batch(env, list(env.current_requests))

    assert env.check_deployment_feasible(env.deployment)[0]
    assert calls and calls[0][2:] == (True, False)
    assert all(info["valid"] for info in env.evaluate_batch_schedules(env.current_requests, schedules))
