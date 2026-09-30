from dataclasses import replace
import numpy as np
import pytest
from edge_drl.comparison.heuristic import (
    NearestReplicaScheduler,
    PopularityDeployScheme,
    PopularityFirstFitPlacement,
    FitDeployScheme,
    UniformFirstFitPlacement,
    UniformFirstFitDeployScheme,
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


def test_first_fit_is_repeatable_and_ignores_compute_and_network():
    env = tiny_replay_env()
    env.reset()
    before = env.deployment.copy()
    planner = PopularityFirstFitPlacement()
    first, details = planner.plan(env)
    env.node_compute_load[:] = 0.9
    env.link_load[:] = 0.9
    env.scenario.bandwidth_mb_s[:] = 0.01
    second, _ = planner.plan(env)
    assert np.array_equal(first, second)
    assert np.array_equal(before, env.deployment)
    assert env.check_deployment_feasible(first)[0]
    assert details["planner"] == "popularity_first_fit"


@pytest.mark.parametrize("planner_class", [PopularityFirstFitPlacement, UniformFirstFitPlacement])
def test_first_fit_repairs_fragmented_coverage(monkeypatch, planner_class):
    env = tiny_replay_env()
    env.reset()
    demands = iter([4.0, 4.0, 6.0, 3.0, 3.0])
    env.scenario.services = [
        replace(service, stages=tuple(replace(stage, memory_gb=next(demands))
                                     for stage in service.stages))
        for service in env.scenario.services
    ]
    monkeypatch.setattr(env, "service_memory_capacities", lambda: np.array([10., 10., 0., 0., 0., 0.]))
    deployment, details = planner_class().plan(env)
    assert details["coverage_repaired"] is True
    assert env.check_deployment_feasible(deployment)[0]
    for service in env.scenario.services:
        assert deployment[service.service_id, :len(service.stages)].any(axis=1).all()


@pytest.mark.parametrize("scheme_class,planner_class", [
    (FitDeployScheme, PopularityFirstFitPlacement),
    (UniformFirstFitDeployScheme, UniformFirstFitPlacement),
])
def test_first_fit_scheme_reuses_fast_and_public_settlement(monkeypatch, scheme_class, planner_class):
    env = tiny_replay_env()
    env.reset()
    calls = []

    class FakeFast:
        def schedule_batch(self, received_env, requests, *, deterministic, record):
            calls.append((deterministic, record))
            return NearestReplicaScheduler().schedule_batch(received_env, requests)

    class FakeAgent:
        fast_agent = FakeFast()

    monkeypatch.setattr("edge_drl.comparison.heuristic.build_proposed_agent",
                        lambda *args, **kwargs: (FakeAgent(), {}))
    scheme = scheme_class(env, "unused.pt", {}, device="cpu")
    result = evaluate_scheme_episode(
        scheme=scheme, env=env, point=ExperimentPoint('request_load', 1., 'nominal'),
        eval_seed=20, routing_repeat=0, sampled_seconds_per_window=6,
    )
    assert isinstance(scheme.planner, planner_class)
    assert calls and all(call == (True, False) for call in calls)
    assert result.invalid_action_rate == 0
    assert result.settlement_steps == 6


def test_uniform_first_fit_balances_counts_without_reading_demand(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    monkeypatch.setattr(env, "service_memory_capacities", lambda: np.full(6, 3.0))
    def unexpected_demand(*args):
        raise AssertionError("Uniform placement must not read demand")
    monkeypatch.setattr("edge_drl.comparison.heuristic.DemandAwareMILPPlacement._demand",
                        unexpected_demand)
    before = env.deployment.copy()
    first, details = UniformFirstFitPlacement().plan(env)
    second, _ = UniformFirstFitPlacement().plan(env)
    counts = [int(first[service.service_id, stage.stage_id].sum())
              for service in env.scenario.services for stage in service.stages]
    assert first.sum() == 18
    assert max(counts) - min(counts) <= 1
    assert np.array_equal(first, second)
    assert np.array_equal(env.deployment, before)
    assert env.check_deployment_feasible(first)[0]
    assert details["planner"] == "uniform_first_fit"
