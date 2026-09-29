import numpy as np

from edge_drl.comparison.classic import HEFTFastScheme, HEFTScheduler
from tests.comparison_helpers import tiny_replay_env


def test_heft_scheduler_is_deterministic_reachable_and_valid():
    env = tiny_replay_env()
    env.reset()
    deployment = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        deployment[service.service_id, :len(service.stages)] = True
    env.apply_deployment(deployment)
    requests = list(env.current_requests)
    scheduler = HEFTScheduler()
    first = scheduler.schedule_batch(env, requests)
    second = scheduler.schedule_batch(env, requests)
    assert first == second
    assert all(len(path) == len(request.stage_compute_gcycles) for request, path in zip(requests, first))
    assert all(info['valid'] for info in env.evaluate_batch_schedules(requests, first))


def test_heft_fast_reuses_frozen_proposed_slow(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    calls = []

    class SlowOnlyAgent:
        def maybe_update_deployment(self, current_env, *, deterministic, record):
            calls.append((deterministic, record))
            placement = np.zeros_like(current_env.deployment)
            for service in current_env.scenario.services:
                placement[service.service_id, :len(service.stages)] = True
            current_env.apply_deployment(placement)

    monkeypatch.setattr(
        'edge_drl.comparison.classic.build_proposed_agent',
        lambda *args, **kwargs: (SlowOnlyAgent(), {}),
    )
    scheme = HEFTFastScheme(env, 'unused.pt', {}, device='cpu')
    scheme.maybe_plan(env)
    assert calls == [(True, False)]
    paths = scheme.schedule_batch(env, list(env.current_requests))
    assert all(info['valid'] for info in env.evaluate_batch_schedules(list(env.current_requests), paths))
