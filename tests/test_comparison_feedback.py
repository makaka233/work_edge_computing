from copy import deepcopy
from dataclasses import replace
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.comparison.classic import HEFTScheme
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.monolithic import MSOScheme
from edge_drl.comparison.proposed import ProposedScheme
from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.trace import ComparisonTrace
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.comparison.window_feedback import SlowWindowFeedbackMixin
from tests.comparison_helpers import tiny_config, tiny_replay_env, tiny_scenario
from tests.test_training_credit_repair import repaired_agent


class FeedbackProbe(SlowWindowFeedbackMixin, BaseComparisonScheme):
    def __init__(self):
        super().__init__()
        self.agent = SimpleNamespace(slow_agent=SimpleNamespace(last_window_feedback={}))
        self._initialize_window_feedback()


def test_weighted_feedback_matches_training_observations_and_clears_accumulator():
    env = tiny_replay_env()
    env.reset()
    probe = FeedbackProbe()
    reference = repaired_agent(env)
    template = next(request for slot in env._comparison_trace.slots for request in slot)
    requests = [replace(template, deadline_s=.3) for _ in range(3)]
    samples = [(.1, 100.), (.8, 20.), (.4, 1000.)]
    env.node_compute_load[1] = .7
    env.link_load[0, 1] = .6
    probe._window_memory_fraction = .4
    probe._window_storage_fraction = .5
    groups = [dict(latency_s=latency, request_count=weight, penalty_latency_s=.05,
                   valid=index != 1) for index, (latency, weight) in enumerate(samples)]
    probe.observe_step(env, requests, [], {"group_infos": groups})
    reference.window_latency_samples = samples.copy()
    reference.window_penalty_latency_sum = .05 * 1120
    reference.window_deadline_violations = 1020
    reference.window_invalid_actions = 20
    reference.window_max_node_compute_load = .7
    reference.window_max_link_load = .6
    reference.window_deployment_memory_fraction = .4
    reference.window_deployment_storage_fraction = .5
    reference.flush_slow_window_reward(done=False, env=env)
    probe._finalize_window_feedback()
    observed = probe.agent.slow_agent.last_window_feedback
    for key, value in observed.items():
        assert value == pytest.approx(reference.slow_agent.last_window_feedback[key])
    assert observed["p95_latency_s"] == .4  # Not the unweighted 0.8-second tail.
    assert observed["avg_latency_s"] == pytest.approx(426 / 1120)
    assert probe._window_weight == 0
    assert not probe._window_latency_samples


def test_deployment_feedback_uses_training_global_capacity_fractions():
    env = tiny_replay_env()
    env.reset()
    env.scenario.nodes[0] = replace(env.scenario.nodes[0], memory_gb=128.)
    env.deployment[0, :2, 0] = True
    probe = FeedbackProbe()
    probe._capture_deployment_usage(env)
    memory, storage = HierarchicalPPOAgent._deployment_resource_fractions(None, env)
    assert probe._window_memory_fraction == memory
    assert probe._window_storage_fraction == storage


def test_empty_window_and_new_episode_cannot_reuse_prior_feedback():
    env = tiny_replay_env()
    env.reset()
    probe = FeedbackProbe()
    probe.agent.slow_agent.last_window_feedback = {"avg_latency_s": 999.}
    probe._window_latency_samples = [(1., 2.)]
    probe._window_weight = 2
    probe._before_slow_plan(env)
    assert probe.agent.slow_agent.last_window_feedback == {}
    assert probe._window_weight == 0
    env.current_time_minute = 10
    probe._before_slow_plan(env)
    assert probe.agent.slow_agent.last_window_feedback == {}
    assert not probe.diagnostics.planning[-1]["has_previous_window_feedback"]


def test_schemes_accumulate_only_their_own_settlement():
    env = tiny_replay_env()
    env.reset()
    request = next(request for slot in env._comparison_trace.slots for request in slot)
    probes = [FeedbackProbe(), FeedbackProbe()]
    for probe, latency in zip(probes, [.2, .9]):
        probe.observe_step(env, [request], [], {"group_infos": [
            dict(request_count=100, latency_s=latency, valid=True)]})
        probe._finalize_window_feedback()
    assert probes[0].agent.slow_agent.last_window_feedback["avg_latency_s"] == .2
    assert probes[1].agent.slow_agent.last_window_feedback["avg_latency_s"] == .9


def test_missing_settlement_results_are_not_silently_ignored():
    env = tiny_replay_env()
    env.reset()
    request = next(request for slot in env._comparison_trace.slots for request in slot)
    with pytest.raises(ValueError, match="one settlement result per request"):
        FeedbackProbe().observe_step(env, [request], [], {"group_infos": []})


def two_window_env():
    source = tiny_replay_env()
    request = next(request for slot in source._comparison_trace.slots for request in slot)
    slots = tuple((replace(request, request_id=step, arrival_minute=step / 60),)
                  if step % 100 == 0 else () for step in range(1200))
    trace = ComparisonTrace(slots, 10, 20, 30, "feedback-fixture")
    return TraceReplayEnv(tiny_config(20), tiny_scenario(), trace)


class FrozenFakeAgent:
    def __init__(self):
        self.snapshots = []
        self.fast_calls = []
        self.slow_agent = SimpleNamespace(last_window_feedback={"leak": 999.},
                                          plan_deployment=self.plan_deployment)
        self.fast_agent = SimpleNamespace(schedule_batch=self.schedule_batch)

    def plan_deployment(self, env, *, deterministic, record):
        assert deterministic and not record
        self.snapshots.append(deepcopy(self.slow_agent.last_window_feedback))
        deployment = np.zeros_like(env.deployment)
        for service in env.scenario.services:
            deployment[service.service_id, :len(service.stages)] = True
        return deployment

    def maybe_update_deployment(self, env, *, deterministic, record):
        env.apply_deployment(self.plan_deployment(env, deterministic=deterministic, record=record))

    def schedule_batch(self, env, requests, *, deterministic, record):
        assert not record
        self.fast_calls.append((deterministic, record))
        return [[request.home_node] * len(request.stage_compute_gcycles) for request in requests]


@pytest.mark.parametrize("scheme_class,module", [
    (ProposedScheme, "edge_drl.comparison.proposed"),
    (HEFTScheme, "edge_drl.comparison.classic"),
    (MSOScheme, "edge_drl.comparison.monolithic"),
])
def test_selected_learned_deployment_adapters_feed_previous_window_without_training(
    monkeypatch, scheme_class, module
):
    env = two_window_env()
    env.reset()
    agent = FrozenFakeAgent()
    if scheme_class is MSOScheme:
        monkeypatch.setattr(module + ".load_checkpoint_configuration",
                            lambda path: (Path(path), {}, {}))
        monkeypatch.setattr(MSOScheme, "_load_agent", lambda *args: agent)
        scheme = scheme_class(env, "unused.pt", device="cpu")
    else:
        monkeypatch.setattr(module + ".build_proposed_agent", lambda *args, **kwargs: (agent, {}))
        scheme = scheme_class(env, "unused.pt", {}, device="cpu")
    for repeat in range(2):
        result = evaluate_scheme_episode(scheme=scheme, env=env,
                                         point=ExperimentPoint("request_load", 1., "test"),
                                         eval_seed=20, routing_repeat=repeat,
                                         sampled_seconds_per_window=6)
        first, second = agent.snapshots[-2:]
        assert first == {}
        assert second["avg_latency_s"] > 0
        assert second["p95_latency_s"] >= second["avg_latency_s"]
        assert second["deployment_memory_fraction"] > 0
        assert result.invalid_action_rate == 0
        assert result.settlement_steps == 12
        assert scheme.agent.slow_agent.last_window_feedback["avg_latency_s"] > 0
        assert scheme._window_weight == 0  # Last window also finalized.
    assert len(agent.snapshots) == 4


def test_unchanged_kkt_settlement_with_fixed_actions():
    env = two_window_env()
    env.reset()
    deployment = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        deployment[service.service_id, :len(service.stages)] = True
    env.apply_deployment(deployment)
    requests = list(env.current_requests)
    schedules = [[r.home_node] * len(r.stage_compute_gcycles) for r in requests]
    before = env.evaluate_batch_schedules(requests, schedules)
    probe = FeedbackProbe()
    _, _, _, info = env.step(schedules, represented_seconds=100)
    loads = env.node_compute_load.copy()
    metrics = env.metrics.copy()
    probe.observe_step(env, requests, schedules, info)
    probe._finalize_window_feedback()
    assert info["group_infos"][0]["latency_s"] == before[0]["latency_s"]
    np.testing.assert_array_equal(loads, env.node_compute_load)
    assert metrics == env.metrics


def test_kkt_action_replay_observes_only_controller_and_keeps_actions_paired(monkeypatch, tmp_path):
    from edge_drl.comparison import kkt_ablation
    env = two_window_env()
    agent = FrozenFakeAgent()
    monkeypatch.setattr("edge_drl.comparison.proposed.build_proposed_agent",
                        lambda *args, **kwargs: (agent, {}))
    monkeypatch.setattr(kkt_ablation, "load_checkpoint_configuration",
                        lambda path: (Path(path), {}, {}))
    monkeypatch.setattr(kkt_ablation, "configured_phase_design", lambda *args, **kwargs:
                        (20, [20], [ExperimentPoint("request_load", 1., "test")], 1, False))
    monkeypatch.setattr(kkt_ablation, "edge_config_from_checkpoint", lambda *args, **kwargs: tiny_config(20))
    monkeypatch.setattr("edge_drl.env.environment.generate_realistic_scenario",
                        lambda **kwargs: tiny_scenario())
    monkeypatch.setattr(kkt_ablation, "generate_comparison_trace", lambda **kwargs: env._comparison_trace)
    output = kkt_ablation.run_kkt_action_replay(
        checkpoint="unused.pt", phase=1, device="cpu", output_root=tmp_path,
        run_id="replay", sampled_seconds_per_window=6)
    assert len(agent.snapshots) == 2
    assert agent.snapshots[0] == {}
    assert agent.snapshots[1]["avg_latency_s"] > 0
    with (output / "raw/request_load.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert len({r["action_trace_hash"] for r in rows}) == 1
    assert len({r["request_count"] for r in rows}) == 1
    assert all(float(r["invalid_action_rate"]) == 0 for r in rows)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["slow_window_feedback_protocol"] == "own-window-weighted-feedback-v1"
