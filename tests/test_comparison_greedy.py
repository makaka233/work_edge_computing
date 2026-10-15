from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from edge_drl.comparison.algorithmic import CausalKKTPathScheduler, DPMSScheme
from edge_drl.comparison.greedy import (
    DemandGreedyPlacement, StageMarginalScheduler, GreedyDARPScheme, GreedyDPMSScheme,
)
from edge_drl.comparison.heuristic import DARPScheme, NearestReplicaScheduler
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.runner import _make_scheme, _require_phase2_validation
from edge_drl.comparison.types import ExperimentPoint
from tests.comparison_helpers import tiny_replay_env


def test_greedy_deployment_feasible_deterministic_without_mutation():
    env = tiny_replay_env()
    env.reset()
    before = env.deployment.copy()
    planner = DemandGreedyPlacement()
    first, details = planner.plan(env)
    second, _ = planner.plan(env)
    assert np.array_equal(first, second)
    assert np.array_equal(env.deployment, before)
    assert env.check_deployment_feasible(first)[0]
    assert details["baseline_implementation"] == "demand-greedy-v2"
    # With generous RAM all stages can have replicas on every node: no cap.
    assert first.sum() == 5 * 6


def test_greedy_deployment_honors_tight_node_budgets(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    monkeypatch.setattr(env, "service_memory_capacities", lambda: np.full(6, 1.0))
    deployment, _ = DemandGreedyPlacement().plan(env)
    assert env.check_deployment_feasible(deployment)[0]
    assert deployment.sum() == 6
    assert all(deployment[s.service_id, :len(s.stages)].any(axis=1).all()
               for s in env.scenario.services)


def test_greedy_coverage_repairs_fragmentation(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    service = env.scenario.services[0]
    env.scenario.services[0] = replace(service, stages=service.stages + (
        replace(service.stages[-1], stage_id=2),))
    sizes = iter([6.0, 5.0, 3.0, 2.0, 2.0, 2.0])
    env.scenario.services = [replace(s, stages=tuple(replace(t, memory_gb=next(sizes))
                                                   for t in s.stages))
                             for s in env.scenario.services]
    monkeypatch.setattr("edge_drl.comparison.greedy.DemandAwareMILPPlacement._transfer_matrices",
                        lambda env: (np.zeros((6, 6)), np.zeros((6, 6))))
    # Greedy 6+3 / 5+2+2 fragments; feasible packing is 6+2+2 / 5+3+2.
    monkeypatch.setattr(env, "service_memory_capacities", lambda: np.array([10., 10., 0., 0., 0., 0.]))
    env.scenario.nodes = [replace(node, compute_gcycles_per_s=1000. if i == 0 else 1.)
                          for i, node in enumerate(env.scenario.nodes)]
    deployment, details = DemandGreedyPlacement().plan(env)
    assert details["coverage_repaired"] is True
    assert env.check_deployment_feasible(deployment)[0]


def test_greedy_reports_infeasible_coverage(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    monkeypatch.setattr(env, "service_memory_capacities", lambda: np.zeros(6))
    with pytest.raises(RuntimeError, match="coverage feasibility repair failed"):
        DemandGreedyPlacement().plan(env)


def _all_nodes(env):
    deployment = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        deployment[service.service_id, :len(service.stages)] = True
    env.apply_deployment(deployment)


def test_stage_scheduler_uses_full_candidate_set_and_batch_ledger():
    env = tiny_replay_env()
    env.reset()
    _all_nodes(env)
    request = next(r for slot in env._comparison_trace.slots for r in slot)
    request = replace(request, input_mb=0., stage_output_mb=(0.,) * len(request.stage_output_mb),
                      stage_compute_gcycles=(20.,) * len(request.stage_compute_gcycles))
    scheduler = StageMarginalScheduler()
    paths = scheduler.schedule_batch(env, [request, request])
    assert paths[0][0] == 5  # Highest-index, fastest node is not truncated away.
    assert paths[0][0] != paths[1][0]
    assert scheduler.schedule_batch(env, [request, request]) == paths
    assert all(info["valid"] for info in env.evaluate_batch_schedules([request, request], paths))


def test_stage_scheduler_is_local_not_whole_chain(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    _all_nodes(env)
    request = next(r for slot in env._comparison_trace.slots for r in slot if r.service_id == 0)
    request = replace(request, home_node=0, input_mb=0., stage_compute_gcycles=(0., 0.),
                      stage_output_mb=(0., 0.))
    mask = np.zeros((2, 6), dtype=bool)
    mask[0, :2], mask[1, 2] = True, True
    monkeypatch.setattr(env, "scheduler_candidate_mask", lambda request: mask)

    def route(env, source, target):
        if source == target:
            return (), 0.
        return (), 10. if (source, target) == (0, 2) else 1.

    greedy, chain = StageMarginalScheduler(), CausalKKTPathScheduler()
    monkeypatch.setattr(greedy, "_path", route)
    monkeypatch.setattr(chain, "_path", route)
    assert greedy.schedule_batch(env, [request]) == [[0, 2]]
    assert chain.schedule_batch(env, [request]) == [[1, 2]]


def test_stage_scheduler_prunes_only_unreachable_dead_ends(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    request = next(r for slot in env._comparison_trace.slots for r in slot if r.service_id == 0)
    request = replace(request, home_node=3)
    mask = np.zeros((2, 6), dtype=bool)
    mask[0, :2], mask[1, 2] = True, True
    monkeypatch.setattr(env, "scheduler_candidate_mask", lambda request: mask)
    scheduler = StageMarginalScheduler()
    monkeypatch.setattr(scheduler, "_path",
                        lambda env, source, target: None if (source, target) == (0, 2) else ((), 0.))
    assert scheduler.schedule_batch(env, [request]) == [[1, 2]]


@pytest.mark.parametrize("implementation,expected", [
    ("legacy-v1", DPMSScheme), ("demand-greedy-v2", GreedyDPMSScheme),
])
def test_factory_keeps_name_and_selects_version(implementation, expected):
    scheme = _make_scheme("DPMS", tiny_replay_env(), Path("unused.pt"), {}, device="cpu",
                          baseline_implementation=implementation)
    assert type(scheme) is expected
    assert scheme.name == "Demand-Aware Placement and Marginal-Cost Scheduling (DPMS)"


@pytest.mark.parametrize("implementation,expected", [
    ("legacy-v1", DARPScheme), ("demand-greedy-v2", GreedyDARPScheme),
])
def test_darp_reuses_frozen_fast_and_original_name(monkeypatch, implementation, expected):
    calls = []
    class Fast:
        def schedule_batch(self, env, requests, *, deterministic, record):
            calls.append((deterministic, record))
            return NearestReplicaScheduler().schedule_batch(env, requests)
    class Agent:
        fast_agent = Fast()
    monkeypatch.setattr("edge_drl.comparison.heuristic.build_proposed_agent",
                        lambda *args, **kwargs: (Agent(), {}))
    env = tiny_replay_env()
    env.reset()
    scheme = _make_scheme("DARP", env, Path("unused.pt"), {}, device="cpu",
                          baseline_implementation=implementation)
    assert type(scheme) is expected
    assert scheme.name == "Demand-Aware Replica Placement (DARP)"
    scheme.maybe_plan(env)
    requests = list(env.current_requests)
    paths = scheme.schedule_batch(env, requests)
    assert calls == [(True, False)]
    assert all(info["valid"] for info in env.evaluate_batch_schedules(requests, paths))


@pytest.mark.parametrize("network_mode", ["edge-only", "shared-nic"])
def test_greedy_dpms_sampled_episode_uses_common_settlement(network_mode):
    env = tiny_replay_env()
    env.config.network_constraint_mode = network_mode
    result = evaluate_scheme_episode(
        scheme=GreedyDPMSScheme(), env=env,
        point=ExperimentPoint("request_load", 1., "nominal"),
        eval_seed=20, routing_repeat=0, sampled_seconds_per_window=6,
    )
    assert result.invalid_action_rate == 0
    assert not result.failed
    assert result.settlement_steps == 6
    assert result.logical_steps == 600
    assert env.config.network_constraint_mode == network_mode


@pytest.mark.parametrize("implementation", ["legacy-v1", "demand-greedy-v2"])
def test_runner_records_implementation_in_metadata_and_rows(monkeypatch, tmp_path, implementation):
    import csv
    from edge_drl.comparison import runner
    from tests.comparison_helpers import tiny_config
    monkeypatch.setattr(runner, "load_checkpoint_configuration",
                        lambda path: (Path("unused.pt"), {}, {}))
    monkeypatch.setattr(runner, "edge_config_from_checkpoint", lambda *args, **kwargs: tiny_config())
    monkeypatch.setattr(runner, "EdgeComputingEnv", lambda config: tiny_replay_env())
    output = runner.run_comparison(
        checkpoint="unused.pt", phase=1, device="cpu", output_root=tmp_path,
        schemes=("DPMS",), sampled_seconds_per_window=6,
        baseline_implementation=implementation,
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["baseline_implementation"] == implementation
    assert metadata["greedy_in_main_results"] == (implementation == "demand-greedy-v2")
    with (output / "raw" / "request_load.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["failed"] == "False"
    assert rows[0]["scheme_implementation"] == implementation
    with (output / "diagnostics" / "service_latency.csv").open(encoding="utf-8", newline="") as stream:
        service_rows = list(csv.DictReader(stream))
    assert service_rows
    assert all(row["scheme_implementation"] == implementation for row in service_rows)


def test_phase3_cannot_reuse_validation_of_another_baseline_version(tmp_path):
    directory = tmp_path / "diagnostics"
    directory.mkdir()
    (directory / "phase2_validated.json").write_text(json.dumps({
        "validated": True, "phase": 2, "schemes": ["Proposed", "DPMS"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation differs"):
        _require_phase2_validation(tmp_path, schemes=("Proposed", "DPMS"),
                                   baseline_implementation="demand-greedy-v2")
    _require_phase2_validation(tmp_path, schemes=("Proposed", "DPMS"),
                               baseline_implementation="legacy-v1")
