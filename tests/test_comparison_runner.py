import numpy as np
import pytest
import torch

from edge_drl.comparison.checkpoint import load_checkpoint_configuration
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.sicp import SICPScheme
from edge_drl.comparison.dmdr import DMDRScheme
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.statistics import paired_differences, summarize_seed_rows
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.comparison.runner import (
    _require_phase2_validation,
    _resolve_monolithic_checkpoint,
    configured_phase_design,
)
from edge_drl.comparison.io import write_json
from tests.comparison_helpers import tiny_replay_env


class FixedScheme(BaseComparisonScheme):
    name = "Proposed"

    def maybe_plan(self, env):
        if env.needs_deployment_update:
            deployment = np.ones(
                (env.config.num_service_types, env.config.max_service_stages, env.config.num_edge_nodes),
                dtype=bool,
            )
            for service in env.scenario.services:
                for stage_id in range(len(service.stages), env.config.max_service_stages):
                    deployment[service.service_id, stage_id] = False
            env.apply_deployment(deployment)

    def schedule_batch(self, env, requests):
        return [[request.home_node] * len(request.stage_compute_gcycles) for request in requests]


def test_runner_counts_seconds_requests_and_zero_invalid_actions() -> None:
    env = tiny_replay_env()
    result = evaluate_scheme_episode(
        scheme=FixedScheme(),
        env=env,
        point=ExperimentPoint("request_load", 1.0, "nominal"),
        eval_seed=20,
        routing_repeat=0,
    )
    assert result.logical_steps == 600
    assert result.settlement_steps == 600
    assert result.request_count == env._comparison_trace.request_count
    assert result.invalid_action_rate == 0.0
    assert result.episode_total_latency_s == result.mean_latency_ms / 1000.0 * result.request_count
    assert result.mean_slot_total_latency_s == result.episode_total_latency_s / 600


def test_comparison_sampling_matches_training_weighted_settlement() -> None:
    env = tiny_replay_env()
    result = evaluate_scheme_episode(
        scheme=FixedScheme(),
        env=env,
        point=ExperimentPoint("request_load", 1.0, "nominal"),
        eval_seed=20,
        routing_repeat=0,
        sampled_seconds_per_window=6,
    )
    assert result.logical_steps == 600
    assert result.settlement_steps == 6
    assert env.metrics["time_steps"] == pytest.approx(600.0)
    assert env.current_time_minute == pytest.approx(10.0)
    assert result.invalid_action_rate == 0.0


def test_statistics_use_seed_rows_and_paired_student_t_intervals() -> None:
    rows = []
    for seed, proposed, baseline in ((1, 10.0, 12.0), (2, 11.0, 14.0), (3, 9.0, 13.0)):
        for scheme, value in (("Proposed", proposed), ("SICP", baseline)):
            rows.append(
                {
                    "scheme": scheme,
                    "scenario_family": "request_load",
                    "scenario_value": 1.0,
                    "eval_seed": seed,
                    "routing_repeat": 0,
                    "failed": False,
                    "mean_latency_ms": value,
                    "p95_latency_ms": value,
                    "episode_total_latency_s": value,
                    "mean_slot_total_latency_s": value,
                    "deadline_violation_rate": value / 100.0,
                }
            )
    summary = summarize_seed_rows(rows)
    paired = paired_differences(rows)
    mean_row = next(row for row in summary if row["scheme"] == "Proposed" and row["metric"] == "mean_latency_ms")
    pair_row = next(row for row in paired if row["metric"] == "mean_latency_ms")
    assert mean_row["n_seeds"] == 3
    assert mean_row["mean"] == 10.0
    assert pair_row["n_pairs"] == 3
    assert pair_row["mean_baseline_minus_proposed"] == 3.0


def test_phase3_requires_a_successful_phase2_marker(tmp_path) -> None:
    with pytest.raises(ValueError):
        _require_phase2_validation(None)
    run = tmp_path / "phase2"
    write_json(run / "diagnostics" / "phase2_validated.json", {"validated": True, "phase": 2})
    _require_phase2_validation(run)


def test_single_seed_phase2_keeps_all_five_scenario_sweeps_but_is_not_formal_validation() -> None:
    minutes, seeds, points, repeats, standard = configured_phase_design(
        2,
        eval_seeds_override=(2026,),
    )
    assert minutes == 60
    assert seeds == [2026]
    assert len(points) == 28
    assert len({point.family for point in points}) == 5
    assert repeats == 1
    assert standard is False


def test_scenario_family_filter_does_not_change_sweep_values() -> None:
    _, seeds, points, _, standard = configured_phase_design(
        2,
        eval_seeds_override=(2026,),
        scenario_families=("request_load",),
    )
    assert seeds == [2026]
    assert [point.value for point in points] == [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    assert standard is False


def test_checkpoint_configuration_uses_embedded_args_without_final_sidecar(tmp_path) -> None:
    checkpoint = tmp_path / "in_progress" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "metadata": {
                "args": {"training_scheme": "Monolithic", "seed": 2026},
                "completed_updates": 12,
            }
        },
        checkpoint,
    )

    resolved, args, metadata = load_checkpoint_configuration(checkpoint)

    assert resolved == checkpoint.resolve()
    assert args == {"training_scheme": "Monolithic", "seed": 2026}
    assert metadata["completed_updates"] == 12


def test_formal_monolithic_checkpoint_is_fixed_across_points(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    first = ExperimentPoint("request_load", 0.8, "low")
    second = ExperimentPoint("wired_bandwidth", 2.0, "high")

    assert _resolve_monolithic_checkpoint(checkpoint, first, mode="fixed") == checkpoint
    assert _resolve_monolithic_checkpoint(checkpoint, second, mode="fixed") == checkpoint


def test_exploratory_per_point_checkpoint_resolution_remains_available(tmp_path) -> None:
    checkpoint = tmp_path / "request_load_0.8" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    point = ExperimentPoint("request_load", 0.8, "low")

    assert _resolve_monolithic_checkpoint(tmp_path, point, mode="per-point") == checkpoint


def test_all_comparison_scheme_interfaces_settle_without_invalid_actions() -> None:
    schemes = (
        FixedScheme(),
        SICPScheme(solver_time_limit_s=10.0, workers=1),
        DMDRScheme(routing_seed=77, solver_max_iterations=100),
    )
    hashes = set()
    for scheme in schemes:
        env = tiny_replay_env()
        env.reset()
        hashes.add(env._comparison_trace.trace_hash)
        scheme.maybe_plan(env)
        requests = scheme.adapt_requests(list(env.current_requests))
        env.current_requests = requests
        env.current_request = requests[0] if requests else None
        schedules = scheme.schedule_batch(env, requests)
        _, _, _, info = env.step(schedules, represented_seconds=1.0)
        assert all(group["valid"] for group in info["group_infos"])
        assert env.metrics["invalid_actions"] == 0.0
    assert len(hashes) == 1
