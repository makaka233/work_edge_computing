from __future__ import annotations

import sys

import pytest

from train_monolithic import (
    _format_duration,
    _monolithic_rollout_args,
    _resume_offsets,
    _synchronized_training_args,
    build_parser,
)
from edge_drl.comparison.trace import generate_comparison_trace
from tests.comparison_helpers import tiny_scenario


def test_monolithic_progress_cli_defaults_to_ten_seconds(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_monolithic.py", "--base-checkpoint", "checkpoint.pt"])
    args = build_parser().parse_args()
    assert args.progress_interval_seconds == 10.0
    assert args.sampled_seconds_per_window is None
    assert args.episode_minutes is None
    assert args.seed is None
    assert args.load_checkpoint is None


def test_monolithic_progress_duration_format() -> None:
    assert _format_duration(5.9) == "00:05"
    assert _format_duration(65.0) == "01:05"
    assert _format_duration(3661.0) == "1:01:01"


def test_monolithic_inherits_proposed_stability_and_sampling_settings() -> None:
    base_args = {
        "sampled_seconds_per_window": 6,
        "best_checkpoint_window": 10,
        "slow_lr_decay": True,
        "slow_lr_decay_patience": 20,
        "load_multipliers": "0.8,1.0,1.2,1.4",
    }

    synchronized = _synchronized_training_args(base_args, seed=2026, episode_minutes=60)

    assert synchronized.sampled_seconds_per_window == 6
    assert synchronized.best_checkpoint_window == 10
    assert synchronized.slow_lr_decay is True
    assert synchronized.slow_lr_decay_patience == 20
    assert synchronized.load_multipliers == "0.8,1.0,1.2,1.4"


def test_monolithic_rollout_inherits_slow_counterfactual_credit() -> None:
    args = _monolithic_rollout_args(
        {
            "slow_counterfactual_credit_coef": 1.0,
            "fast_controllable_latency_credit": True,
            "fast_oracle_beam_width": 16,
            "fast_oracle_candidates_per_stage": 4,
        },
        sampled_seconds_per_window=6,
        fast_counterfactual_credit_coef=0.45,
    )

    assert args.slow_counterfactual_credit_coef == 1.0
    assert args.fast_counterfactual_credit_coef == 0.45
    assert args.fast_controllable_latency_credit is True
    assert args.sampled_seconds_per_window == 6
    assert args.fast_oracle_diagnostic_requests == 0


def test_monolithic_rollout_defaults_missing_slow_credit_to_zero() -> None:
    args = _monolithic_rollout_args(
        {},
        sampled_seconds_per_window=3,
        fast_counterfactual_credit_coef=0.5,
    )

    assert args.slow_counterfactual_credit_coef == 0.0


def test_monolithic_resume_offsets_require_an_update_boundary() -> None:
    metadata = {
        "completed_updates": 7,
        "completed_episodes": 14,
        "completed_rollouts": 84,
    }
    assert _resume_offsets(
        metadata,
        episodes_per_update=2,
        windows_per_update=12,
    ) == (7, 14, 84)

    metadata["completed_rollouts"] = 83
    with pytest.raises(ValueError, match="update-boundary"):
        _resume_offsets(
            metadata,
            episodes_per_update=2,
            windows_per_update=12,
        )


def test_scheduled_trace_matches_constant_trace_for_constant_rate() -> None:
    kwargs = dict(
        scenario=tiny_scenario(),
        logical_steps=1200,
        physical_seed=10,
        demand_seed=20,
        request_seed=30,
        task_compute_scale=1.0,
        task_data_scale=1.0,
    )
    constant = generate_comparison_trace(requests_per_minute=30.0, **kwargs)
    scheduled = generate_comparison_trace(
        requests_per_minute=30.0,
        requests_per_minute_schedule=(30.0, 30.0),
        schedule_window_seconds=600,
        **kwargs,
    )
    assert constant.trace_hash == scheduled.trace_hash
