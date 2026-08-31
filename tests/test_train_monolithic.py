from __future__ import annotations

import sys

from train_monolithic import _format_duration, build_parser
from edge_drl.comparison.trace import generate_comparison_trace
from tests.comparison_helpers import tiny_scenario


def test_monolithic_progress_cli_defaults_to_ten_seconds(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_monolithic.py", "--base-checkpoint", "checkpoint.pt"])
    args = build_parser().parse_args()
    assert args.progress_interval_seconds == 10.0
    assert args.sampled_seconds_per_window is None
    assert args.episode_minutes is None
    assert args.seed is None


def test_monolithic_progress_duration_format() -> None:
    assert _format_duration(5.9) == "00:05"
    assert _format_duration(65.0) == "01:05"
    assert _format_duration(3661.0) == "1:01:01"


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
