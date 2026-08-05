import subprocess
import sys
import csv
from argparse import Namespace
from pathlib import Path

from train_dual_ppo import (
    demand_seed_for_training_rollout,
    effective_replicas_per_stage,
    load_multiplier_for_rollout,
    rollout_start_minute,
    scenario_seed_for_offset,
)


def test_scenario_refresh_groups_training_episodes():
    args = Namespace(seed=2026, fixed_scenario=False, scenario_refresh_episodes=20)

    assert scenario_seed_for_offset(args, 0, group_by_refresh=True) == 2026
    assert scenario_seed_for_offset(args, 19, group_by_refresh=True) == 2026
    assert scenario_seed_for_offset(args, 20, group_by_refresh=True) == 2027
    assert scenario_seed_for_offset(args, 20, group_by_refresh=False) == 2046


def test_demand_sampling_mode_controls_training_demand_seed():
    episode_args = Namespace(seed=2026, fixed_scenario=False, scenario_refresh_episodes=20, demand_sampling_mode="episode")
    rollout_args = Namespace(seed=2026, fixed_scenario=False, scenario_refresh_episodes=20, demand_sampling_mode="rollout")

    assert demand_seed_for_training_rollout(episode_args, rollout_idx=19, episode_idx=0) == 2026
    assert demand_seed_for_training_rollout(episode_args, rollout_idx=20, episode_idx=1) == 2026
    assert demand_seed_for_training_rollout(rollout_args, rollout_idx=19, episode_idx=0) == 2045
    assert demand_seed_for_training_rollout(rollout_args, rollout_idx=20, episode_idx=1) == 2046


def test_zero_replica_cap_uses_node_count():
    args = Namespace(replicas_per_stage=0, num_edge_nodes=32)
    assert effective_replicas_per_stage(args) == 32

    explicit_args = Namespace(replicas_per_stage=5, num_edge_nodes=32)
    assert effective_replicas_per_stage(explicit_args) == 5


def test_rollout_load_and_start_modes_cycle():
    args = Namespace(
        seed=2026,
        load_multipliers="1.0,1.5,2.0",
        rollout_start_mode="cycle-window",
        eval_rollout_start_mode="same",
        episode_hours=24,
        deployment_interval_minutes=10,
    )

    assert [load_multiplier_for_rollout(args, idx) for idx in range(5)] == [1.0, 1.5, 2.0, 1.0, 1.5]
    assert [rollout_start_minute(args, idx) for idx in range(7)] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    stationary_args = Namespace(
        seed=2026,
        rollout_start_mode="cycle-window",
        eval_rollout_start_mode="same",
        episode_hours=4,
        deployment_interval_minutes=10,
        arrival_profile="stationary",
    )
    assert [rollout_start_minute(stationary_args, idx) for idx in range(3)] == [0.0, 0.0, 0.0]


def test_dual_ppo_entrypoint_writes_log_and_checkpoint(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--rollout-unit",
        "window",
        "--deployment-interval-minutes",
        "1",
        "--requests-per-update",
        "4",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--request-aggregation-window-seconds",
        "1",
        "--compute-hotspot-coef",
        "0.08",
        "--link-hotspot-coef",
        "0.04",
        "--eval-baseline",
        "--eval-requests",
        "4",
        "--save-best",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "baseline requests=" in result.stdout
    assert "update=001" in result.stdout
    assert "artificial_cap=none" in result.stdout
    assert "diag_res=" in result.stdout
    assert "slowR=" in result.stdout
    assert (log_dir / "training.csv").exists()
    assert (save_dir / "last.pt").exists()
    assert (save_dir / "best.pt").exists()

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "avg_train_resource_penalty" in rows[0]
    assert "avg_diagnostic_resource_penalty" in rows[0]
    assert "slow_window_return" in rows[0]
    assert "slow_window_count" in rows[0]
    assert "slow_critic_explained_variance" in rows[0]
    assert "avg_service_memory_util" in rows[0]
    assert "active_node_rate" in rows[0]
    assert "hot_link_rate" in rows[0]


def test_fast_only_entrypoint_writes_mode_tagged_log(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--train-mode",
        "fast-only",
        "--updates",
        "1",
        "--rollout-unit",
        "requests",
        "--requests-per-update",
        "4",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--request-aggregation-window-seconds",
        "1",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "train_mode=fast-only" in result.stdout
    assert (log_dir / "training.csv").exists()
    assert (save_dir / "last.pt").exists()


def test_default_entrypoint_writes_under_runs(tmp_path):
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--train-mode",
        "fast-only",
        "--rollout-unit",
        "requests",
        "--requests-per-update",
        "4",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--request-aggregation-window-seconds",
        "1",
        "--run-root",
        str(tmp_path / "runs"),
        "--run-name",
        "organized_run",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "runs" in result.stdout
    assert (tmp_path / "runs" / "organized_run" / "logs" / "training.csv").exists()
    assert (tmp_path / "runs" / "organized_run" / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "runs" / "organized_run" / "metadata.json").exists()


def test_episode_rollout_unit_aligns_update_and_episode(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--rollout-unit",
        "episode",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "30",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "rollout_unit=episode" in result.stdout
    assert "update=001 episode=001 demand_seed=2026-2026 load=1.00-1.00 start_min=0-0 rollouts=1 complete=1" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[-1]
    assert row["update"] == "1"
    assert row["episode"] == "1"
    assert row["episode_complete"] == "1"
    assert float(row["episode_fraction"]) >= 1.0
    assert float(row["deployment_updates"]) >= 2.0


def test_window_rollout_unit_allows_multiple_updates_per_episode(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "2",
        "--rollout-unit",
        "window",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "30",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "rollout_unit=window" in result.stdout
    assert "update=001 episode=001 demand_seed=2026-2026 load=1.00-1.00 start_min=0-0 rollouts=1 complete=0 window=01" in result.stdout
    assert "update=002 episode=001 demand_seed=2026-2026 load=1.00-1.00 start_min=0-0 rollouts=1 complete=1 window=02" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["update"] for row in rows] == ["1", "2"]
    assert [row["episode"] for row in rows] == ["1", "1"]
    assert [row["rollouts_collected"] for row in rows] == ["1", "1"]
    assert [row["window_in_episode"] for row in rows] == ["1", "2"]
    assert rows[0]["episode_complete"] == "0"
    assert rows[1]["episode_complete"] == "1"
    assert "avg_valid_latency_s" in rows[0]
    assert "avg_penalty_latency_s" in rows[0]
    assert "invalid_action_rate" in rows[0]
    assert "avg_node_compute_load" in rows[0]
    assert "load_multiplier" in rows[0]
    assert "start_minute" in rows[0]
    assert "used_replica_rate" in rows[0]
    assert "avg_replica_use_entropy" in rows[0]
    assert "cross_node_stage_transition_rate" in rows[0]
    assert "max_node_memory_util" in rows[0]


def test_window_rollout_demand_sampling_resets_each_update(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "2",
        "--rollout-unit",
        "window",
        "--demand-sampling-mode",
        "rollout",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "30",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "demand_sampling_mode=rollout" in result.stdout
    assert "update=001 episode=001 demand_seed=2026-2026 load=1.00-1.00 start_min=0-0 rollouts=1 complete=0 window=01" in result.stdout
    assert "update=002 episode=002 demand_seed=2027-2027 load=1.00-1.00 start_min=0-0 rollouts=1 complete=0 window=01" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["episode"] for row in rows] == ["1", "2"]
    assert [row["demand_seed"] for row in rows] == ["2026", "2027"]
    assert [row["demand_seed_end"] for row in rows] == ["2026", "2027"]
    assert [row["window_in_episode"] for row in rows] == ["1", "1"]


def test_rollouts_per_update_batches_independent_demand_samples(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--rollout-unit",
        "window",
        "--demand-sampling-mode",
        "rollout",
        "--rollouts-per-update",
        "2",
        "--mean-requests-per-minute",
        "60",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "fast_windows_per_update=2" in result.stdout
    assert "slow_windows_per_update=12" in result.stdout
    assert "episode_horizon=4h deployment_windows=24" in result.stdout
    assert "arrival_profile=stationary" in result.stdout
    assert "update=001 episodes=001-002 demand_seed=2026-2027 load=1.00-1.00 start_min=0-0 rollouts=2 complete=0 window=01" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["rollouts_collected"] == "2"
    assert rows[-1]["demand_seed"] == "2026"
    assert rows[-1]["demand_seed_end"] == "2027"


def test_rollouts_per_update_batches_pressure_levels_and_start_windows(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--rollout-unit",
        "window",
        "--demand-sampling-mode",
        "rollout",
        "--rollouts-per-update",
        "2",
        "--load-multipliers",
        "1.0,1.5",
        "--rollout-start-mode",
        "cycle-window",
        "--arrival-profile",
        "daily",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "30",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "load_multipliers=1.0,1.5" in result.stdout
    assert "rollout_start_mode=cycle-window" in result.stdout
    assert "load=1.00-1.50 start_min=0-30" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["load_multiplier"] == "1.0"
    assert rows[-1]["load_multiplier_end"] == "1.5"
    assert rows[-1]["start_minute"] == "0.0"
    assert rows[-1]["start_minute_end"] == "30.0"


def test_eval_interval_does_not_run_initial_eval_by_default(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--train-mode",
        "fast-only",
        "--rollout-unit",
        "requests",
        "--requests-per-update",
        "4",
        "--eval-interval",
        "1",
        "--eval-requests",
        "4",
        "--eval-seeds",
        "1",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "update=000" not in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["update"] for row in rows] == ["1"]
    assert rows[0]["eval_avg_latency_s"] != "nan"
    assert rows[0]["seen_eval_avg_latency_s"] == "nan"


def test_training_samples_window_but_eval_runs_full_window(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--rollout-unit",
        "window",
        "--deployment-interval-minutes",
        "10",
        "--sampled-seconds-per-window",
        "6",
        "--eval-interval",
        "1",
        "--eval-rollout-unit",
        "window",
        "--eval-seeds",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "temporal_sampling train=6/600s (1.0%) eval=full" in result.stdout
    assert "sampled_steps=6/600" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["settlement_steps"] == "6"
    assert row["logical_steps"] == "600"
    assert float(row["temporal_sampling_fraction"]) == 0.01
    assert float(row["eval_settlement_steps"]) == 600.0
    assert float(row["eval_logical_steps"]) == 600.0
    assert float(row["eval_temporal_sampling_fraction"]) == 1.0


def test_fast_and_slow_ppo_use_independent_window_update_periods(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "3",
        "--rollout-unit",
        "window",
        "--fast-windows-per-update",
        "1",
        "--slow-windows-per-update",
        "2",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "slow_update=0 slow_buffer=1/2" in result.stdout
    assert "slow_update=1 slow_buffer=0/2" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["slow_updated"] for row in rows] == ["0", "1", "0"]
    assert [row["slow_windows_available"] for row in rows] == ["1", "2", "1"]
    assert [row["slow_windows_buffered"] for row in rows] == ["1", "0", "1"]
    assert [row["slow_window_count"] for row in rows] == ["0.0", "2.0", "0.0"]


def test_convergence_analyzer_reads_training_log(tmp_path):
    log_path = tmp_path / "training.csv"
    log_path.write_text(
        "update,requests,avg_latency_s,eval_avg_latency_s,invalid_actions\n"
        "1,10,2.0,nan,0\n"
        "2,10,1.6,nan,0\n"
        "3,10,1.2,nan,0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/analyze_convergence.py", str(log_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Convergence report" in result.stdout
    assert "verdict=" in result.stdout
