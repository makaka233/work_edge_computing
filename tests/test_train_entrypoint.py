import subprocess
import sys
import csv
from argparse import Namespace
from pathlib import Path

from train_dual_ppo import (
    apply_pressure_profile,
    demand_seed_for_training_rollout,
    demand_profile_summary,
    effective_replicas_per_stage,
    load_multiplier_for_rollout,
    parse_args,
    rollout_start_minute,
    scenario_seed_for_offset,
    use_deterministic_fast_collection,
)


def test_policy_stability_defaults_reach_the_training_entrypoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dual_ppo.py"])
    args = parse_args()

    assert args.fast_lr == 2e-4
    assert args.fast_entropy_coef == 0.001
    assert args.slow_placement_entropy_coef == 0.005


def test_mec_pressure_profile_scales_demand_and_fixed_capacity():
    args = Namespace(
        pressure_profile="mec-moderate",
        active_user_ratio=0.15,
        active_user_request_rate_per_minute=1.5,
        traffic_scale=1.0,
        load_multipliers="1.0",
        task_compute_scale=1.0,
        task_data_scale=1.0,
        node_compute_capacity_scale=1.0,
        wired_link_bandwidth_scale=1.0,
        service_resource_fraction=0.5,
    )

    apply_pressure_profile(args, argv=["--pressure-profile", "mec-moderate"])

    assert args.active_user_ratio == 0.20
    assert args.active_user_request_rate_per_minute == 1.75
    assert args.task_compute_scale == 1.65
    assert args.task_data_scale == 2.5
    assert args.node_compute_capacity_scale == 0.65
    assert args.wired_link_bandwidth_scale == 0.15
    assert args.service_resource_fraction == 0.25
    assert args.deadline_scale == 2.75
    assert args.load_multipliers == "0.8,1.1,1.4,1.7"


def test_pressure_profile_does_not_override_explicit_scale():
    args = Namespace(
        pressure_profile="mec-moderate",
        active_user_ratio=0.20,
        active_user_request_rate_per_minute=1.75,
        traffic_scale=1.0,
        load_multipliers="0.8,1.1,1.4,1.7",
        task_compute_scale=3.0,
        task_data_scale=2.0,
        node_compute_capacity_scale=0.65,
        wired_link_bandwidth_scale=0.35,
        service_resource_fraction=0.45,
    )

    apply_pressure_profile(
        args,
        argv=["--pressure-profile", "mec-moderate", "--task-compute-scale", "3.0"],
    )

    assert args.task_compute_scale == 3.0
    assert args.task_data_scale == 2.5


def test_slow_collection_uses_frozen_stochastic_fast_by_default():
    assert not use_deterministic_fast_collection(True, "stochastic")
    assert use_deterministic_fast_collection(True, "deterministic")
    assert not use_deterministic_fast_collection(False, "deterministic")


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


def test_shuffled_demand_pool_reuses_each_seed_once_per_cycle():
    args = Namespace(
        seed=2026,
        fixed_scenario=False,
        scenario_refresh_episodes=2,
        demand_sampling_mode="episode",
        demand_scenario_schedule="shuffled-pool",
        demand_scenario_pool_size=4,
    )

    first_cycle = [
        demand_seed_for_training_rollout(args, rollout_idx=0, episode_idx=episode)
        for episode in range(0, 8, 2)
    ]
    second_cycle = [
        demand_seed_for_training_rollout(args, rollout_idx=0, episode_idx=episode)
        for episode in range(8, 16, 2)
    ]

    assert sorted(first_cycle) == [2026, 2027, 2028, 2029]
    assert sorted(second_cycle) == [2026, 2027, 2028, 2029]
    assert first_cycle == [
        demand_seed_for_training_rollout(args, rollout_idx=0, episode_idx=episode)
        for episode in range(0, 8, 2)
    ]


def test_demand_profile_summary_exposes_policy_independent_difficulty():
    from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig

    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=2026,
            physical_seed=2026,
            scenario_seed=2030,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
        )
    )
    env.reset()

    summary = demand_profile_summary(env)

    assert summary["demand_expected_compute_gcycles"] > 0.0
    assert summary["demand_expected_data_mb"] > 0.0
    assert summary["demand_expected_deadline_s"] > 0.0
    assert summary["demand_service_popularity_entropy"] > 0.0


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
        "--slow-warmup-updates",
        "0",
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
        "--slow-warmup-updates",
        "0",
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
        "--slow-warmup-updates",
        "0",
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
    assert "slow_windows_per_update=32" in result.stdout
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
        "--slow-warmup-updates",
        "0",
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
        "--joint-training-schedule",
        "simultaneous",
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


def test_alternating_schedule_freezes_one_controller_per_phase(tmp_path):
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
        "--joint-training-schedule",
        "alternating",
        "--fast-updates-per-cycle",
        "2",
        "--fast-warmup-updates",
        "0",
        "--slow-warmup-updates",
        "0",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "1",
        "--sampled-seconds-per-window",
        "5",
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
    subprocess.run(command, check=True, capture_output=True, text=True)

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["training_phase"] for row in rows] == ["fast", "fast", "slow"]
    assert float(rows[0]["fast_loss"]) != 0.0
    assert float(rows[1]["fast_loss"]) != 0.0
    assert float(rows[2]["fast_loss"]) == 0.0
    assert [row["slow_updated"] for row in rows] == ["0", "0", "1"]
    assert float(rows[2]["slow_count_return_std"]) > 0.0


def test_alternating_schedule_runs_fast_then_slow_warmup(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "5",
        "--rollout-unit",
        "window",
        "--fast-windows-per-update",
        "1",
        "--slow-windows-per-update",
        "1",
        "--joint-training-schedule",
        "alternating",
        "--fast-updates-per-cycle",
        "2",
        "--fast-warmup-updates",
        "2",
        "--slow-warmup-updates",
        "2",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "1",
        "--sampled-seconds-per-window",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "8",
        "--num-service-types",
        "2",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["training_phase"] for row in rows] == [
        "fast_warmup",
        "fast_warmup",
        "slow_warmup",
        "slow_warmup",
        "fast",
    ]
    assert [row["slow_updated"] for row in rows] == ["0", "0", "1", "1", "0"]
    assert any(float(row["fast_loss"]) != 0.0 for row in rows[:2])
    assert [float(row["fast_loss"]) == 0.0 for row in rows[2:4]] == [True, True]
    assert float(rows[4]["fast_loss"]) != 0.0
    assert "slow_count_effective_replicas_per_stage" in rows[0]
    assert "slow_count_redundant_replica_fraction" in rows[0]
    assert "slow_count_advantage_std" in rows[0]


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
