import subprocess
import sys
import csv
from argparse import Namespace
from pathlib import Path

from train_dual_ppo import scenario_seed_for_offset


def test_scenario_refresh_groups_training_episodes():
    args = Namespace(seed=2026, fixed_scenario=False, scenario_refresh_episodes=20)

    assert scenario_seed_for_offset(args, 0, group_by_refresh=True) == 2026
    assert scenario_seed_for_offset(args, 19, group_by_refresh=True) == 2026
    assert scenario_seed_for_offset(args, 20, group_by_refresh=True) == 2027
    assert scenario_seed_for_offset(args, 20, group_by_refresh=False) == 2046


def test_dual_ppo_entrypoint_writes_log_and_checkpoint(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
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
        "--request-aggregation-window-seconds",
        "0",
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
    assert "baseline requests=4" in result.stdout
    assert "update=001" in result.stdout
    assert (log_dir / "training.csv").exists()
    assert (save_dir / "last.pt").exists()
    assert (save_dir / "best.pt").exists()


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
        "--request-aggregation-window-seconds",
        "0",
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
        "--request-aggregation-window-seconds",
        "0",
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
        "8",
        "--request-aggregation-window-seconds",
        "60",
        "--max-representative-groups-per-window",
        "4",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "rollout_unit=episode" in result.stdout
    assert "update=001 episode=001 complete=1" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[-1]
    assert row["update"] == "1"
    assert row["episode"] == "1"
    assert row["episode_complete"] == "1"
    assert float(row["episode_fraction"]) >= 1.0
    assert float(row["deployment_updates"]) >= 2.0


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
