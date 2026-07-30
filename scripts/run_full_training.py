from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged full-training pipeline.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-root", type=str, default="runs")
    parser.add_argument("--num-users", type=int, default=10_000)
    parser.add_argument("--num-edge-nodes", type=int, default=32)
    parser.add_argument("--num-service-types", type=int, default=10)
    parser.add_argument("--episode-hours", type=int, default=24)
    parser.add_argument("--scenario-refresh-episodes", type=int, default=20)
    parser.add_argument("--mean-requests-per-minute", type=float, default=None)
    parser.add_argument("--active-user-ratio", type=float, default=0.15)
    parser.add_argument("--active-user-request-rate-per-minute", type=float, default=1.5)
    parser.add_argument("--traffic-scale", type=float, default=1.0)
    parser.add_argument("--request-aggregation-window-seconds", type=float, default=10.0)
    parser.add_argument("--max-representative-groups-per-window", type=int, default=16)
    parser.add_argument("--load-ewma-tau-minutes", type=float, default=1.0)
    parser.add_argument("--wireless-uplink-mbps", type=float, default=150.0)
    parser.add_argument("--radio-rtt-ms", type=float, default=10.0)
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--eval-requests", type=int, default=256)
    parser.add_argument("--fast-bc-requests", type=int, default=2048)
    parser.add_argument("--fast-bc-epochs", type=int, default=30)
    parser.add_argument("--fast-updates", type=int, default=80)
    parser.add_argument("--joint-updates", type=int, default=120)
    parser.add_argument("--requests-per-update", type=int, default=256)
    parser.add_argument("--rollout-unit", choices=["requests", "episode"], default="requests")
    parser.add_argument("--eval-rollout-unit", choices=["requests", "episode", "same"], default="requests")
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--reward-mode", choices=["latency", "greedy-advantage", "mixed"], default="latency")
    parser.add_argument("--fast-policy-kind", choices=["node_scorer", "gat_node_scorer"], default="gat_node_scorer")
    parser.add_argument("--max-replicas-per-stage", dest="replicas_per_stage", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    parser.add_argument("--fixed-scenario", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    fixed = ["--fixed-scenario"] if args.fixed_scenario else []
    common = [
        sys.executable,
        "train_dual_ppo.py",
        "--seed",
        str(args.seed),
        "--num-users",
        str(args.num_users),
        "--num-edge-nodes",
        str(args.num_edge_nodes),
        "--num-service-types",
        str(args.num_service_types),
        "--episode-hours",
        str(args.episode_hours),
        "--scenario-refresh-episodes",
        str(args.scenario_refresh_episodes),
        "--active-user-ratio",
        str(args.active_user_ratio),
        "--active-user-request-rate-per-minute",
        str(args.active_user_request_rate_per_minute),
        "--traffic-scale",
        str(args.traffic_scale),
        "--request-aggregation-window-seconds",
        str(args.request_aggregation_window_seconds),
        "--max-representative-groups-per-window",
        str(args.max_representative_groups_per_window),
        "--load-ewma-tau-minutes",
        str(args.load_ewma_tau_minutes),
        "--wireless-uplink-mbps",
        str(args.wireless_uplink_mbps),
        "--radio-rtt-ms",
        str(args.radio_rtt_ms),
        "--requests-per-update",
        str(args.requests_per_update),
        "--rollout-unit",
        args.rollout_unit,
        "--eval-rollout-unit",
        args.eval_rollout_unit,
        "--reward-scale",
        str(args.reward_scale),
        "--reward-mode",
        args.reward_mode,
        "--fast-policy-kind",
        args.fast_policy_kind,
        "--max-replicas-per-stage",
        str(args.replicas_per_stage),
        "--eval-seeds",
        str(args.eval_seeds),
        "--eval-requests",
        str(args.eval_requests),
        "--device",
        args.device,
        "--progress-interval-seconds",
        str(args.progress_interval_seconds),
        "--run-root",
        str(root),
        "--save-best",
        *fixed,
    ]
    if args.mean_requests_per_minute is not None:
        common.extend(["--mean-requests-per-minute", str(args.mean_requests_per_minute)])

    baseline_cmd = [
        *common,
        "--run-name",
        "phase0_baseline_eval",
        "--updates",
        "0",
        "--eval-baseline",
        "--deterministic-eval",
    ]
    fast_run_name = "phase1_fast_only"
    fast_cmd = [
        *common,
        "--run-name",
        fast_run_name,
        "--train-mode",
        "fast-only",
        "--fast-bc-requests",
        str(args.fast_bc_requests),
        "--fast-bc-epochs",
        str(args.fast_bc_epochs),
        "--updates",
        str(args.fast_updates),
        "--eval-interval",
        str(max(args.fast_updates // 10, 1)),
    ]
    fast_best = root / fast_run_name / "checkpoints" / "best.pt"
    joint_cmd = [
        *common,
        "--run-name",
        "phase2_joint",
        "--train-mode",
        "joint",
        "--load-checkpoint",
        str(fast_best),
        "--updates",
        str(args.joint_updates),
        "--eval-interval",
        str(max(args.joint_updates // 10, 1)),
    ]

    for label, command in [
        ("phase0_baseline_eval", baseline_cmd),
        ("phase1_fast_only", fast_cmd),
        ("phase2_joint", joint_cmd),
    ]:
        print(f"\n=== {label} ===")
        print(" ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
