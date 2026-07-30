from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from edge_drl.agents.hierarchical import FastGreedyScheduler, SlowGreedyDeploymentPolicy, build_baseline_agent
from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical dual-agent PPO for edge services.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fixed-scenario", action="store_true")
    parser.add_argument("--num-users", type=int, default=10_000)
    parser.add_argument("--num-edge-nodes", type=int, default=16)
    parser.add_argument("--num-service-types", type=int, default=3)
    parser.add_argument("--episode-hours", type=int, default=8)
    parser.add_argument("--mean-requests-per-minute", type=float, default=None)
    parser.add_argument("--active-user-ratio", type=float, default=0.15)
    parser.add_argument("--active-user-request-rate-per-minute", type=float, default=1.5)
    parser.add_argument("--traffic-scale", type=float, default=1.0)
    parser.add_argument("--load-ewma-tau-minutes", type=float, default=1.0)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--requests-per-update", type=int, default=4096)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--reward-mode", choices=["latency", "greedy-advantage", "mixed"], default="latency")
    parser.add_argument("--mixed-latency-weight", type=float, default=0.1)
    parser.add_argument("--train-mode", choices=["joint", "fast-only"], default="joint")
    parser.add_argument("--replicas-per-stage", type=int, default=5)
    parser.add_argument("--slow-lr", type=float, default=3e-4)
    parser.add_argument("--fast-lr", type=float, default=3e-4)
    parser.add_argument("--slow-k-epochs", type=int, default=3)
    parser.add_argument("--fast-k-epochs", type=int, default=4)
    parser.add_argument("--slow-entropy-coef", type=float, default=0.001)
    parser.add_argument("--fast-entropy-coef", type=float, default=0.0)
    parser.add_argument("--slow-target-kl", type=float, default=0.03)
    parser.add_argument("--fast-target-kl", type=float, default=0.03)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-checkpoint", type=str, default="")
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--eval-baseline", action="store_true")
    parser.add_argument("--eval-requests", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--fast-bc-requests", type=int, default=0)
    parser.add_argument("--fast-bc-epochs", type=int, default=3)
    parser.add_argument("--run-root", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--append-log", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace, seed_offset: int = 0) -> EdgeComputingEnv:
    return EdgeComputingEnv(
        EdgeEnvConfig(
            seed=args.seed + seed_offset,
            scenario_seed=args.seed if getattr(args, "fixed_scenario", False) else None,
            num_users=args.num_users,
            num_edge_nodes=args.num_edge_nodes,
            num_service_types=args.num_service_types,
            episode_hours=args.episode_hours,
            mean_requests_per_minute=args.mean_requests_per_minute,
            active_user_ratio=args.active_user_ratio,
            active_user_request_rate_per_minute=args.active_user_request_rate_per_minute,
            traffic_scale=args.traffic_scale,
            load_ewma_tau_minutes=args.load_ewma_tau_minutes,
        )
    )


def traffic_rate_summary(env: EdgeComputingEnv) -> dict[str, float]:
    rates = []
    original_time = env.current_time_minute
    for minute in range(24 * 60):
        env.current_time_minute = float(minute)
        rates.append(env._arrival_rate_per_minute())
    env.current_time_minute = original_time
    values = np.asarray(rates, dtype=np.float64)
    return {
        "avg_requests_per_second": float(values.mean() / 60.0),
        "min_requests_per_second": float(values.min() / 60.0),
        "max_requests_per_second": float(values.max() / 60.0),
        "expected_requests_per_day": float(values.sum()),
    }


def rollout(
    env: EdgeComputingEnv,
    agent: HierarchicalPPOAgent,
    max_requests: int,
    args: argparse.Namespace | None = None,
    deterministic: bool = False,
    record: bool = True,
    reward_scale: float = 1.0,
    train_mode: str = "joint",
    frozen_slow_policy: SlowGreedyDeploymentPolicy | None = None,
) -> dict[str, float]:
    if args is None:
        args = argparse.Namespace(reward_mode="latency", mixed_latency_weight=0.1)
    env.reset()
    if frozen_slow_policy is None:
        frozen_slow_policy = SlowGreedyDeploymentPolicy()
    greedy_scheduler = FastGreedyScheduler()
    rewards: list[float] = []
    train_rewards: list[float] = []
    latencies: list[float] = []
    greedy_latencies: list[float] = []
    window_latencies: dict[int, list[float]] = {}
    invalid = 0
    while not env.done and env.metrics["requests"] < max_requests:
        request = env.current_request
        assert request is not None
        if train_mode == "fast-only":
            if env.needs_deployment_update:
                env.apply_deployment(frozen_slow_policy.act(env))
            action = agent.fast_agent.schedule(env, deterministic=deterministic, record=record)
        else:
            action = agent.act(env, deterministic=deterministic, record=record)
        deployment_window = int(env.metrics["deployment_updates"])
        greedy_info = None
        if record and getattr(args, "reward_mode", "latency") in {"greedy-advantage", "mixed"}:
            greedy_action = greedy_scheduler.act(env, request)
            greedy_info = env.evaluate_schedule(request, greedy_action)
        _, reward, done, info = env.step(action)
        train_reward = _training_reward(args, env_reward=reward, policy_info=info, greedy_info=greedy_info)
        if record:
            agent.observe_step_reward(train_reward * reward_scale, stage_count=len(request.stage_compute_gcycles), done=done)
        rewards.append(float(reward))
        train_rewards.append(float(train_reward))
        latencies.append(float(info["latency_s"]))
        window_latencies.setdefault(deployment_window, []).append(float(info["latency_s"]))
        if greedy_info is not None:
            greedy_latencies.append(float(greedy_info["latency_s"]))
        invalid += int(not info["valid"])
    if record and env.metrics["requests"] > 0:
        agent.flush_slow_window_reward(done=True)
    window_stats = _deployment_window_latency_stats(window_latencies)
    return {
        "requests": float(env.metrics["requests"]),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_train_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
        "avg_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "avg_greedy_latency_s": float(np.mean(greedy_latencies)) if greedy_latencies else float("nan"),
        "invalid_actions": float(invalid),
        "deadline_violation_rate": float(env.metrics["deadline_violations"] / max(env.metrics["requests"], 1.0)),
        "deployment_updates": float(env.metrics["deployment_updates"]),
        **window_stats,
    }


def _deployment_window_latency_stats(window_latencies: dict[int, list[float]]) -> dict[str, float]:
    non_empty = [(window, values) for window, values in sorted(window_latencies.items()) if values]
    if not non_empty:
        return {
            "first_window_avg_latency_s": float("nan"),
            "last_window_avg_latency_s": float("nan"),
            "window_latency_delta_s": float("nan"),
        }
    first_avg = float(np.mean(non_empty[0][1]))
    last_avg = float(np.mean(non_empty[-1][1]))
    return {
        "first_window_avg_latency_s": first_avg,
        "last_window_avg_latency_s": last_avg,
        "window_latency_delta_s": last_avg - first_avg,
    }


def _training_reward(
    args: argparse.Namespace,
    *,
    env_reward: float,
    policy_info: dict[str, object],
    greedy_info: dict[str, object] | None,
) -> float:
    mode = getattr(args, "reward_mode", "latency")
    if mode == "latency" or greedy_info is None:
        return -float(policy_info["latency_s"])
    advantage = float(greedy_info["latency_s"]) - float(policy_info["latency_s"])
    if mode == "greedy-advantage":
        return advantage
    return advantage - args.mixed_latency_weight * float(policy_info["latency_s"])


def evaluate_agent(
    args: argparse.Namespace,
    agent: HierarchicalPPOAgent,
    *,
    seed_base: int,
    max_requests: int,
    train_mode: str,
) -> dict[str, float]:
    runs = []
    for seed_idx in range(args.eval_seeds):
        eval_env = build_env(args, seed_offset=seed_base + seed_idx)
        runs.append(
            rollout(
                eval_env,
                agent,
                max_requests,
                args=args,
                deterministic=True,
                record=False,
                train_mode=train_mode,
            )
        )
    avg_latencies = np.array([r["avg_latency_s"] for r in runs], dtype=np.float64)
    p95_latencies = np.array([r["p95_latency_s"] for r in runs], dtype=np.float64)
    invalid_actions = np.array([r["invalid_actions"] for r in runs], dtype=np.float64)
    violation_rates = np.array([r["deadline_violation_rate"] for r in runs], dtype=np.float64)
    deployment_updates = np.array([r["deployment_updates"] for r in runs], dtype=np.float64)
    first_window_latencies = np.array([r["first_window_avg_latency_s"] for r in runs], dtype=np.float64)
    last_window_latencies = np.array([r["last_window_avg_latency_s"] for r in runs], dtype=np.float64)
    return {
        "eval_avg_latency_s": float(avg_latencies.mean()),
        "eval_avg_latency_std": float(avg_latencies.std()),
        "eval_p95_latency_s": float(p95_latencies.mean()),
        "eval_invalid_actions": float(invalid_actions.mean()),
        "eval_deadline_violation_rate": float(violation_rates.mean()),
        "eval_deployment_updates": float(deployment_updates.mean()),
        "eval_first_window_avg_latency_s": float(np.nanmean(first_window_latencies)),
        "eval_last_window_avg_latency_s": float(np.nanmean(last_window_latencies)),
        "eval_window_latency_delta_s": float(np.nanmean(last_window_latencies - first_window_latencies)),
    }


def evaluate_policy_diagnostics(
    args: argparse.Namespace,
    agent: HierarchicalPPOAgent,
    *,
    seed_base: int,
    max_requests: int,
    train_mode: str,
    previous_actions: list[int] | None,
) -> tuple[dict[str, float], list[int]]:
    deterministic_actions: list[int] = []
    entropies: list[float] = []
    top1_probs: list[float] = []
    top1_margins: list[float] = []
    deterministic_latencies: list[float] = []
    stochastic_latencies: list[float] = []
    slow_policy = SlowGreedyDeploymentPolicy()

    for seed_idx in range(args.eval_seeds):
        env = build_env(args, seed_offset=seed_base + seed_idx)
        env.reset()
        while not env.done and env.metrics["requests"] < max_requests:
            request = env.current_request
            assert request is not None
            if train_mode == "fast-only":
                if env.needs_deployment_update:
                    env.apply_deployment(slow_policy.act(env))
            else:
                agent.maybe_update_deployment(env, deterministic=True, record=False)
            action, stage_stats = agent.fast_agent.schedule_with_diagnostics(env, request)
            _, _, _, info = env.step(action)
            deterministic_latencies.append(float(info["latency_s"]))
            deterministic_actions.extend(action)
            entropies.extend(float(s["entropy"]) for s in stage_stats)
            top1_probs.extend(float(s["top1_prob"]) for s in stage_stats)
            top1_margins.extend(float(s["top1_margin"]) for s in stage_stats)

        stochastic_env = build_env(args, seed_offset=seed_base + seed_idx)
        stochastic_stats = rollout(
            stochastic_env,
            agent,
            max_requests,
            args=args,
            deterministic=False,
            record=False,
            train_mode=train_mode,
        )
        stochastic_latencies.append(stochastic_stats["avg_latency_s"])

    action_change_rate = float("nan")
    if previous_actions is not None:
        common = min(len(previous_actions), len(deterministic_actions))
        if common > 0:
            action_change_rate = float(
                np.mean(np.asarray(previous_actions[:common]) != np.asarray(deterministic_actions[:common]))
            )

    diagnostics = {
        "eval_policy_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "eval_top1_prob": float(np.mean(top1_probs)) if top1_probs else float("nan"),
        "eval_top1_margin": float(np.mean(top1_margins)) if top1_margins else float("nan"),
        "eval_action_change_rate": action_change_rate,
        "eval_stochastic_avg_latency_s": float(np.mean(stochastic_latencies)) if stochastic_latencies else float("nan"),
        "eval_deterministic_avg_latency_s": float(np.mean(deterministic_latencies)) if deterministic_latencies else float("nan"),
    }
    return diagnostics, deterministic_actions


def rollout_baseline(env: EdgeComputingEnv, max_requests: int) -> dict[str, float]:
    agent = build_baseline_agent()
    env.reset()
    rewards: list[float] = []
    latencies: list[float] = []
    invalid = 0
    while not env.done and env.metrics["requests"] < max_requests:
        action = agent.act(env)
        _, reward, _, info = env.step(action)
        rewards.append(float(reward))
        latencies.append(float(info["latency_s"]))
        invalid += int(not info["valid"])
    return {
        "requests": float(env.metrics["requests"]),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "invalid_actions": float(invalid),
        "deadline_violation_rate": float(env.metrics["deadline_violations"] / max(env.metrics["requests"], 1.0)),
    }


def pretrain_fast_agent(
    args: argparse.Namespace,
    agent: HierarchicalPPOAgent,
    *,
    requests: int,
    epochs: int,
) -> dict[str, float]:
    if requests <= 0:
        return {"bc_loss": 0.0, "bc_accuracy": 0.0, "bc_samples": 0.0}

    slow_policy = SlowGreedyDeploymentPolicy()
    expert = FastGreedyScheduler()
    states = []
    masks = []
    actions = []
    collected_requests = 0
    episode_idx = 0
    while collected_requests < requests:
        env = build_env(args, seed_offset=40_000 + episode_idx)
        env.reset()
        episode_idx += 1
        while not env.done and collected_requests < requests:
            if env.needs_deployment_update:
                env.apply_deployment(slow_policy.act(env))
            request = env.current_request
            assert request is not None
            expert_nodes = expert.act(env, request)
            partial_nodes: list[int] = []
            for stage_id, node_id in enumerate(expert_nodes):
                state = agent.fast_agent._build_state(env, request, stage_id, partial_nodes)
                mask = agent.fast_agent._build_mask(env, request, stage_id, partial_nodes)
                if mask[node_id]:
                    states.append(state)
                    masks.append(mask)
                    actions.append(node_id)
                partial_nodes.append(node_id)
            _, _, _, _ = env.step(expert_nodes)
            collected_requests += 1

    if not actions:
        return {"bc_loss": 0.0, "bc_accuracy": 0.0, "bc_samples": 0.0}
    metrics = agent.fast_agent.ppo.behavior_clone(
        np.stack(states),
        np.stack(masks),
        np.asarray(actions, dtype=np.int64),
        epochs=epochs,
    )
    metrics["bc_samples"] = float(len(actions))
    return metrics


def save_checkpoint(agent: HierarchicalPPOAgent, path: Path, metadata: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "slow_agent": agent.slow_agent.ppo.policy.state_dict(),
            "fast_agent": agent.fast_agent.ppo.policy.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(agent: HierarchicalPPOAgent, path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=agent.fast_agent.ppo.device)
    if "slow_agent" in checkpoint:
        agent.slow_agent.ppo.policy.load_state_dict(checkpoint["slow_agent"])
    if "fast_agent" in checkpoint:
        agent.fast_agent.ppo.policy.load_state_dict(checkpoint["fast_agent"])
    return checkpoint.get("metadata", {})


def append_log(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_single_row_csv(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def make_run_paths(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    mode_tag = args.train_mode.replace("-", "_")
    run_name = args.run_name or (
        f"dual_ppo_{mode_tag}_u{args.num_users}_n{args.num_edge_nodes}_"
        f"s{args.num_service_types}_seed{args.seed}"
    )
    run_dir = Path(args.run_root) / run_name
    log_dir = Path(args.log_dir) if args.log_dir else run_dir / "logs"
    save_dir = Path(args.save_dir) if args.save_dir else run_dir / "checkpoints"
    return run_name, run_dir, log_dir, save_dir


def write_metadata(path: Path, args: argparse.Namespace, bc_metrics: dict[str, float], loaded_metadata: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": vars(args),
        "bc_metrics": bc_metrics,
        "loaded_checkpoint_metadata": loaded_metadata,
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "reference_style": "DRL-AC-Allocation sequential masked PPO with staged train/eval separation",
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)
    env.reset()
    traffic = traffic_rate_summary(env)
    agent = HierarchicalPPOAgent.from_env(
        env,
        device=args.device,
        replicas_per_stage=args.replicas_per_stage,
        slow_lr=args.slow_lr,
        fast_lr=args.fast_lr,
        slow_k_epochs=args.slow_k_epochs,
        fast_k_epochs=args.fast_k_epochs,
        slow_entropy_coef=args.slow_entropy_coef,
        fast_entropy_coef=args.fast_entropy_coef,
        slow_target_kl=args.slow_target_kl,
        fast_target_kl=args.fast_target_kl,
    )
    loaded_metadata: dict[str, object] = {}
    if args.load_checkpoint:
        loaded_metadata = load_checkpoint(agent, Path(args.load_checkpoint))
    bc_metrics = pretrain_fast_agent(
        args,
        agent,
        requests=args.fast_bc_requests,
        epochs=args.fast_bc_epochs,
    )

    print("Hierarchical dual-agent PPO")
    print(f"  reference_style=DRL-AC-Allocation sequential masked PPO")
    print(f"  users={args.num_users}, nodes={args.num_edge_nodes}, services={args.num_service_types}")
    print(
        "  traffic avg={:.2f}/s min={:.2f}/s peak={:.2f}/s expected_day={:.0f}".format(
            traffic["avg_requests_per_second"],
            traffic["min_requests_per_second"],
            traffic["max_requests_per_second"],
            traffic["expected_requests_per_day"],
        )
    )
    print(f"  train_mode={args.train_mode}")
    print(f"  reward_mode={args.reward_mode}")
    print(f"  optimizer_reward_scale={args.reward_scale}")
    print(
        "  ppo slow_lr={} fast_lr={} slow_entropy={} fast_entropy={}".format(
            args.slow_lr,
            args.fast_lr,
            args.slow_entropy_coef,
            args.fast_entropy_coef,
        )
    )
    print("  slow_agent=service deployment every 240 minutes")
    print("  fast_agent=stage scheduling per task request")
    if args.fast_bc_requests > 0:
        print(
            "  fast_bc samples={} loss={:.4f} accuracy={:.4f}".format(
                int(bc_metrics["bc_samples"]),
                bc_metrics["bc_loss"],
                bc_metrics["bc_accuracy"],
            )
    )
    start = datetime.now().replace(microsecond=0)
    run_name, run_dir, log_dir, save_dir = make_run_paths(args)
    log_path = log_dir / "training.csv"
    if log_path.exists() and not args.append_log:
        log_path.unlink()
    write_metadata(run_dir, args, bc_metrics, loaded_metadata)
    best_latency = float("inf")
    previous_eval_actions: list[int] | None = None

    if args.eval_baseline:
        baseline_env = build_env(args, seed_offset=20_000)
        baseline_stats = rollout_baseline(baseline_env, args.eval_requests)
        write_single_row_csv(run_dir / "evaluation" / "baseline.csv", baseline_stats)
        print(
            "baseline requests={} avg_latency={:.4f}s p95_latency={:.4f}s invalid={} violation_rate={:.4f}".format(
                int(baseline_stats["requests"]),
                baseline_stats["avg_latency_s"],
                baseline_stats["p95_latency_s"],
                int(baseline_stats["invalid_actions"]),
                baseline_stats["deadline_violation_rate"],
            )
        )

    if args.eval_interval:
        eval_stats = evaluate_agent(
            args,
            agent,
            seed_base=30_000,
            max_requests=args.eval_requests,
            train_mode=args.train_mode,
        )
        diagnostic_stats, previous_eval_actions = evaluate_policy_diagnostics(
            args,
            agent,
            seed_base=30_000,
            max_requests=args.eval_requests,
            train_mode=args.train_mode,
            previous_actions=None,
        )
        initial_row = {
            "update": 0,
            "requests": 0,
            "avg_reward": np.nan,
            "avg_train_reward": np.nan,
            "avg_latency_s": np.nan,
            "p95_latency_s": np.nan,
            "avg_greedy_latency_s": np.nan,
            "invalid_actions": 0,
            "deadline_violation_rate": np.nan,
            "deployment_updates": 0,
            "first_window_avg_latency_s": np.nan,
            "last_window_avg_latency_s": np.nan,
            "window_latency_delta_s": np.nan,
            "slow_loss": 0.0,
            "slow_policy_loss": 0.0,
            "slow_value_loss": 0.0,
            "slow_approx_kl": 0.0,
            "fast_loss": 0.0,
            "fast_policy_loss": 0.0,
            "fast_value_loss": 0.0,
            "fast_approx_kl": 0.0,
            "eval_avg_latency_s": eval_stats["eval_avg_latency_s"],
            "eval_avg_latency_std": eval_stats["eval_avg_latency_std"],
            "eval_p95_latency_s": eval_stats["eval_p95_latency_s"],
            "eval_invalid_actions": eval_stats["eval_invalid_actions"],
            "eval_deadline_violation_rate": eval_stats["eval_deadline_violation_rate"],
            "eval_deployment_updates": eval_stats["eval_deployment_updates"],
            "eval_first_window_avg_latency_s": eval_stats["eval_first_window_avg_latency_s"],
            "eval_last_window_avg_latency_s": eval_stats["eval_last_window_avg_latency_s"],
            "eval_window_latency_delta_s": eval_stats["eval_window_latency_delta_s"],
            **diagnostic_stats,
        }
        append_log(log_path, initial_row)
        print(
            "update=000 eval_mean_latency={:.4f}s eval_std={:.4f}s eval_p95={:.4f}s invalid={:.2f}".format(
                eval_stats["eval_avg_latency_s"],
                eval_stats["eval_avg_latency_std"],
                eval_stats["eval_p95_latency_s"],
                eval_stats["eval_invalid_actions"],
            )
        )
        if args.save_best:
            best_latency = eval_stats["eval_avg_latency_s"]
            save_checkpoint(
                agent,
                save_dir / "best.pt",
                {
                    "update": 0,
                    "avg_latency_s": best_latency,
                    "avg_reward": np.nan,
                    "run_name": run_name,
                    "train_mode": args.train_mode,
                },
            )

    for update in range(args.updates):
        env = build_env(args, seed_offset=update)
        stats = rollout(
            env,
            agent,
            args.requests_per_update,
            args=args,
            reward_scale=args.reward_scale,
            train_mode=args.train_mode,
        )
        if args.train_mode == "fast-only":
            losses = {
                "slow": {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0},
                "fast": agent.fast_agent.update(),
            }
            agent.slow_agent.ppo.buffer.clear()
            agent.window_reward = 0.0
            agent.window_steps = 0
        else:
            losses = agent.update()
        eval_stats = {}
        if args.eval_interval and (update + 1) % args.eval_interval == 0:
            eval_stats = evaluate_agent(
                args,
                agent,
                seed_base=30_000,
                max_requests=args.eval_requests,
                train_mode=args.train_mode,
            )
            diagnostic_stats, previous_eval_actions = evaluate_policy_diagnostics(
                args,
                agent,
                seed_base=30_000,
                max_requests=args.eval_requests,
                train_mode=args.train_mode,
                previous_actions=previous_eval_actions,
            )
        else:
            diagnostic_stats = {}
        log_row = {
            "update": update + 1,
            "requests": int(stats["requests"]),
            "avg_reward": stats["avg_reward"],
            "avg_train_reward": stats["avg_train_reward"],
            "avg_latency_s": stats["avg_latency_s"],
            "p95_latency_s": stats["p95_latency_s"],
            "avg_greedy_latency_s": stats["avg_greedy_latency_s"],
            "invalid_actions": int(stats["invalid_actions"]),
            "deadline_violation_rate": stats["deadline_violation_rate"],
            "deployment_updates": int(stats["deployment_updates"]),
            "first_window_avg_latency_s": stats["first_window_avg_latency_s"],
            "last_window_avg_latency_s": stats["last_window_avg_latency_s"],
            "window_latency_delta_s": stats["window_latency_delta_s"],
            "slow_loss": losses["slow"]["loss"],
            "slow_policy_loss": losses["slow"]["policy_loss"],
            "slow_value_loss": losses["slow"]["value_loss"],
            "slow_approx_kl": losses["slow"].get("approx_kl", 0.0),
            "fast_loss": losses["fast"]["loss"],
            "fast_policy_loss": losses["fast"]["policy_loss"],
            "fast_value_loss": losses["fast"]["value_loss"],
            "fast_approx_kl": losses["fast"].get("approx_kl", 0.0),
            "eval_avg_latency_s": eval_stats.get("eval_avg_latency_s", np.nan),
            "eval_avg_latency_std": eval_stats.get("eval_avg_latency_std", np.nan),
            "eval_p95_latency_s": eval_stats.get("eval_p95_latency_s", np.nan),
            "eval_invalid_actions": eval_stats.get("eval_invalid_actions", np.nan),
            "eval_deadline_violation_rate": eval_stats.get("eval_deadline_violation_rate", np.nan),
            "eval_deployment_updates": eval_stats.get("eval_deployment_updates", np.nan),
            "eval_first_window_avg_latency_s": eval_stats.get("eval_first_window_avg_latency_s", np.nan),
            "eval_last_window_avg_latency_s": eval_stats.get("eval_last_window_avg_latency_s", np.nan),
            "eval_window_latency_delta_s": eval_stats.get("eval_window_latency_delta_s", np.nan),
            "eval_policy_entropy": diagnostic_stats.get("eval_policy_entropy", np.nan),
            "eval_top1_prob": diagnostic_stats.get("eval_top1_prob", np.nan),
            "eval_top1_margin": diagnostic_stats.get("eval_top1_margin", np.nan),
            "eval_action_change_rate": diagnostic_stats.get("eval_action_change_rate", np.nan),
            "eval_stochastic_avg_latency_s": diagnostic_stats.get("eval_stochastic_avg_latency_s", np.nan),
            "eval_deterministic_avg_latency_s": diagnostic_stats.get("eval_deterministic_avg_latency_s", np.nan),
        }
        append_log(log_path, log_row)
        print(
            "update={:03d} requests={} avg_reward={:.4f} avg_latency={:.4f}s "
            "train_reward={:.4f} invalid={} deployments={} slow_loss={:.4f} fast_loss={:.4f}".format(
                update + 1,
                int(stats["requests"]),
                stats["avg_reward"],
                stats["avg_latency_s"],
                stats["avg_train_reward"],
                int(stats["invalid_actions"]),
                int(stats["deployment_updates"]),
                losses["slow"]["loss"],
                losses["fast"]["loss"],
            )
        )
        if eval_stats:
            print(
                "  eval_mean_latency={:.4f}s eval_std={:.4f}s eval_p95={:.4f}s invalid={:.2f} entropy={:.4f} action_change={:.4f}".format(
                    eval_stats["eval_avg_latency_s"],
                    eval_stats["eval_avg_latency_std"],
                    eval_stats["eval_p95_latency_s"],
                    eval_stats["eval_invalid_actions"],
                    diagnostic_stats.get("eval_policy_entropy", np.nan),
                    diagnostic_stats.get("eval_action_change_rate", np.nan),
                )
            )
        selection_latency = eval_stats.get("eval_avg_latency_s", stats["avg_latency_s"])
        if args.save_best and selection_latency < best_latency:
            best_latency = selection_latency
            save_checkpoint(
                agent,
                save_dir / "best.pt",
                {
                    "update": update + 1,
                    "avg_latency_s": best_latency,
                    "avg_reward": stats["avg_reward"],
                    "run_name": run_name,
                    "train_mode": args.train_mode,
                },
            )

    if args.deterministic_eval:
        stats = evaluate_agent(
            args,
            agent,
            seed_base=10_000,
            max_requests=args.eval_requests,
            train_mode=args.train_mode,
        )
        print(
            "eval seeds={} requests_per_seed={} avg_latency={:.4f}s std={:.4f}s p95_latency={:.4f}s invalid={:.2f} violation_rate={:.4f}".format(
                args.eval_seeds,
                args.eval_requests,
                stats["eval_avg_latency_s"],
                stats["eval_avg_latency_std"],
                stats["eval_p95_latency_s"],
                stats["eval_invalid_actions"],
                stats["eval_deadline_violation_rate"],
            )
        )

    if args.updates > 0:
        save_checkpoint(
            agent,
            save_dir / "last.pt",
            {
                "update": args.updates,
                "run_name": run_name,
                "train_mode": args.train_mode,
            },
        )
        print(f"log={log_path}")
        print(f"checkpoint={save_dir / 'last.pt'}")
    print(f"elapsed={datetime.now().replace(microsecond=0) - start}")


if __name__ == "__main__":
    main()
