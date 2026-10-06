"""Train the single-policy end-to-end PPO comparison baseline.

The physical environment and demand distribution are inherited from an
explicit Proposed checkpoint.  One shared PPO policy controls deployment,
stage routing, and continuous resource priorities. The environment uses
``learned-share`` capacity normalization, never the analytical KKT allocator.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from edge_drl.agents.unified import UnifiedPPOAgent
from edge_drl.comparison.checkpoint import load_checkpoint_configuration
from edge_drl.env.environment import EdgeComputingEnv
from train_dual_ppo import (
    TrainingRandomStreams,
    build_training_env,
    load_assignments_for_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the one-policy UnifiedPPO comparison baseline"
    )
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--updates", type=int, default=320)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--sampled-seconds-per-window", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--environment-gamma",
        type=float,
        default=0.95,
        help="Discount between real sampled settlement steps inside one deployment window",
    )
    parser.add_argument("--k-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--run-root", default="runs/unified_ppo")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--best-checkpoint-window", type=int, default=10)
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.updates < 1 or args.episodes_per_update < 1:
        parser.error("--updates and --episodes-per-update must be positive")
    if args.sampled_seconds_per_window < 1:
        parser.error("--sampled-seconds-per-window must be positive")
    if args.hidden_dim < 16 or args.minibatch_size < 1 or args.k_epochs < 1:
        parser.error("invalid PPO network or batch setting")
    if not np.isfinite(args.lr) or args.lr <= 0.0:
        parser.error("--lr must be finite and positive")
    if not 0.0 <= args.environment_gamma <= 1.0:
        parser.error("--environment-gamma must be in [0,1]")
    if args.target_kl <= 0.0:
        parser.error("--target-kl must be positive")
    if args.best_checkpoint_window < 1:
        parser.error("--best-checkpoint-window must be positive")
    if any(not np.isfinite(value) or value < 0.0
           for value in (args.entropy_coef, args.value_coef, args.reward_scale)) or args.reward_scale == 0:
        parser.error("loss coefficients must be finite and nonnegative; reward scale must be positive")
    return args


def weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    denominator = float(weight.sum())
    return float(np.sum(array * weight) / denominator) if denominator > 0.0 else 0.0


def weighted_percentile(values: list[float], weights: list[float], q: float) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    order = np.argsort(array)
    array, weight = array[order], weight[order]
    cutoff = float(weight.sum()) * q / 100.0
    index = int(np.searchsorted(np.cumsum(weight), cutoff, side="left"))
    return float(array[min(index, len(array) - 1)])


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    weights = [float(row["requests"]) for row in rows]
    return {
        "requests": float(sum(weights)),
        "request_events": float(sum(float(row["request_events"]) for row in rows)),
        "settlement_steps": float(sum(float(row["settlement_steps"]) for row in rows)),
        "logical_steps": float(sum(float(row["logical_steps"]) for row in rows)),
        "avg_train_reward": weighted_mean(
            [float(row["avg_train_reward"]) for row in rows], weights
        ),
        "avg_reward": weighted_mean([float(row["avg_reward"]) for row in rows], weights),
        "avg_latency_s": weighted_mean([float(row["avg_latency_s"]) for row in rows], weights),
        "p95_latency_s": weighted_mean([float(row["p95_latency_s"]) for row in rows], weights),
        "deadline_violation_rate": weighted_mean(
            [float(row["deadline_violation_rate"]) for row in rows], weights
        ),
        "invalid_action_rate": weighted_mean(
            [float(row["invalid_action_rate"]) for row in rows], weights
        ),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> Path:
    """Save without destroying the previous checkpoint on transient Windows locks.

    Windows refuses ``os.replace`` while another process (including antivirus
    or an indexer) briefly has the destination open.  Retry the atomic replace;
    if the lock persists, retain exactly one ``.pending`` checkpoint instead of
    terminating a long training run or accumulating one fallback per update.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    delay_seconds = 0.05
    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            temporary.replace(path)
            return path
        except PermissionError as error:
            last_error = error
            if attempt < 9:
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2.0, 1.0)

    pending = path.with_suffix(path.suffix + ".pending")
    try:
        temporary.replace(pending)
    except PermissionError:
        assert last_error is not None
        raise last_error
    print(
        f"WARNING: {path.name} remained locked; newest valid checkpoint was "
        f"preserved as {pending.name}. The next update will retry {path.name}.",
        flush=True,
    )
    return pending


def checkpoint_payload(
    agent: UnifiedPPOAgent,
    cli: argparse.Namespace,
    reference_args: dict[str, Any],
    streams: TrainingRandomStreams,
    *,
    completed_updates: int,
    completed_rollouts: int,
    completed_episodes: int,
    best_score: float,
    reward_history: list[float],
) -> dict[str, Any]:
    metadata_args = {
        **vars(cli),
        "reference_checkpoint": str(Path(cli.reference_checkpoint).resolve()),
        "compute_allocation_mode": "learned-share",
        "link_allocation_mode": "learned-share",
        "agent_design": "single-policy-ledger-credit-v3",
    }
    return {
        "agent": agent.state_dict(),
        "metadata": {
            "args": metadata_args,
            "reference_args": reference_args,
            "completed_updates": completed_updates,
            "completed_rollouts": completed_rollouts,
            "completed_episodes": completed_episodes,
            "best_checkpoint_score": best_score,
            "reward_history": reward_history,
            "saved_at": datetime.now().replace(microsecond=0).isoformat(),
        },
        "training_random_streams": streams.state_dict(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def configure_learned_share(env: EdgeComputingEnv) -> None:
    env.config.compute_allocation_mode = "learned-share"
    env.config.link_allocation_mode = "learned-share"


def collect_window(
    env: EdgeComputingEnv,
    agent: UnifiedPPOAgent,
    *,
    sampled_seconds: int,
    deterministic: bool,
    record: bool,
    progress_label: str,
    progress_interval_seconds: float,
) -> dict[str, float]:
    window_seconds = float(env.config.deployment_interval_minutes * 60)
    represented_seconds = window_seconds / min(float(sampled_seconds), window_seconds)
    stop_minute = min(
        env.current_time_minute + env.config.deployment_interval_minutes,
        float(env.config.episode_minutes),
    )
    latencies: list[float] = []
    rewards: list[float] = []
    weights: list[float] = []
    deadline_violations = 0.0
    invalid = 0.0
    request_events = 0.0
    settlement_steps = 0
    logical_steps = 0.0
    last_progress = time.perf_counter()
    started = last_progress
    if record:
        agent.begin_window()

    while env.current_time_minute < stop_minute - 1e-9 and not env.done:
        step_seconds = min(
            represented_seconds,
            max((stop_minute - env.current_time_minute) * 60.0, 0.0),
        )
        if step_seconds < 1.0 - 1e-9:
            break
        start_index = agent.begin_environment_step()
        if env.needs_deployment_update:
            deployment = agent.plan_deployment(
                env,
                deterministic=deterministic,
                record=record,
            )
            env.apply_deployment(deployment)
        requests = list(env.current_requests)
        schedules, resource_weights = agent.schedule_batch(
            env,
            requests,
            deterministic=deterministic,
            record=record,
        )
        _, environment_reward, done, info = env.step(
            schedules,
            represented_seconds=step_seconds,
            batch_resource_weights=resource_weights,
        )
        group_latencies = [float(item["latency_s"]) for item in info["group_infos"]]
        group_weights = [float(item["request_count"]) for item in info["group_infos"]]
        step_latency = weighted_mean(group_latencies, group_weights)
        if record:
            agent.finish_environment_step(start_index, -step_latency, done=done, group_infos=info["group_infos"])
        latencies.extend(group_latencies)
        weights.extend(group_weights)
        rewards.extend([-value for value in group_latencies])
        deadline_violations += sum(
            float(latency > request.deadline_s) * weight
            for latency, request, weight in zip(group_latencies, requests, group_weights)
        )
        invalid += sum(
            float(not item["valid"]) * weight
            for item, weight in zip(info["group_infos"], group_weights)
        )
        request_events += float(info["request_event_count"]) * step_seconds
        settlement_steps += 1
        logical_steps += step_seconds

        now = time.perf_counter()
        if progress_interval_seconds > 0.0 and now - last_progress >= progress_interval_seconds:
            fraction = (env.current_time_minute % env.config.deployment_interval_minutes) / max(
                env.config.deployment_interval_minutes, 1
            )
            print(
                f"{progress_label} progress={fraction:.1%} "
                f"steps={settlement_steps}/{sampled_seconds} "
                f"requests={sum(weights):.0f} latency={weighted_mean(latencies, weights):.4f}s "
                f"elapsed={now-started:.1f}s",
                flush=True,
            )
            last_progress = now

    # Six 100-second samples represent a ten-minute window. Repeated binary
    # floating-point additions can finish at 59.999999999... minutes, leaving
    # ``env.done`` false and making every later rollout an empty window. Snap
    # only the numerical boundary; this does not skip any logical time.
    if env.current_time_minute >= float(env.config.episode_minutes) - 1e-9:
        env.current_time_minute = float(env.config.episode_minutes)
        env.current_requests = []
        env.current_request = None
    if record:
        agent.finish_window()

    requests_total = float(sum(weights))
    average_latency = weighted_mean(latencies, weights)
    return {
        "total_reward": float(sum(value * weight for value, weight in zip(rewards, weights))),
        "total_train_reward": float(sum(value * weight for value, weight in zip(rewards, weights))),
        "requests": requests_total,
        "request_events": request_events,
        "settlement_steps": float(settlement_steps),
        "logical_steps": logical_steps,
        "avg_reward": -average_latency,
        "avg_train_reward": -average_latency,
        "avg_latency_s": average_latency,
        "p95_latency_s": weighted_percentile(latencies, weights, 95.0),
        "deadline_violation_rate": deadline_violations / max(requests_total, 1.0),
        "invalid_action_rate": invalid / max(requests_total, 1.0),
        "ledger_settlement_error_s": agent.max_settlement_error if record else 0.0,
        "episode_complete": float(env.done),
    }


def main() -> None:
    cli = parse_args()
    reference_path, reference_args, _ = load_checkpoint_configuration(
        cli.reference_checkpoint
    )
    env_values = dict(reference_args)
    env_values.update(
        {
            "seed": cli.seed,
            "sampled_seconds_per_window": cli.sampled_seconds_per_window,
            "episode_minutes": int(reference_args.get("episode_minutes", 60)),
            "episode_hours": None,
        }
    )
    env_args = SimpleNamespace(**env_values)
    episode_minutes = int(env_args.episode_minutes)
    deployment_minutes = int(env_args.deployment_interval_minutes)
    if episode_minutes % deployment_minutes != 0:
        raise ValueError("reference episode must contain whole deployment windows")
    windows_per_episode = episode_minutes // deployment_minutes
    windows_per_update = windows_per_episode * cli.episodes_per_update

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    streams = TrainingRandomStreams(cli.seed)

    probe = build_training_env(
        env_args,
        rollout_idx=0,
        episode_idx=0,
        load_multiplier=1.0,
        load_group=0,
        demand_seed=cli.seed,
        environment_seed=cli.seed,
    )
    configure_learned_share(probe)
    probe.reset()
    agent = UnifiedPPOAgent.from_env(
        probe,
        hidden_dim=cli.hidden_dim,
        lr=cli.lr,
        environment_gamma=cli.environment_gamma,
        k_epochs=cli.k_epochs,
        entropy_coef=cli.entropy_coef,
        value_coef=cli.value_coef,
        target_kl=cli.target_kl,
        minibatch_size=cli.minibatch_size,
        reward_scale=cli.reward_scale,
        device=cli.device,
    )

    run_name = cli.run_name or (
        f"unified_ppo_v3_s{cli.sampled_seconds_per_window}_u{cli.updates}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir = Path(cli.run_root) / run_name
    if (run_dir / "logs" / "training.csv").exists():
        raise FileExistsError("Training output already exists; use a new --run-name, also when resuming")
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    snapshot_dir = checkpoint_dir / "snapshots"
    for directory in (log_dir, checkpoint_dir, snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    completed_updates = 0
    completed_rollouts = 0
    completed_episodes = 0
    best_score = -float("inf")
    reward_history: list[float] = []
    if cli.load_checkpoint:
        # Our own resumable checkpoint includes NumPy RNG state, not tensors
        # alone. Only load trusted, locally produced checkpoints here.
        checkpoint = torch.load(cli.load_checkpoint, map_location=cli.device, weights_only=False)
        metadata = dict(checkpoint.get("metadata", {}))
        saved_design = str(
            dict(metadata.get("args", {})).get("agent_design", "unknown")
        )
        if saved_design != "single-policy-ledger-credit-v3":
            raise ValueError(
                "--load-checkpoint is incompatible with macro-credit v2: "
                f"checkpoint agent_design={saved_design!r}. Start a fresh run; "
                "the earlier component-action GAE checkpoints must not be resumed."
            )
        agent.load_state_dict(checkpoint["agent"], load_optimizer=True)
        completed_updates = int(metadata.get("completed_updates", 0))
        completed_rollouts = int(metadata.get("completed_rollouts", 0))
        completed_episodes = int(metadata.get("completed_episodes", 0))
        best_score = float(metadata.get("best_checkpoint_score", -float("inf")))
        reward_history = [float(value) for value in metadata.get("reward_history", [])]
        if "training_random_streams" in checkpoint:
            streams.load_state_dict(checkpoint["training_random_streams"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state_all"])

    metadata = {
        "args": {
            **vars(cli),
            "reference_checkpoint": str(reference_path),
            "compute_allocation_mode": "learned-share",
            "link_allocation_mode": "learned-share",
            "agent_design": "single-policy-ledger-credit-v3",
        },
        "reference_args": reference_args,
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "description": (
            "One PPO actor-critic jointly learns deployment, stage routing, and "
            "resource priorities. No KKT optimizer is used."
        ),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("UnifiedPPO single-agent comparison")
    print(f"  reference={reference_path}")
    print(f"  run={run_dir.resolve()}")
    print(
        f"  episode={episode_minutes}min windows/episode={windows_per_episode} "
        f"episodes/update={cli.episodes_per_update} windows/update={windows_per_update}"
    )
    print(
        f"  sampled_steps/window={cli.sampled_seconds_per_window} "
        "allocation=learned-share continuous_compute_and_link_priorities KKT=disabled"
    )
    print(
        f"  policy=single conditional actor-critic node_action_dim={agent.action_dim} "
        f"obs_dim={agent.ppo.policy.shared[0].in_features} lr={cli.lr} "
        f"macro_gamma={cli.environment_gamma} reward_scale={cli.reward_scale}"
    )

    training_started = time.perf_counter()
    target_updates = completed_updates + cli.updates
    global_rollout = completed_rollouts
    episode_index = completed_episodes
    train_env: EdgeComputingEnv | None = None
    episode_rows: list[dict[str, Any]] = []

    for update_index in range(completed_updates, target_updates):
        update_started = time.perf_counter()
        assignments = load_assignments_for_update(
            env_args,
            update_index,
            windows_per_update,
            rollout_start_idx=global_rollout,
            rng=streams.load_rng,
        )
        window_rows: list[dict[str, Any]] = []
        for within_update, (load_multiplier, load_group) in enumerate(assignments, start=1):
            if train_env is None or train_env.done:
                demand_seed = streams.demand_seed_for_episode(
                    episode_index,
                    int(getattr(env_args, "scenario_refresh_episodes", 1)),
                )
                environment_seed = streams.next_environment_seed()
                train_env = build_training_env(
                    env_args,
                    rollout_idx=global_rollout,
                    episode_idx=episode_index,
                    load_multiplier=load_multiplier,
                    load_group=load_group,
                    demand_seed=demand_seed,
                    environment_seed=environment_seed,
                )
                configure_learned_share(train_env)
                train_env.reset()
            previous_multiplier = float(train_env.config.demand_load_multiplier)
            train_env.config.demand_load_multiplier = float(load_multiplier)
            train_env.config.demand_load_group = int(load_group)
            if not np.isclose(previous_multiplier, load_multiplier):
                train_env.current_requests = train_env._generate_current_second_requests()
                train_env.current_request = (
                    train_env.current_requests[0] if train_env.current_requests else None
                )
            episode_number = episode_index + 1
            # The simulated clock may be a few ulps below an exact ten-minute
            # boundary after six 100-second samples. Round the known window
            # index so logs remain 1..6 instead of repeating windows 4 and 5.
            window_number = int(
                round(
                    train_env.current_time_minute
                    / train_env.config.deployment_interval_minutes
                )
            ) + 1
            row = collect_window(
                train_env,
                agent,
                sampled_seconds=cli.sampled_seconds_per_window,
                deterministic=False,
                record=True,
                progress_label=(
                    f"update={update_index+1:03d}/{target_updates:03d} "
                    f"rollout={within_update:02d}/{windows_per_update:02d} "
                    f"episode={episode_number:03d} window={window_number:02d}/{windows_per_episode:02d}"
                ),
                progress_interval_seconds=cli.progress_interval_seconds,
            )
            expected_settlements = min(
                cli.sampled_seconds_per_window,
                int(train_env.config.deployment_interval_minutes * 60),
            )
            if int(row["settlement_steps"]) != expected_settlements:
                raise RuntimeError(
                    "UnifiedPPO collected an incomplete window: "
                    f"episode={episode_number}, window={window_number}, "
                    f"settlements={int(row['settlement_steps'])}, "
                    f"expected={expected_settlements}, "
                    f"time_minute={train_env.current_time_minute:.12f}. "
                    "Refusing to record or optimize an empty rollout."
                )
            global_rollout += 1
            rollout_row = {
                "rollout": global_rollout,
                "update": update_index + 1,
                "rollout_in_update": within_update,
                "episode": episode_number,
                "window": window_number,
                "training_phase": "unified",
                "demand_seed": int(train_env.config.scenario_seed),
                "environment_seed": int(train_env.config.seed),
                "load_multiplier": float(load_multiplier),
                "load_group": int(load_group),
                "start_minute": float((window_number - 1) * deployment_minutes),
                **row,
            }
            append_csv(log_dir / "rollout_metrics.csv", rollout_row)
            window_rows.append(rollout_row)
            episode_rows.append(rollout_row)
            print(
                f"  rollout={global_rollout:04d} update={update_index+1:03d} "
                f"episode={episode_number:03d} window={window_number:02d} "
                f"load={load_multiplier:.3f} requests={row['requests']:.0f} "
                f"reward={row['avg_train_reward']:.4f} latency={row['avg_latency_s']:.4f}s",
                flush=True,
            )
            if train_env.done:
                episode_summary = aggregate_rows(episode_rows)
                append_csv(
                    log_dir / "episode_metrics.csv",
                    {
                        "episode": episode_number,
                        "update_end": update_index + 1,
                        "rollout_start": int(episode_rows[0]["rollout"]),
                        "rollout_end": int(episode_rows[-1]["rollout"]),
                        "rollouts_collected": len(episode_rows),
                        "episode_complete": 1,
                        **episode_summary,
                    },
                )
                episode_rows.clear()
                episode_index += 1
                train_env = None

        ppo_metrics = agent.update(
            progress_label=f"update={update_index+1:03d}/{target_updates:03d} UnifiedPPO",
            progress_interval_seconds=cli.progress_interval_seconds,
        )
        update_summary = aggregate_rows(window_rows)
        reward_history.append(float(update_summary["avg_train_reward"]))
        rolling_window = min(cli.best_checkpoint_window, len(reward_history))
        rolling_score = float(np.mean(reward_history[-rolling_window:]))
        elapsed = time.perf_counter() - training_started
        update_elapsed = time.perf_counter() - update_started
        completed_now = update_index + 1
        training_row = {
            "update": completed_now,
            "training_phase": "unified",
            "episode_end": episode_index,
            "episodes_collected": cli.episodes_per_update,
            "rollouts_collected": len(window_rows),
            **update_summary,
            "rolling_reward": rolling_score,
            "unified_loss": ppo_metrics["loss"],
            "unified_policy_loss": ppo_metrics["policy_loss"],
            "unified_value_loss": ppo_metrics["value_loss"],
            "unified_entropy": ppo_metrics["entropy"],
            "unified_normalized_entropy": ppo_metrics["normalized_entropy"],
            "unified_resource_entropy": ppo_metrics["resource_entropy"],
            "unified_advantage_std": ppo_metrics["advantage_std"],
            "unified_approx_kl": ppo_metrics["approx_kl"],
            "unified_clip_fraction": ppo_metrics["clip_fraction"],
            "unified_epochs_completed": ppo_metrics["epochs_completed"],
            "unified_kl_early_stop": ppo_metrics["kl_early_stop"],
            "unified_optimizer_steps": ppo_metrics["optimizer_steps"],
            "unified_samples_seen_fraction": ppo_metrics["samples_seen_fraction"],
            "coverage_actions": ppo_metrics["coverage_actions"],
            "extra_actions": ppo_metrics["extra_actions"],
            "schedule_actions": ppo_metrics["schedule_actions"],
            "stop_actions": ppo_metrics["stop_actions"],
            "mean_deployment_replicas": ppo_metrics["mean_deployment_replicas"],
            "macro_return_mean": ppo_metrics["macro_return_mean"],
            "macro_return_std": ppo_metrics["macro_return_std"],
            "macro_effective_groups": ppo_metrics["macro_effective_groups"],
            "ledger_settlement_error_s": ppo_metrics["ledger_settlement_error_s"],
            "compute_priority_mean": ppo_metrics["compute_priority_mean"],
            "link_priority_mean": ppo_metrics["link_priority_mean"],
            "elapsed_seconds": elapsed,
            "update_seconds": update_elapsed,
        }
        append_csv(log_dir / "training.csv", training_row)

        payload = checkpoint_payload(
            agent,
            cli,
            reference_args,
            streams,
            completed_updates=completed_now,
            completed_rollouts=global_rollout,
            completed_episodes=episode_index,
            best_score=(max(best_score, rolling_score)
                        if rolling_window >= cli.best_checkpoint_window else best_score),
            reward_history=reward_history,
        )
        atomic_torch_save(payload, checkpoint_dir / "latest.pt")
        if rolling_window >= cli.best_checkpoint_window and rolling_score > best_score:
            best_score = rolling_score
            payload["metadata"]["best_checkpoint_score"] = best_score
            atomic_torch_save(payload, checkpoint_dir / "best.pt")
        if cli.checkpoint_interval > 0 and completed_now % cli.checkpoint_interval == 0:
            atomic_torch_save(
                payload,
                snapshot_dir / f"update_{completed_now:04d}.pt",
            )
        remaining = max(target_updates - completed_now, 0)
        eta = remaining * elapsed / max(completed_now - completed_updates, 1)
        print(
            f"update={completed_now:03d}/{target_updates:03d} "
            f"reward={update_summary['avg_train_reward']:.4f} "
            f"latency={update_summary['avg_latency_s']:.4f}s "
            f"p95={update_summary['p95_latency_s']:.4f}s "
            f"rollingR={rolling_score:.4f} entropy={ppo_metrics['entropy']:.3f} "
            f"kl={ppo_metrics['approx_kl']:.5f} actions={int(ppo_metrics['schedule_actions'])} "
            f"elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h",
            flush=True,
        )

    final_payload = checkpoint_payload(
        agent,
        cli,
        reference_args,
        streams,
        completed_updates=target_updates,
        completed_rollouts=global_rollout,
        completed_episodes=episode_index,
        best_score=best_score,
        reward_history=reward_history,
    )
    atomic_torch_save(final_payload, checkpoint_dir / "last.pt")
    if not (checkpoint_dir / "best.pt").exists():
        atomic_torch_save(final_payload, checkpoint_dir / "best.pt")
    print(f"training complete: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
