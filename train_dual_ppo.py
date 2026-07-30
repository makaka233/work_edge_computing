from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    parser.add_argument(
        "--scenario-refresh-episodes",
        type=int,
        default=1,
        help="Without --fixed-scenario, reuse one scenario instance for this many training episodes.",
    )
    parser.add_argument("--num-users", type=int, default=10_000)
    parser.add_argument("--num-edge-nodes", type=int, default=32)
    parser.add_argument("--num-service-types", type=int, default=10)
    parser.add_argument("--episode-hours", type=int, default=24)
    parser.add_argument("--mean-requests-per-minute", type=float, default=None)
    parser.add_argument("--active-user-ratio", type=float, default=0.15)
    parser.add_argument("--active-user-request-rate-per-minute", type=float, default=1.5)
    parser.add_argument("--traffic-scale", type=float, default=1.0)
    parser.add_argument("--request-aggregation-window-seconds", type=float, default=10.0)
    parser.add_argument("--max-representative-groups-per-window", type=int, default=16)
    parser.add_argument("--load-ewma-tau-minutes", type=float, default=1.0)
    parser.add_argument("--wireless-uplink-mbps", type=float, default=150.0)
    parser.add_argument("--radio-rtt-ms", type=float, default=10.0)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--requests-per-update", type=int, default=4096)
    parser.add_argument(
        "--rollout-unit",
        choices=["requests", "episode"],
        default="requests",
        help="Collect each PPO update by request count or by one full environment episode.",
    )
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--reward-mode", choices=["latency", "greedy-advantage", "mixed"], default="latency")
    parser.add_argument("--mixed-latency-weight", type=float, default=0.1)
    parser.add_argument("--train-mode", choices=["joint", "fast-only"], default="joint")
    parser.add_argument(
        "--replicas-per-stage",
        "--max-replicas-per-stage",
        dest="replicas_per_stage",
        type=int,
        default=5,
        help="Maximum replicas the slow PPO may place per service stage; actual count is learned with a STOP action.",
    )
    parser.add_argument("--fast-policy-kind", choices=["node_scorer", "gat_node_scorer"], default="gat_node_scorer")
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
    parser.add_argument(
        "--eval-rollout-unit",
        choices=["requests", "episode", "same"],
        default="requests",
        help="Use request-count eval, full-episode eval, or the same unit as training.",
    )
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--fast-bc-requests", type=int, default=0)
    parser.add_argument("--fast-bc-epochs", type=int, default=3)
    parser.add_argument("--run-root", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--append-log", action="store_true")
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=10.0,
        help="Print in-rollout terminal progress every N seconds. Use 0 to disable.",
    )
    return parser.parse_args()


def scenario_seed_for_offset(args: argparse.Namespace, seed_offset: int = 0, *, group_by_refresh: bool = False) -> int:
    if getattr(args, "fixed_scenario", False):
        return int(args.seed)
    refresh = max(int(getattr(args, "scenario_refresh_episodes", 1)), 1)
    scenario_offset = seed_offset // refresh if group_by_refresh else seed_offset
    return int(args.seed + scenario_offset)


def build_env(args: argparse.Namespace, seed_offset: int = 0, *, group_scenario_by_refresh: bool = False) -> EdgeComputingEnv:
    return EdgeComputingEnv(
        EdgeEnvConfig(
            seed=args.seed + seed_offset,
            scenario_seed=scenario_seed_for_offset(args, seed_offset, group_by_refresh=group_scenario_by_refresh),
            num_users=args.num_users,
            num_edge_nodes=args.num_edge_nodes,
            num_service_types=args.num_service_types,
            episode_hours=args.episode_hours,
            mean_requests_per_minute=args.mean_requests_per_minute,
            active_user_ratio=args.active_user_ratio,
            active_user_request_rate_per_minute=args.active_user_request_rate_per_minute,
            traffic_scale=args.traffic_scale,
            request_aggregation_window_seconds=args.request_aggregation_window_seconds,
            max_representative_groups_per_window=args.max_representative_groups_per_window,
            load_ewma_tau_minutes=args.load_ewma_tau_minutes,
            wireless_uplink_mbps=args.wireless_uplink_mbps,
            radio_rtt_ms=args.radio_rtt_ms,
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


class RolloutProgress:
    def __init__(
        self,
        *,
        label: str,
        target_requests: int | None,
        interval_seconds: float,
        episode_hours: int,
        rollout_unit: str,
        deployment_interval_minutes: int,
        overall_index: int | None = None,
        overall_total: int | None = None,
        overall_started_at: float | None = None,
    ):
        self.label = label
        self.target_requests = target_requests
        self.interval_seconds = max(interval_seconds, 0.0)
        self.episode_hours = episode_hours
        self.rollout_unit = rollout_unit
        self.deployment_interval_minutes = deployment_interval_minutes
        self.overall_index = overall_index
        self.overall_total = overall_total
        self.overall_started_at = overall_started_at
        self.started_at = time.monotonic()
        self.last_print_at = self.started_at
        self.printed = False

    def should_print(self) -> bool:
        if self.interval_seconds <= 0:
            return False
        now = time.monotonic()
        return not self.printed or now - self.last_print_at >= self.interval_seconds

    def maybe_print(
        self,
        env: EdgeComputingEnv,
        *,
        avg_reward: float = float("nan"),
        avg_train_reward: float = float("nan"),
        avg_latency_s: float | None = None,
    ) -> None:
        if self.should_print():
            self._print(
                env,
                now=time.monotonic(),
                avg_reward=avg_reward,
                avg_train_reward=avg_train_reward,
                avg_latency_s=avg_latency_s,
            )

    def finish(
        self,
        env: EdgeComputingEnv,
        *,
        avg_reward: float = float("nan"),
        avg_train_reward: float = float("nan"),
        avg_latency_s: float | None = None,
    ) -> None:
        if self.interval_seconds <= 0:
            return
        self._print(
            env,
            now=time.monotonic(),
            final=True,
            avg_reward=avg_reward,
            avg_train_reward=avg_train_reward,
            avg_latency_s=avg_latency_s,
        )

    def _print(
        self,
        env: EdgeComputingEnv,
        *,
        now: float,
        final: bool = False,
        avg_reward: float = float("nan"),
        avg_train_reward: float = float("nan"),
        avg_latency_s: float | None = None,
    ) -> None:
        requests = float(env.metrics.get("requests", 0.0))
        aggregate_events = int(env.metrics.get("aggregate_events", 0.0))
        sim_hours = env.current_time_minute / 60.0
        episode_fraction = sim_hours / max(float(self.episode_hours), 1e-9)
        if self.rollout_unit == "episode":
            progress = min(episode_fraction, 1.0)
        else:
            progress = min(requests / max(float(self.target_requests or 1), 1.0), 1.0)
        if avg_latency_s is None:
            avg_latency_s = env.metrics["total_latency_s"] / max(requests, 1.0)
        elapsed = max(now - self.started_at, 1e-9)
        eta = elapsed * (1.0 - progress) / max(progress, 1e-9) if progress > 0 else float("nan")
        total_windows = max(int(np.ceil(self.episode_hours * 60.0 / max(self.deployment_interval_minutes, 1))), 1)
        current_window = min(int(env.current_time_minute // max(self.deployment_interval_minutes, 1)) + 1, total_windows)
        simulated_seconds = max(env.current_time_minute * 60.0, 1e-9)
        sim_request_rate = requests / simulated_seconds
        wall_event_rate = aggregate_events / elapsed
        request_target = str(self.target_requests) if self.target_requests is not None else "episode"
        overall_text = ""
        if self.overall_index is not None and self.overall_total:
            overall_progress = (self.overall_index + progress) / max(float(self.overall_total), 1.0)
            overall_progress = min(max(overall_progress, 0.0), 1.0)
            if self.overall_started_at is None:
                total_elapsed = elapsed
            else:
                total_elapsed = max(now - self.overall_started_at, 1e-9)
            total_eta = total_elapsed * (1.0 - overall_progress) / max(overall_progress, 1e-9)
            overall_text = (
                f" | all [{_progress_bar(overall_progress, width=12)}] "
                f"{overall_progress * 100:4.1f}% ETA {_format_duration(total_eta)}"
            )
        line = (
            f"\r{self.label} [{_progress_bar(progress, width=24)}] {progress * 100:5.1f}% "
            f"| epR={_format_metric(avg_reward, 4)} "
            f"trainR={_format_metric(avg_train_reward, 4)} "
            f"Lat={avg_latency_s * 1000:6.1f}ms "
            f"| win={current_window}/{total_windows} deploy={int(env.metrics.get('deployment_updates', 0.0)):>2} "
            f"sim={sim_hours:4.1f}/{self.episode_hours}h "
            f"req={int(requests)}/{request_target} "
            f"ev={aggregate_events} "
            f"spd={wall_event_rate:4.1f}ev/s "
            f"ETA {_format_duration(eta)}"
            f"{overall_text}"
        )
        sys.stdout.write(line[:240])
        if final:
            sys.stdout.write("\n")
        sys.stdout.flush()
        self.printed = True
        self.last_print_at = now


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--"
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_bar(progress: float, *, width: int) -> str:
    progress = min(max(float(progress), 0.0), 1.0)
    filled = int(round(progress * width))
    return "#" * filled + "-" * (width - filled)


def _format_metric(value: float, precision: int) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{precision}f}"


def estimate_episode_requests(env: EdgeComputingEnv) -> int:
    original_time = env.current_time_minute
    expected = 0.0
    total_minutes = int(env.config.episode_hours * 60)
    for minute in range(total_minutes):
        env.current_time_minute = float(minute)
        expected += env._arrival_rate_per_minute()
    env.current_time_minute = original_time
    return max(int(round(expected)), 1)


def _rollout_active(env: EdgeComputingEnv, *, max_requests: int, rollout_unit: str) -> bool:
    if env.done:
        return False
    if rollout_unit == "episode":
        return True
    return env.metrics["requests"] < max_requests


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
    progress_label: str = "",
    progress_interval_seconds: float = 0.0,
    rollout_unit: str = "requests",
    overall_index: int | None = None,
    overall_total: int | None = None,
    overall_started_at: float | None = None,
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
    weights: list[float] = []
    greedy_latencies: list[float] = []
    greedy_weights: list[float] = []
    window_latencies: dict[int, list[tuple[float, float]]] = {}
    target_requests = max_requests if rollout_unit == "requests" else estimate_episode_requests(env)
    progress = RolloutProgress(
        label=progress_label,
        target_requests=target_requests,
        interval_seconds=progress_interval_seconds,
        episode_hours=env.config.episode_hours,
        rollout_unit=rollout_unit,
        deployment_interval_minutes=env.config.deployment_interval_minutes,
        overall_index=overall_index,
        overall_total=overall_total,
        overall_started_at=overall_started_at,
    )
    while _rollout_active(env, max_requests=max_requests, rollout_unit=rollout_unit):
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
        request_count = float(info.get("request_count", 1.0))
        train_reward = _training_reward(args, env_reward=reward, policy_info=info, greedy_info=greedy_info)
        if record:
            agent.observe_step_reward(
                train_reward * reward_scale,
                stage_count=len(request.stage_compute_gcycles),
                done=done,
                weight=request_count,
            )
        rewards.append(float(reward))
        train_rewards.append(float(train_reward))
        latencies.append(float(info["latency_s"]))
        weights.append(request_count)
        window_latencies.setdefault(deployment_window, []).append((float(info["latency_s"]), request_count))
        if greedy_info is not None:
            greedy_latencies.append(float(greedy_info["latency_s"]))
            greedy_weights.append(request_count)
        if progress.should_print():
            progress.maybe_print(
                env,
                avg_reward=_weighted_mean(rewards, weights),
                avg_train_reward=_weighted_mean(train_rewards, weights),
                avg_latency_s=_weighted_mean(latencies, weights),
            )
    progress.finish(
        env,
        avg_reward=_weighted_mean(rewards, weights),
        avg_train_reward=_weighted_mean(train_rewards, weights),
        avg_latency_s=_weighted_mean(latencies, weights),
    )
    if record and env.metrics["requests"] > 0:
        agent.flush_slow_window_reward(done=True)
    window_stats = _deployment_window_latency_stats(window_latencies)
    return {
        "requests": float(env.metrics["requests"]),
        "aggregate_events": float(env.metrics["aggregate_events"]),
        "simulated_hours": float(env.current_time_minute / 60.0),
        "episode_fraction": float(env.current_time_minute / max(env.config.episode_hours * 60.0, 1e-9)),
        "episode_complete": float(env.done),
        "avg_reward": _weighted_mean(rewards, weights),
        "avg_train_reward": _weighted_mean(train_rewards, weights),
        "avg_latency_s": _weighted_mean(latencies, weights),
        "p95_latency_s": _weighted_percentile(latencies, weights, 95.0),
        "avg_greedy_latency_s": _weighted_mean(greedy_latencies, greedy_weights) if greedy_latencies else float("nan"),
        "invalid_actions": float(env.metrics["invalid_actions"]),
        "deadline_violation_rate": float(env.metrics["deadline_violations"] / max(env.metrics["requests"], 1.0)),
        "deployment_updates": float(env.metrics["deployment_updates"]),
        **window_stats,
    }


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    values_np = np.asarray(values, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    return float(np.average(values_np, weights=weights_np))


def _weighted_percentile(values: list[float], weights: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values_np = np.asarray(values, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values_np)
    sorted_values = values_np[order]
    sorted_weights = weights_np[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = percentile / 100.0 * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, threshold, side="left")])


def _deployment_window_latency_stats(window_latencies: dict[int, list[tuple[float, float]]]) -> dict[str, float]:
    non_empty = [(window, values) for window, values in sorted(window_latencies.items()) if values]
    if not non_empty:
        return {
            "first_window_avg_latency_s": float("nan"),
            "last_window_avg_latency_s": float("nan"),
            "window_latency_delta_s": float("nan"),
        }
    first_values, first_weights = zip(*non_empty[0][1])
    last_values, last_weights = zip(*non_empty[-1][1])
    first_avg = _weighted_mean(list(first_values), list(first_weights))
    last_avg = _weighted_mean(list(last_values), list(last_weights))
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
    rollout_unit: str = "requests",
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
                progress_label=f"eval seed={seed_idx + 1:02d}/{args.eval_seeds:02d}",
                progress_interval_seconds=getattr(args, "progress_interval_seconds", 0.0),
                rollout_unit=rollout_unit,
            )
        )
    avg_latencies = np.array([r["avg_latency_s"] for r in runs], dtype=np.float64)
    p95_latencies = np.array([r["p95_latency_s"] for r in runs], dtype=np.float64)
    invalid_actions = np.array([r["invalid_actions"] for r in runs], dtype=np.float64)
    violation_rates = np.array([r["deadline_violation_rate"] for r in runs], dtype=np.float64)
    deployment_updates = np.array([r["deployment_updates"] for r in runs], dtype=np.float64)
    aggregate_events = np.array([r["aggregate_events"] for r in runs], dtype=np.float64)
    first_window_latencies = np.array([r["first_window_avg_latency_s"] for r in runs], dtype=np.float64)
    last_window_latencies = np.array([r["last_window_avg_latency_s"] for r in runs], dtype=np.float64)
    return {
        "eval_avg_latency_s": float(avg_latencies.mean()),
        "eval_avg_latency_std": float(avg_latencies.std()),
        "eval_p95_latency_s": float(p95_latencies.mean()),
        "eval_invalid_actions": float(invalid_actions.mean()),
        "eval_deadline_violation_rate": float(violation_rates.mean()),
        "eval_deployment_updates": float(deployment_updates.mean()),
        "eval_aggregate_events": float(aggregate_events.mean()),
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
    rollout_unit: str = "requests",
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
        det_target_requests = max_requests if rollout_unit == "requests" else estimate_episode_requests(env)
        det_progress = RolloutProgress(
            label=f"diag-det seed={seed_idx + 1:02d}/{args.eval_seeds:02d}",
            target_requests=det_target_requests,
            interval_seconds=getattr(args, "progress_interval_seconds", 0.0),
            episode_hours=env.config.episode_hours,
            rollout_unit=rollout_unit,
            deployment_interval_minutes=env.config.deployment_interval_minutes,
        )
        while _rollout_active(env, max_requests=max_requests, rollout_unit=rollout_unit):
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
            det_progress.maybe_print(env)
        det_progress.finish(env)

        stochastic_env = build_env(args, seed_offset=seed_base + seed_idx)
        stochastic_stats = rollout(
            stochastic_env,
            agent,
            max_requests,
            args=args,
            deterministic=False,
            record=False,
            train_mode=train_mode,
            progress_label=f"diag-sto seed={seed_idx + 1:02d}/{args.eval_seeds:02d}",
            progress_interval_seconds=getattr(args, "progress_interval_seconds", 0.0),
            rollout_unit=rollout_unit,
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
        env = build_env(args, seed_offset=40_000 + episode_idx, group_scenario_by_refresh=True)
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


def resolve_eval_rollout_unit(args: argparse.Namespace) -> str:
    if args.eval_rollout_unit == "same":
        return args.rollout_unit
    return args.eval_rollout_unit


def main() -> None:
    args = parse_args()
    eval_rollout_unit = resolve_eval_rollout_unit(args)
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
        fast_policy_kind=args.fast_policy_kind,
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
    print(f"  fast_policy_kind={args.fast_policy_kind}")
    print(f"  rollout_unit={args.rollout_unit}")
    print(f"  eval_rollout_unit={eval_rollout_unit}")
    print(f"  scenario_refresh_episodes={args.scenario_refresh_episodes}")
    print(f"  reward_mode={args.reward_mode}")
    print(f"  optimizer_reward_scale={args.reward_scale}")
    print(f"  max_replicas_per_stage={args.replicas_per_stage} actual_replica_count=learned")
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
            rollout_unit=eval_rollout_unit,
        )
        diagnostic_stats, previous_eval_actions = evaluate_policy_diagnostics(
            args,
            agent,
            seed_base=30_000,
            max_requests=args.eval_requests,
            train_mode=args.train_mode,
            previous_actions=None,
            rollout_unit=eval_rollout_unit,
        )
        initial_row = {
            "update": 0,
            "episode": 0,
            "requests": 0,
            "aggregate_events": 0,
            "simulated_hours": np.nan,
            "episode_fraction": np.nan,
            "episode_complete": 0,
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
            "eval_aggregate_events": eval_stats["eval_aggregate_events"],
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

    training_started_at = time.monotonic()
    for update in range(args.updates):
        env = build_env(args, seed_offset=update, group_scenario_by_refresh=True)
        stats = rollout(
            env,
            agent,
            args.requests_per_update,
            args=args,
            reward_scale=args.reward_scale,
            train_mode=args.train_mode,
            progress_label=f"update={update + 1:03d}/{args.updates:03d}",
            progress_interval_seconds=args.progress_interval_seconds,
            rollout_unit=args.rollout_unit,
            overall_index=update,
            overall_total=args.updates,
            overall_started_at=training_started_at,
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
                rollout_unit=eval_rollout_unit,
            )
            diagnostic_stats, previous_eval_actions = evaluate_policy_diagnostics(
                args,
                agent,
                seed_base=30_000,
                max_requests=args.eval_requests,
                train_mode=args.train_mode,
                previous_actions=previous_eval_actions,
                rollout_unit=eval_rollout_unit,
            )
        else:
            diagnostic_stats = {}
        log_row = {
            "update": update + 1,
            "episode": update + 1,
            "requests": int(stats["requests"]),
            "aggregate_events": int(stats["aggregate_events"]),
            "simulated_hours": stats["simulated_hours"],
            "episode_fraction": stats["episode_fraction"],
            "episode_complete": int(stats["episode_complete"]),
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
            "eval_aggregate_events": eval_stats.get("eval_aggregate_events", np.nan),
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
            "update={:03d} episode={:03d} complete={} requests={} aggregate_events={} sim_hours={:.2f} episode_frac={:.1%} "
            "avg_reward={:.4f} avg_latency={:.4f}s train_reward={:.4f} invalid={} deployments={} "
            "slow_loss={:.4f} fast_loss={:.4f}".format(
                update + 1,
                update + 1,
                int(stats["episode_complete"]),
                int(stats["requests"]),
                int(stats["aggregate_events"]),
                stats["simulated_hours"],
                stats["episode_fraction"],
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
        save_checkpoint(
            agent,
            save_dir / "latest.pt",
            {
                "update": update + 1,
                "avg_latency_s": stats["avg_latency_s"],
                "avg_reward": stats["avg_reward"],
                "best_latency_s": best_latency,
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
            rollout_unit=eval_rollout_unit,
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
