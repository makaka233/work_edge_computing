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
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    tqdm = None

from edge_drl.agents.hierarchical import FastGreedyScheduler, SlowGreedyDeploymentPolicy, build_baseline_agent
from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig


PRESSURE_PROFILE_DEFAULTS: dict[str, dict[str, float | str]] = {
    "baseline": {},
    # Keep topology and physical seed fixed while moving the operating point
    # into a moderate MEC stress regime.
    "mec-moderate": {
        "active_user_ratio": 0.20,
        "active_user_request_rate_per_minute": 1.75,
        "traffic_scale": 1.0,
        "load_multipliers": "0.8,1.1,1.4,1.7",
        "task_compute_scale": 1.5,
        "task_data_scale": 2.0,
        "node_compute_capacity_scale": 0.65,
        "wired_link_bandwidth_scale": 0.35,
        "service_resource_fraction": 0.45,
    },
    # Validate moderate pressure first; this profile is a bounded stress case
    # rather than a saturation-first experiment.
    "mec-stress": {
        "active_user_ratio": 0.30,
        "active_user_request_rate_per_minute": 2.0,
        "traffic_scale": 1.0,
        "load_multipliers": "0.8,1.1,1.4,1.7",
        "task_compute_scale": 2.5,
        "task_data_scale": 3.0,
        "node_compute_capacity_scale": 0.45,
        "wired_link_bandwidth_scale": 0.15,
        "service_resource_fraction": 0.35,
    },
}


def apply_pressure_profile(args: argparse.Namespace, argv: list[str] | None = None) -> argparse.Namespace:
    """Apply a reproducible demand/capacity profile without changing topology."""

    profile = PRESSURE_PROFILE_DEFAULTS[args.pressure_profile]
    if not profile:
        return args
    explicit_options = {
        token.split("=", 1)[0]
        for token in (sys.argv[1:] if argv is None else argv)
        if token.startswith("--")
    }
    for field, value in profile.items():
        option = "--" + field.replace("_", "-")
        if option not in explicit_options:
            setattr(args, field, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical dual-agent PPO for edge services.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--physical-seed",
        type=int,
        default=None,
        help="Seed for fixed physical edge infrastructure: nodes, capacities, service catalogue, and wired links.",
    )
    parser.add_argument("--fixed-scenario", action="store_true")
    parser.add_argument(
        "--scenario-refresh-episodes",
        type=int,
        default=1,
        help="Without --fixed-scenario, reuse one scenario instance for this many training episodes.",
    )
    parser.add_argument(
        "--demand-sampling-mode",
        choices=["episode", "rollout"],
        default="episode",
        help=(
            "episode keeps the demand scenario for --scenario-refresh-episodes training episodes; "
            "rollout samples a new demand scenario for every PPO rollout while keeping physical_seed fixed."
        ),
    )
    parser.add_argument("--num-users", type=int, default=10_000)
    parser.add_argument("--num-edge-nodes", type=int, default=32)
    parser.add_argument("--num-service-types", type=int, default=10)
    parser.add_argument(
        "--pressure-profile",
        choices=sorted(PRESSURE_PROFILE_DEFAULTS),
        default="baseline",
        help=(
            "Reproducible MEC demand/capacity preset. The profile changes demand and fixed per-run "
            "capacity scales only; explicit scale flags override it."
        ),
    )
    parser.add_argument(
        "--episode-hours",
        type=int,
        default=4,
        help="Environment episode horizon in hours. One episode contains one-second environment steps.",
    )
    parser.add_argument(
        "--deployment-interval-minutes",
        type=int,
        default=10,
        help="Slow deployment period. The default 10 minutes equals 600 environment steps.",
    )
    parser.add_argument(
        "--arrival-profile",
        choices=["stationary", "daily"],
        default="stationary",
        help="Use stationary demand for independent 4h episodes, or the legacy 24h morning/lunch/evening profile.",
    )
    parser.add_argument("--mean-requests-per-minute", type=float, default=None)
    parser.add_argument("--active-user-ratio", type=float, default=0.15)
    parser.add_argument("--active-user-request-rate-per-minute", type=float, default=1.5)
    parser.add_argument("--traffic-scale", type=float, default=1.0)
    parser.add_argument(
        "--load-multipliers",
        type=str,
        default="1.0",
        help="Comma-separated demand load multipliers cycled across rollout seeds, e.g. 1.0,1.4,1.8,2.2.",
    )
    parser.add_argument(
        "--rollout-start-mode",
        choices=["beginning", "cycle-window", "random-window"],
        default="beginning",
        help="Initial time for each training rollout. Nonzero starts only affect the legacy daily arrival profile.",
    )
    parser.add_argument(
        "--eval-rollout-start-mode",
        choices=["same", "beginning", "cycle-window", "random-window"],
        default="same",
        help="Initial time for eval rollouts. same reuses --rollout-start-mode.",
    )
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--task-data-scale", type=float, default=1.0)
    parser.add_argument(
        "--node-compute-capacity-scale",
        type=float,
        default=1.0,
        help="Scale fixed edge-node compute capacities for this run. Values below 1.0 create a heavier compute bottleneck.",
    )
    parser.add_argument(
        "--wired-link-bandwidth-scale",
        type=float,
        default=1.0,
        help="Scale fixed wired-link bandwidths for this run. Values below 1.0 create stronger link bottlenecks.",
    )
    parser.add_argument(
        "--service-resource-fraction",
        type=float,
        default=0.5,
        help="Fixed fraction of each node's memory/storage available to this service-placement controller.",
    )
    parser.add_argument(
        "--request-aggregation-window-seconds",
        type=float,
        default=1.0,
        help="Legacy name; fixed at 1 second so one env.step settles all individual requests arriving in that second.",
    )
    parser.add_argument(
        "--sampled-seconds-per-window",
        type=int,
        default=60,
        help=(
            "Training-only temporal sampling budget for each deployment window. "
            "Default 60 represents a 10-minute window with 60 weighted settlement steps; use 0 for a full rollout."
        ),
    )
    parser.add_argument("--load-ewma-tau-minutes", type=float, default=1.0)
    parser.add_argument("--wireless-uplink-mbps", type=float, default=150.0)
    parser.add_argument("--radio-rtt-ms", type=float, default=10.0)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--requests-per-update", type=int, default=4096)
    parser.add_argument(
        "--rollout-unit",
        choices=["requests", "window", "episode"],
        default="window",
        help=(
            "Collect each PPO rollout by request count, one slow-deployment window, or one full environment episode."
        ),
    )
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--reward-mode", choices=["latency"], default="latency")
    parser.add_argument("--train-mode", choices=["joint", "fast-only"], default="joint")
    parser.add_argument(
        "--replicas-per-stage",
        "--max-replicas-per-stage",
        dest="replicas_per_stage",
        type=int,
        default=0,
        help="Maximum replicas the slow count PPO may choose per service stage. Use 0 for no artificial cap, i.e. num_edge_nodes.",
    )
    parser.add_argument("--compute-hotspot-threshold", type=float, default=0.60)
    parser.add_argument("--link-hotspot-threshold", type=float, default=0.60)
    parser.add_argument("--resource-active-load-threshold", type=float, default=0.01)
    parser.add_argument("--compute-hotspot-coef", type=float, default=0.0)
    parser.add_argument("--link-hotspot-coef", type=float, default=0.0)
    parser.add_argument("--compute-imbalance-coef", type=float, default=0.0)
    parser.add_argument("--link-imbalance-coef", type=float, default=0.0)
    parser.add_argument("--idle-deployed-node-coef", type=float, default=0.0)
    parser.add_argument("--fast-policy-kind", choices=["node_scorer", "gat_node_scorer"], default="gat_node_scorer")
    parser.add_argument("--slow-lr", type=float, default=3e-4)
    parser.add_argument("--fast-lr", type=float, default=3e-4)
    parser.add_argument("--slow-k-epochs", type=int, default=3)
    parser.add_argument("--fast-k-epochs", type=int, default=4)
    parser.add_argument("--slow-entropy-coef", type=float, default=0.001)
    parser.add_argument("--slow-count-entropy-coef", type=float, default=None)
    parser.add_argument("--slow-placement-entropy-coef", type=float, default=None)
    parser.add_argument(
        "--slow-tail-latency-coef",
        type=float,
        default=0.35,
        help="Weight of the window P95 latency in the Slow deployment return; 0 uses mean latency only.",
    )
    parser.add_argument("--fast-entropy-coef", type=float, default=0.0)
    parser.add_argument("--slow-value-coef", type=float, default=0.5)
    parser.add_argument("--slow-critic-lr", type=float, default=3e-4)
    parser.add_argument("--slow-critic-k-epochs", type=int, default=4)
    parser.add_argument("--slow-deployment-memory-coef", type=float, default=0.03)
    parser.add_argument("--slow-deployment-storage-coef", type=float, default=0.01)
    parser.add_argument(
        "--slow-migration-coef",
        type=float,
        default=0.0,
        help="Slow-return migration penalty. Disabled by default; deployment changes remain logged.",
    )
    parser.add_argument("--fast-value-coef", type=float, default=0.5)
    parser.add_argument("--slow-target-kl", type=float, default=0.03)
    parser.add_argument("--fast-target-kl", type=float, default=0.03)
    parser.add_argument("--slow-minibatch-size", type=int, default=2048)
    parser.add_argument("--fast-minibatch-size", type=int, default=512)
    parser.add_argument(
        "--fast-windows-per-update",
        "--rollouts-per-update",
        dest="fast_windows_per_update",
        type=int,
        default=1,
        help=(
            "Collect this many windows before each Fast PPO update. "
            "--rollouts-per-update is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--slow-windows-per-update",
        type=int,
        default=12,
        help="Accumulate this many completed deployment windows before each Slow PPO update.",
    )
    parser.add_argument(
        "--synchronized-window-block",
        type=int,
        default=0,
        help="When positive, collect this many windows and update Fast and Slow PPO together each block.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-checkpoint", type=str, default="")
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--eval-baseline", action="store_true")
    parser.add_argument(
        "--eval-before-training",
        action="store_true",
        help="Run the eval sweep at update 0 before training starts. Disabled by default for long episode runs.",
    )
    parser.add_argument("--eval-requests", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument(
        "--eval-rollout-unit",
        choices=["requests", "window", "episode", "same"],
        default="requests",
        help="Use request-count eval, full-episode eval, or the same unit as training.",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        default=1,
        help="Number of held-out demand seeds per eval point. Default 1 performs one eval rollout.",
    )
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
    args = apply_pressure_profile(parser.parse_args())
    if args.synchronized_window_block > 0:
        args.fast_windows_per_update = args.synchronized_window_block
        args.slow_windows_per_update = args.synchronized_window_block
    if args.fast_windows_per_update < 1:
        parser.error("--fast-windows-per-update must be >= 1")
    if args.slow_windows_per_update < 1:
        parser.error("--slow-windows-per-update must be >= 1")
    if args.eval_seeds < 1:
        parser.error("--eval-seeds must be >= 1")
    if args.replicas_per_stage < 0:
        parser.error("--replicas-per-stage must be >= 0")
    if not 0.0 < args.service_resource_fraction <= 1.0:
        parser.error("--service-resource-fraction must be in (0, 1]")
    if args.slow_critic_k_epochs < 1:
        parser.error("--slow-critic-k-epochs must be >= 1")
    if not 0.0 <= args.slow_tail_latency_coef <= 1.0:
        parser.error("--slow-tail-latency-coef must be in [0, 1]")
    if args.synchronized_window_block < 0:
        parser.error("--synchronized-window-block must be >= 0")
    if args.episode_hours < 1:
        parser.error("--episode-hours must be >= 1")
    if args.deployment_interval_minutes < 1:
        parser.error("--deployment-interval-minutes must be >= 1")
    if not np.isclose(args.request_aggregation_window_seconds, 1.0):
        parser.error("--request-aggregation-window-seconds must remain exactly 1.0")
    if args.sampled_seconds_per_window < 0:
        parser.error("--sampled-seconds-per-window must be >= 0")
    if args.train_mode == "joint" and args.rollout_unit == "requests":
        parser.error("joint training requires --rollout-unit window or episode so each slow action receives a complete return")
    try:
        _parse_float_list(args.load_multipliers, "--load-multipliers")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def effective_replicas_per_stage(args: argparse.Namespace) -> int:
    if int(args.replicas_per_stage) == 0:
        return int(args.num_edge_nodes)
    return min(int(args.replicas_per_stage), int(args.num_edge_nodes))


def _parse_float_list(raw: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of positive numbers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain at least one positive number")
    return values


def load_multiplier_for_rollout(args: argparse.Namespace, rollout_idx: int) -> float:
    multipliers = _parse_float_list(getattr(args, "load_multipliers", "1.0"), "--load-multipliers")
    return multipliers[int(rollout_idx) % len(multipliers)]


def rollout_start_minute(args: argparse.Namespace, rollout_idx: int, *, eval_mode: bool = False) -> float:
    if getattr(args, "arrival_profile", "daily") == "stationary":
        return 0.0
    mode = getattr(args, "rollout_start_mode", "beginning")
    if eval_mode:
        eval_mode_value = getattr(args, "eval_rollout_start_mode", "same")
        mode = mode if eval_mode_value == "same" else eval_mode_value
    if mode == "beginning":
        return 0.0
    interval = int(getattr(args, "deployment_interval_minutes", 10))
    total_minutes = int(getattr(args, "episode_hours", 24) * 60)
    starts = list(range(0, max(total_minutes - interval + 1, 1), interval))
    if not starts:
        return 0.0
    if mode == "cycle-window":
        return float(starts[int(rollout_idx) % len(starts)])
    rng = np.random.default_rng(int(getattr(args, "seed", 2026)) + 100_000 + int(rollout_idx))
    return float(starts[int(rng.integers(0, len(starts)))])


def start_env_at_minute(env: EdgeComputingEnv, start_minute: float) -> None:
    if start_minute <= 0.0:
        return
    env.current_time_minute = float(start_minute)
    env.next_deployment_update_minute = float(start_minute)
    env.last_load_update_minute = float(start_minute)
    env.current_requests = env._generate_current_second_requests()
    env.current_request = env.current_requests[0] if env.current_requests else None


def scenario_seed_for_offset(args: argparse.Namespace, seed_offset: int = 0, *, group_by_refresh: bool = False) -> int:
    if getattr(args, "fixed_scenario", False):
        return int(args.seed)
    refresh = max(int(getattr(args, "scenario_refresh_episodes", 1)), 1)
    scenario_offset = seed_offset // refresh if group_by_refresh else seed_offset
    return int(args.seed + scenario_offset)


def demand_seed_for_training_rollout(args: argparse.Namespace, rollout_idx: int, episode_idx: int) -> int:
    if getattr(args, "demand_sampling_mode", "episode") == "rollout":
        return scenario_seed_for_offset(args, rollout_idx, group_by_refresh=False)
    return scenario_seed_for_offset(args, episode_idx, group_by_refresh=True)


def build_env(args: argparse.Namespace, seed_offset: int = 0, *, group_scenario_by_refresh: bool = False) -> EdgeComputingEnv:
    return EdgeComputingEnv(
        EdgeEnvConfig(
            seed=args.seed + seed_offset,
            physical_seed=args.seed if args.physical_seed is None else args.physical_seed,
            scenario_seed=scenario_seed_for_offset(args, seed_offset, group_by_refresh=group_scenario_by_refresh),
            num_users=args.num_users,
            num_edge_nodes=args.num_edge_nodes,
            num_service_types=args.num_service_types,
            episode_hours=args.episode_hours,
            deployment_interval_minutes=args.deployment_interval_minutes,
            arrival_profile=args.arrival_profile,
            mean_requests_per_minute=args.mean_requests_per_minute,
            active_user_ratio=args.active_user_ratio,
            active_user_request_rate_per_minute=args.active_user_request_rate_per_minute,
            traffic_scale=args.traffic_scale,
            demand_load_multiplier=load_multiplier_for_rollout(args, seed_offset),
            task_compute_scale=args.task_compute_scale,
            task_data_scale=args.task_data_scale,
            node_compute_capacity_scale=args.node_compute_capacity_scale,
            wired_link_bandwidth_scale=args.wired_link_bandwidth_scale,
            service_resource_fraction=args.service_resource_fraction,
            request_aggregation_window_seconds=args.request_aggregation_window_seconds,
            load_ewma_tau_minutes=args.load_ewma_tau_minutes,
            wireless_uplink_mbps=args.wireless_uplink_mbps,
            radio_rtt_ms=args.radio_rtt_ms,
        )
    )


def build_training_env(args: argparse.Namespace, *, rollout_idx: int, episode_idx: int) -> EdgeComputingEnv:
    demand_seed = demand_seed_for_training_rollout(args, rollout_idx, episode_idx)
    return EdgeComputingEnv(
        EdgeEnvConfig(
            seed=args.seed + rollout_idx,
            physical_seed=args.seed if args.physical_seed is None else args.physical_seed,
            scenario_seed=demand_seed,
            num_users=args.num_users,
            num_edge_nodes=args.num_edge_nodes,
            num_service_types=args.num_service_types,
            episode_hours=args.episode_hours,
            deployment_interval_minutes=args.deployment_interval_minutes,
            arrival_profile=args.arrival_profile,
            mean_requests_per_minute=args.mean_requests_per_minute,
            active_user_ratio=args.active_user_ratio,
            active_user_request_rate_per_minute=args.active_user_request_rate_per_minute,
            traffic_scale=args.traffic_scale,
            demand_load_multiplier=load_multiplier_for_rollout(args, rollout_idx),
            task_compute_scale=args.task_compute_scale,
            task_data_scale=args.task_data_scale,
            node_compute_capacity_scale=args.node_compute_capacity_scale,
            wired_link_bandwidth_scale=args.wired_link_bandwidth_scale,
            service_resource_fraction=args.service_resource_fraction,
            request_aggregation_window_seconds=args.request_aggregation_window_seconds,
            load_ewma_tau_minutes=args.load_ewma_tau_minutes,
            wireless_uplink_mbps=args.wireless_uplink_mbps,
            radio_rtt_ms=args.radio_rtt_ms,
        )
    )


def traffic_rate_summary(env: EdgeComputingEnv) -> dict[str, float]:
    rates = []
    original_time = env.current_time_minute
    episode_minutes = max(int(env.config.episode_hours * 60), 1)
    for minute in range(episode_minutes):
        env.current_time_minute = float(minute)
        rates.append(env._arrival_rate_per_minute())
    env.current_time_minute = original_time
    values = np.asarray(rates, dtype=np.float64)
    return {
        "avg_requests_per_second": float(values.mean() / 60.0),
        "min_requests_per_second": float(values.min() / 60.0),
        "max_requests_per_second": float(values.max() / 60.0),
        "expected_requests_per_episode": float(values.sum()),
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
        rollout_start_minute: float = 0.0,
        start_requests: float = 0.0,
        start_request_events: float = 0.0,
    ):
        self.label = label
        self.target_requests = target_requests
        self.interval_seconds = max(interval_seconds, 0.0) if tqdm is not None else 0.0
        self.episode_hours = episode_hours
        self.rollout_unit = rollout_unit
        self.deployment_interval_minutes = deployment_interval_minutes
        self.rollout_start_minute = rollout_start_minute
        self.start_requests = start_requests
        self.start_request_events = start_request_events
        self.started_at = time.monotonic()
        self.last_print_at = self.started_at
        self.printed = False
        self.progress_units = 0.0
        self.tqdm_bar = None
        if self.interval_seconds > 0:
            total_units = self._total_progress_units()
            self.tqdm_bar = tqdm(
                total=total_units,
                desc=self.label,
                unit="h" if self.rollout_unit in {"episode", "window"} else "req",
                dynamic_ncols=True,
                mininterval=self.interval_seconds,
                leave=True,
                bar_format="{desc}: {percentage:5.1f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                file=sys.stdout,
            )

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
        avg_valid_latency_s: float | None = None,
        avg_penalty_latency_s: float | None = None,
    ) -> None:
        if self.should_print():
            self._print_tqdm(
                env,
                now=time.monotonic(),
                avg_reward=avg_reward,
                avg_train_reward=avg_train_reward,
                avg_latency_s=avg_latency_s,
                avg_valid_latency_s=avg_valid_latency_s,
                avg_penalty_latency_s=avg_penalty_latency_s,
            )

    def finish(
        self,
        env: EdgeComputingEnv,
        *,
        avg_reward: float = float("nan"),
        avg_train_reward: float = float("nan"),
        avg_latency_s: float | None = None,
        avg_valid_latency_s: float | None = None,
        avg_penalty_latency_s: float | None = None,
    ) -> None:
        if self.interval_seconds <= 0 or self.tqdm_bar is None:
            return
        self._print_tqdm(
            env,
            now=time.monotonic(),
            final=True,
            avg_reward=avg_reward,
            avg_train_reward=avg_train_reward,
            avg_latency_s=avg_latency_s,
            avg_valid_latency_s=avg_valid_latency_s,
            avg_penalty_latency_s=avg_penalty_latency_s,
        )

    def _print_tqdm(
        self,
        env: EdgeComputingEnv,
        *,
        now: float,
        final: bool = False,
        avg_reward: float = float("nan"),
        avg_train_reward: float = float("nan"),
        avg_latency_s: float | None = None,
        avg_valid_latency_s: float | None = None,
        avg_penalty_latency_s: float | None = None,
    ) -> None:
        assert self.tqdm_bar is not None
        requests = max(float(env.metrics.get("requests", 0.0)) - self.start_requests, 0.0)
        request_events = int(max(float(env.metrics.get("request_events", 0.0)) - self.start_request_events, 0.0))
        sim_hours = env.current_time_minute / 60.0
        episode_fraction = sim_hours / max(float(self.episode_hours), 1e-9)
        if self.rollout_unit == "episode":
            progress = min(episode_fraction, 1.0)
        elif self.rollout_unit == "window":
            elapsed_window_minutes = max(env.current_time_minute - self.rollout_start_minute, 0.0)
            progress = min(elapsed_window_minutes / max(float(self.deployment_interval_minutes), 1e-9), 1.0)
        else:
            progress = min(requests / max(float(self.target_requests or 1), 1.0), 1.0)
        if avg_latency_s is None:
            avg_latency_s = env.metrics["total_latency_s"] / max(requests, 1.0)
        if avg_valid_latency_s is None:
            avg_valid_latency_s = env.metrics["total_valid_latency_s"] / max(env.metrics["valid_requests"], 1.0)
        if avg_penalty_latency_s is None:
            avg_penalty_latency_s = env.metrics["total_penalty_latency_s"] / max(requests, 1.0)
        elapsed = max(now - self.started_at, 1e-9)
        total_windows = max(int(np.ceil(self.episode_hours * 60.0 / max(self.deployment_interval_minutes, 1))), 1)
        current_window = min(int(env.current_time_minute // max(self.deployment_interval_minutes, 1)) + 1, total_windows)
        wall_event_rate = request_events / elapsed
        units = self._progress_units(env)
        delta = max(units - self.progress_units, 0.0)
        if delta > 0:
            self.tqdm_bar.update(delta)
            self.progress_units = units
        request_target = self.target_requests or requests or 1
        postfix = (
            f"R={_format_metric(avg_reward, 4)} "
            f"trainR={_format_metric(avg_train_reward, 4)} "
            f"Lat={avg_latency_s * 1000:.1f}ms "
            f"VLat={avg_valid_latency_s * 1000:.1f}ms "
            f"Pen={avg_penalty_latency_s * 1000:.1f}ms "
            f"win={current_window}/{total_windows} "
            f"t={sim_hours:.1f}/{self.episode_hours}h "
            f"req={_format_count(requests)}/{_format_count(request_target)} "
            f"speed={wall_event_rate:.1f}ev/s"
        )
        self.tqdm_bar.set_postfix_str(postfix, refresh=True)
        self.printed = True
        self.last_print_at = now
        if final:
            self.tqdm_bar.close()

    def _total_progress_units(self) -> float:
        if self.rollout_unit == "episode":
            return max(float(self.episode_hours), 1.0)
        if self.rollout_unit == "window":
            return max(float(self.deployment_interval_minutes) / 60.0, 1e-9)
        return max(float(self.target_requests or 1), 1.0)

    def _progress_units(self, env: EdgeComputingEnv) -> float:
        if self.rollout_unit == "episode":
            return min(max(env.current_time_minute / 60.0, 0.0), self._total_progress_units())
        if self.rollout_unit == "window":
            elapsed_window_hours = max(env.current_time_minute - self.rollout_start_minute, 0.0) / 60.0
            return min(elapsed_window_hours, self._total_progress_units())
        return min(float(env.metrics.get("requests", 0.0)), self._total_progress_units())


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--"
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_metric(value: float, precision: int) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{precision}f}"


def _format_count(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(round(value)))


def estimate_episode_requests(env: EdgeComputingEnv) -> int:
    original_time = env.current_time_minute
    expected = 0.0
    total_minutes = int(env.config.episode_hours * 60)
    for minute in range(total_minutes):
        env.current_time_minute = float(minute)
        expected += env._arrival_rate_per_minute()
    env.current_time_minute = original_time
    return max(int(round(expected)), 1)


def _rollout_active(
    env: EdgeComputingEnv,
    *,
    max_requests: int,
    rollout_unit: str,
    stop_time_minute: float | None = None,
) -> bool:
    if env.done:
        return False
    if stop_time_minute is not None and env.current_time_minute >= stop_time_minute:
        return False
    if rollout_unit == "episode":
        return True
    if rollout_unit == "window":
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
    reset_env: bool = True,
) -> dict[str, float]:
    if args is None:
        args = argparse.Namespace(
            reward_mode="latency",
            compute_hotspot_threshold=0.60,
            link_hotspot_threshold=0.60,
            resource_active_load_threshold=0.01,
            compute_hotspot_coef=0.0,
            link_hotspot_coef=0.0,
            compute_imbalance_coef=0.0,
            link_imbalance_coef=0.0,
            idle_deployed_node_coef=0.0,
        )
    if reset_env:
        env.reset()
    if record:
        agent.last_slow_window_metrics = {}
    if frozen_slow_policy is None:
        frozen_slow_policy = SlowGreedyDeploymentPolicy()
    rewards: list[float] = []
    train_rewards: list[float] = []
    train_latency_costs: list[float] = []
    train_resource_penalties: list[float] = []
    diagnostic_resource_penalties: list[float] = []
    compute_hotspot_penalties: list[float] = []
    link_hotspot_penalties: list[float] = []
    compute_imbalance_penalties: list[float] = []
    link_imbalance_penalties: list[float] = []
    idle_deployed_node_penalties: list[float] = []
    latencies: list[float] = []
    valid_latencies: list[float] = []
    valid_weights: list[float] = []
    penalty_latencies: list[float] = []
    weights: list[float] = []
    window_latencies: dict[int, list[tuple[float, float]]] = {}
    scheduled_replica_counts: dict[tuple[int, int, int], float] = {}
    cross_node_stage_transitions = 0.0
    total_stage_transitions = 0.0
    start_metrics = dict(env.metrics)
    rollout_start_minute = env.current_time_minute
    stop_time_minute = None
    if rollout_unit == "window":
        stop_time_minute = min(
            rollout_start_minute + float(env.config.deployment_interval_minutes),
            float(env.config.episode_hours * 60),
        )
    represented_seconds = 1.0
    if record and rollout_unit == "window":
        sampled_seconds = int(getattr(args, "sampled_seconds_per_window", 0))
        window_seconds = float(env.config.deployment_interval_minutes * 60)
        if 0 < sampled_seconds < window_seconds:
            represented_seconds = window_seconds / float(sampled_seconds)
    if rollout_unit == "requests":
        target_requests = max_requests
    elif rollout_unit == "window":
        target_requests = max(
            int(round(estimate_episode_requests(env) * env.config.deployment_interval_minutes / max(env.config.episode_hours * 60.0, 1.0))),
            1,
        )
    else:
        target_requests = estimate_episode_requests(env)
    progress = RolloutProgress(
        label=progress_label,
        target_requests=target_requests,
        interval_seconds=progress_interval_seconds,
        episode_hours=env.config.episode_hours,
        rollout_unit=rollout_unit,
        deployment_interval_minutes=env.config.deployment_interval_minutes,
        rollout_start_minute=rollout_start_minute,
        start_requests=float(start_metrics.get("requests", 0.0)),
        start_request_events=float(start_metrics.get("request_events", 0.0)),
    )
    while _rollout_active(env, max_requests=max_requests, rollout_unit=rollout_unit, stop_time_minute=stop_time_minute):
        step_represented_seconds = represented_seconds
        if stop_time_minute is not None:
            remaining_seconds = max((stop_time_minute - env.current_time_minute) * 60.0, 0.0)
            if remaining_seconds < 1.0 - 1e-9:
                env.current_time_minute = stop_time_minute
                break
            step_represented_seconds = min(step_represented_seconds, remaining_seconds)
        requests = list(env.current_requests)
        if train_mode == "fast-only":
            if env.needs_deployment_update:
                env.apply_deployment(frozen_slow_policy.act(env))
            actions = agent.fast_agent.schedule_batch(
                env,
                requests,
                deterministic=deterministic,
                record=record,
            )
        else:
            actions = agent.act_batch(env, deterministic=deterministic, record=record)
        deployment_window = int(env.metrics["deployment_updates"])
        _, _, done, batch_info = env.step(actions, represented_seconds=step_represented_seconds)
        group_infos = batch_info["group_infos"]
        for group_idx, (request, info) in enumerate(zip(requests, group_infos)):
            request_count = float(info.get("request_count", request.request_count))
            train_reward_info = _training_reward_components(info, env, args)
            train_reward = train_reward_info["train_reward"]
            if record:
                agent.observe_step_reward(
                    train_reward * reward_scale,
                    stage_count=len(request.stage_compute_gcycles),
                    done=bool(done and group_idx == len(group_infos) - 1),
                    weight=request_count,
                    slow_reward=-float(info["latency_s"]) * reward_scale,
                    latency_s=float(info["latency_s"]),
                    penalty_latency_s=float(info["penalty_latency_s"]),
                    deadline_s=float(request.deadline_s),
                    invalid=not bool(info["valid"]),
                    max_node_compute_load=float(np.max(env.node_compute_load)),
                    max_link_load=float(np.max(env.link_load[np.isfinite(env.link_load)]))
                    if np.any(np.isfinite(env.link_load))
                    else 0.0,
                )
            rewards.append(float(info["reward"]))
            train_rewards.append(float(train_reward))
            train_latency_costs.append(float(train_reward_info["train_latency_cost_s"]))
            train_resource_penalties.append(float(train_reward_info["train_resource_penalty"]))
            diagnostic_resource_penalties.append(float(train_reward_info["diagnostic_resource_penalty"]))
            compute_hotspot_penalties.append(float(train_reward_info["compute_hotspot_penalty"]))
            link_hotspot_penalties.append(float(train_reward_info["link_hotspot_penalty"]))
            compute_imbalance_penalties.append(float(train_reward_info["compute_imbalance_penalty"]))
            link_imbalance_penalties.append(float(train_reward_info["link_imbalance_penalty"]))
            idle_deployed_node_penalties.append(float(train_reward_info["idle_deployed_node_penalty"]))
            latencies.append(float(info["latency_s"]))
            penalty_latencies.append(float(info["penalty_latency_s"]))
            if info["valid"]:
                valid_latencies.append(float(info["physical_latency_s"]))
                valid_weights.append(request_count)
                _accumulate_schedule_usage(
                    scheduled_replica_counts,
                    request.service_id,
                    info["stage_nodes"],
                    request_count,
                )
                for previous_node, next_node in zip(info["stage_nodes"], info["stage_nodes"][1:]):
                    total_stage_transitions += request_count
                    if previous_node != next_node:
                        cross_node_stage_transitions += request_count
            weights.append(request_count)
            window_latencies.setdefault(deployment_window, []).append((float(info["latency_s"]), request_count))
        if record and done and not group_infos:
            agent.flush_slow_window_reward(done=True)
        if progress.should_print():
            progress.maybe_print(
                env,
                avg_reward=_weighted_mean(rewards, weights),
                avg_train_reward=_weighted_mean(train_rewards, weights),
                avg_latency_s=_weighted_mean(latencies, weights),
                avg_valid_latency_s=_weighted_mean(valid_latencies, valid_weights),
                avg_penalty_latency_s=_weighted_mean(penalty_latencies, weights),
            )
    progress.finish(
        env,
        avg_reward=_weighted_mean(rewards, weights),
        avg_train_reward=_weighted_mean(train_rewards, weights),
        avg_latency_s=_weighted_mean(latencies, weights),
        avg_valid_latency_s=_weighted_mean(valid_latencies, valid_weights),
        avg_penalty_latency_s=_weighted_mean(penalty_latencies, weights),
    )
    if record and (env.done or env.needs_deployment_update):
        agent.flush_slow_window_reward(done=env.done)
    slow_window_metrics = {
        "slow_window_return": float("nan"),
        "slow_window_latency_return": float("nan"),
        "slow_window_avg_latency": float("nan"),
        "slow_window_p95_latency": float("nan"),
        "slow_tail_latency_cost": float("nan"),
        "slow_deployment_memory_cost": float("nan"),
        "slow_deployment_storage_cost": float("nan"),
        "slow_migration_cost": float("nan"),
        "slow_deployment_memory_fraction": float("nan"),
        "slow_deployment_storage_fraction": float("nan"),
        "slow_migration_fraction": float("nan"),
        **(agent.last_slow_window_metrics if record else {}),
    }
    window_stats = _deployment_window_latency_stats(window_latencies)
    replica_stats = _deployment_replica_stats(env)
    resource_stats = _resource_usage_stats(env, args)
    schedule_usage_stats = _schedule_usage_stats(
        env,
        scheduled_replica_counts,
        cross_node_stage_transitions=cross_node_stage_transitions,
        total_stage_transitions=total_stage_transitions,
    )
    rollout_requests = float(env.metrics["requests"] - start_metrics.get("requests", 0.0))
    rollout_request_events = float(env.metrics["request_events"] - start_metrics.get("request_events", 0.0))
    rollout_invalid_actions = float(env.metrics["invalid_actions"] - start_metrics.get("invalid_actions", 0.0))
    rollout_valid_requests = float(env.metrics["valid_requests"] - start_metrics.get("valid_requests", 0.0))
    rollout_deadline_violations = float(env.metrics["deadline_violations"] - start_metrics.get("deadline_violations", 0.0))
    rollout_deployment_updates = float(env.metrics["deployment_updates"] - start_metrics.get("deployment_updates", 0.0))
    rollout_settlement_steps = float(env.metrics["settlement_steps"] - start_metrics.get("settlement_steps", 0.0))
    rollout_logical_steps = float(env.metrics["time_steps"] - start_metrics.get("time_steps", 0.0))
    rollout_duration_minutes = max(env.current_time_minute - rollout_start_minute, 0.0)
    return {
        "requests": rollout_requests,
        "request_events": rollout_request_events,
        "settlement_steps": rollout_settlement_steps,
        "logical_steps": rollout_logical_steps,
        "temporal_sampling_fraction": float(rollout_settlement_steps / max(rollout_logical_steps, 1.0)),
        "simulated_hours": float(rollout_duration_minutes / 60.0),
        "episode_fraction": float(env.current_time_minute / max(env.config.episode_hours * 60.0, 1e-9)),
        "episode_complete": float(env.done),
        "avg_reward": _weighted_mean(rewards, weights),
        "avg_train_reward": _weighted_mean(train_rewards, weights),
        "avg_train_latency_cost_s": _weighted_mean(train_latency_costs, weights),
        "avg_train_resource_penalty": _weighted_mean(train_resource_penalties, weights),
        "avg_diagnostic_resource_penalty": _weighted_mean(diagnostic_resource_penalties, weights),
        "avg_compute_hotspot_penalty": _weighted_mean(compute_hotspot_penalties, weights),
        "avg_link_hotspot_penalty": _weighted_mean(link_hotspot_penalties, weights),
        "avg_compute_imbalance_penalty": _weighted_mean(compute_imbalance_penalties, weights),
        "avg_link_imbalance_penalty": _weighted_mean(link_imbalance_penalties, weights),
        "avg_idle_deployed_node_penalty": _weighted_mean(idle_deployed_node_penalties, weights),
        "avg_latency_s": _weighted_mean(latencies, weights),
        "p95_latency_s": _weighted_percentile(latencies, weights, 95.0),
        "avg_valid_latency_s": _weighted_mean(valid_latencies, valid_weights),
        "p95_valid_latency_s": _weighted_percentile(valid_latencies, valid_weights, 95.0),
        "valid_requests": rollout_valid_requests,
        "avg_penalty_latency_s": _weighted_mean(penalty_latencies, weights),
        "penalty_latency_share": _weighted_mean(penalty_latencies, weights) / max(_weighted_mean(latencies, weights), 1e-9),
        "invalid_actions": rollout_invalid_actions,
        "invalid_action_rate": float(rollout_invalid_actions / max(rollout_requests, 1.0)),
        "deadline_violation_rate": float(rollout_deadline_violations / max(rollout_requests, 1.0)),
        "deployment_updates": rollout_deployment_updates,
        **slow_window_metrics,
        **window_stats,
        **replica_stats,
        **resource_stats,
        **schedule_usage_stats,
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


def _deployment_replica_stats(env: EdgeComputingEnv) -> dict[str, float]:
    if env.deployment is None or env.scenario is None:
        return {
            "avg_replicas_per_stage": float("nan"),
            "min_replicas_per_stage": float("nan"),
            "max_replicas_per_stage": float("nan"),
            "single_replica_stage_rate": float("nan"),
            "total_deployed_replicas": float("nan"),
        }
    counts = []
    for service in env.scenario.services:
        for stage in service.stages:
            counts.append(float(env.deployment[service.service_id, stage.stage_id].sum()))
    if not counts:
        return {
            "avg_replicas_per_stage": float("nan"),
            "min_replicas_per_stage": float("nan"),
            "max_replicas_per_stage": float("nan"),
            "single_replica_stage_rate": float("nan"),
            "total_deployed_replicas": float("nan"),
        }
    counts_np = np.asarray(counts, dtype=np.float64)
    return {
        "avg_replicas_per_stage": float(counts_np.mean()),
        "min_replicas_per_stage": float(counts_np.min()),
        "max_replicas_per_stage": float(counts_np.max()),
        "single_replica_stage_rate": float(np.mean(counts_np <= 1.0)),
        "total_deployed_replicas": float(counts_np.sum()),
    }


def _accumulate_schedule_usage(
    usage: dict[tuple[int, int, int], float],
    service_id: int,
    stage_nodes: list[int] | tuple[int, ...],
    request_count: float,
) -> None:
    for stage_id, node_id in enumerate(stage_nodes):
        key = (int(service_id), int(stage_id), int(node_id))
        usage[key] = usage.get(key, 0.0) + float(request_count)


def _schedule_usage_stats(
    env: EdgeComputingEnv,
    usage: dict[tuple[int, int, int], float],
    *,
    cross_node_stage_transitions: float,
    total_stage_transitions: float,
) -> dict[str, float]:
    if env.scenario is None or env.deployment is None:
        return {
            "scheduled_stage_invocations": float("nan"),
            "used_replica_rate": float("nan"),
            "idle_replica_rate": float("nan"),
            "used_replicas_per_stage": float("nan"),
            "avg_replica_use_entropy": float("nan"),
            "avg_replica_use_top1_share": float("nan"),
            "cross_node_stage_transition_rate": float("nan"),
        }

    deployed_total = 0
    used_total = 0
    used_per_stage = []
    entropies = []
    top1_shares = []
    scheduled_stage_invocations = float(sum(usage.values()))

    for service in env.scenario.services:
        for stage in service.stages:
            deployed_nodes = np.flatnonzero(env.deployment[service.service_id, stage.stage_id])
            deployed_total += int(len(deployed_nodes))
            counts = np.asarray(
                [usage.get((service.service_id, stage.stage_id, int(node_id)), 0.0) for node_id in deployed_nodes],
                dtype=np.float64,
            )
            used = int(np.count_nonzero(counts > 0.0))
            used_total += used
            used_per_stage.append(float(used))
            total = float(counts.sum())
            if total > 0.0 and counts.size > 1:
                probabilities = counts[counts > 0.0] / total
                entropy = -float(np.sum(probabilities * np.log(probabilities)))
                entropies.append(entropy / max(np.log(float(counts.size)), 1e-9))
                top1_shares.append(float(counts.max() / total))
            elif total > 0.0:
                entropies.append(0.0)
                top1_shares.append(1.0)

    used_replica_rate = used_total / max(float(deployed_total), 1.0)
    return {
        "scheduled_stage_invocations": scheduled_stage_invocations,
        "used_replica_rate": float(used_replica_rate),
        "idle_replica_rate": float(1.0 - used_replica_rate),
        "used_replicas_per_stage": float(np.mean(used_per_stage)) if used_per_stage else float("nan"),
        "avg_replica_use_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "avg_replica_use_top1_share": float(np.mean(top1_shares)) if top1_shares else 0.0,
        "cross_node_stage_transition_rate": float(cross_node_stage_transitions / max(total_stage_transitions, 1.0)),
    }


def _resource_usage_stats(env: EdgeComputingEnv, args: argparse.Namespace | None = None) -> dict[str, float]:
    if env.scenario is None or env.deployment is None:
        return {
            "avg_node_compute_load": float("nan"),
            "max_node_compute_load": float("nan"),
            "p95_node_compute_load": float("nan"),
            "std_node_compute_load": float("nan"),
            "active_node_rate": float("nan"),
            "hot_node_rate": float("nan"),
            "avg_link_load": float("nan"),
            "max_link_load": float("nan"),
            "p95_link_load": float("nan"),
            "std_link_load": float("nan"),
            "active_link_rate": float("nan"),
            "hot_link_rate": float("nan"),
            "avg_node_memory_util": float("nan"),
            "max_node_memory_util": float("nan"),
            "avg_node_storage_util": float("nan"),
            "max_node_storage_util": float("nan"),
            "avg_service_memory_util": float("nan"),
            "max_service_memory_util": float("nan"),
            "avg_service_storage_util": float("nan"),
            "max_service_storage_util": float("nan"),
            "deployed_node_rate": float("nan"),
            "idle_deployed_node_rate": float("nan"),
        }

    compute_load = np.asarray(env.node_compute_load, dtype=np.float64)
    finite_links = np.isfinite(env.scenario.bandwidth_mb_s) & env.scenario.adjacency
    np.fill_diagonal(finite_links, False)
    link_load = np.asarray(env.link_load[finite_links], dtype=np.float64)

    memory_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
    storage_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
    for service in env.scenario.services:
        for stage in service.stages:
            placed = env.deployment[service.service_id, stage.stage_id]
            memory_used += placed * stage.memory_gb
            storage_used += placed * stage.storage_gb

    memory_capacity = np.asarray([node.memory_gb for node in env.scenario.nodes], dtype=np.float64)
    storage_capacity = np.asarray([node.storage_gb for node in env.scenario.nodes], dtype=np.float64)
    memory_util = memory_used / np.maximum(memory_capacity, 1e-9)
    storage_util = storage_used / np.maximum(storage_capacity, 1e-9)
    service_memory_util = memory_used / np.maximum(env.service_memory_capacities(), 1e-9)
    service_storage_util = storage_used / np.maximum(env.service_storage_capacities(), 1e-9)
    deployed_nodes = np.logical_or(memory_used > 0.0, storage_used > 0.0)
    active_threshold = float(getattr(args, "resource_active_load_threshold", 0.01))
    hot_threshold = float(getattr(args, "compute_hotspot_threshold", 0.60))
    link_hot_threshold = float(getattr(args, "link_hotspot_threshold", hot_threshold))

    if link_load.size == 0:
        link_load = np.asarray([float("nan")], dtype=np.float64)
    finite_link_load = link_load[np.isfinite(link_load)]
    if finite_link_load.size == 0:
        finite_link_load = np.asarray([float("nan")], dtype=np.float64)

    return {
        "avg_node_compute_load": float(np.mean(compute_load)),
        "max_node_compute_load": float(np.max(compute_load)),
        "p95_node_compute_load": float(np.percentile(compute_load, 95)),
        "std_node_compute_load": float(np.std(compute_load)),
        "active_node_rate": float(np.mean(compute_load > active_threshold)),
        "hot_node_rate": float(np.mean(compute_load > hot_threshold)),
        "avg_link_load": float(np.nanmean(link_load)),
        "max_link_load": float(np.nanmax(link_load)),
        "p95_link_load": float(np.nanpercentile(link_load, 95)),
        "std_link_load": float(np.nanstd(link_load)),
        "active_link_rate": float(np.mean(finite_link_load > active_threshold)),
        "hot_link_rate": float(np.mean(finite_link_load > link_hot_threshold)),
        "avg_node_memory_util": float(np.mean(memory_util)),
        "max_node_memory_util": float(np.max(memory_util)),
        "avg_node_storage_util": float(np.mean(storage_util)),
        "max_node_storage_util": float(np.max(storage_util)),
        "avg_service_memory_util": float(np.mean(service_memory_util)),
        "max_service_memory_util": float(np.max(service_memory_util)),
        "avg_service_storage_util": float(np.mean(service_storage_util)),
        "max_service_storage_util": float(np.max(service_storage_util)),
        "deployed_node_rate": float(np.mean(deployed_nodes)),
        "idle_deployed_node_rate": float(
            np.mean(compute_load[deployed_nodes] <= active_threshold) if deployed_nodes.any() else 0.0
        ),
    }


def aggregate_rollout_stats(rollouts: list[dict[str, float]]) -> dict[str, float]:
    if not rollouts:
        raise ValueError("cannot aggregate an empty rollout batch")
    if len(rollouts) == 1:
        return dict(rollouts[0])

    requests = np.asarray([r["requests"] for r in rollouts], dtype=np.float64)
    valid_requests = np.asarray([r["valid_requests"] for r in rollouts], dtype=np.float64)
    request_events = np.asarray([r["request_events"] for r in rollouts], dtype=np.float64)
    deployment_updates = np.asarray([r["deployment_updates"] for r in rollouts], dtype=np.float64)

    request_weighted = [
        "avg_reward",
        "avg_train_reward",
        "avg_train_latency_cost_s",
        "avg_train_resource_penalty",
        "avg_diagnostic_resource_penalty",
        "avg_compute_hotspot_penalty",
        "avg_link_hotspot_penalty",
        "avg_compute_imbalance_penalty",
        "avg_link_imbalance_penalty",
        "avg_idle_deployed_node_penalty",
        "avg_latency_s",
        "p95_latency_s",
        "avg_penalty_latency_s",
        "penalty_latency_share",
        "invalid_action_rate",
        "deadline_violation_rate",
        "first_window_avg_latency_s",
        "last_window_avg_latency_s",
        "window_latency_delta_s",
    ]
    valid_weighted = ["avg_valid_latency_s", "p95_valid_latency_s"]
    simple_mean = [
        "episode_fraction",
        "avg_replicas_per_stage",
        "min_replicas_per_stage",
        "max_replicas_per_stage",
        "single_replica_stage_rate",
        "total_deployed_replicas",
        "avg_node_compute_load",
        "max_node_compute_load",
        "p95_node_compute_load",
        "std_node_compute_load",
        "active_node_rate",
        "hot_node_rate",
        "avg_link_load",
        "max_link_load",
        "p95_link_load",
        "std_link_load",
        "active_link_rate",
        "hot_link_rate",
        "avg_node_memory_util",
        "max_node_memory_util",
        "avg_node_storage_util",
        "max_node_storage_util",
        "avg_service_memory_util",
        "max_service_memory_util",
        "avg_service_storage_util",
        "max_service_storage_util",
        "deployed_node_rate",
        "idle_deployed_node_rate",
        "scheduled_stage_invocations",
        "used_replica_rate",
        "idle_replica_rate",
        "used_replicas_per_stage",
        "avg_replica_use_entropy",
        "avg_replica_use_top1_share",
        "cross_node_stage_transition_rate",
        "slow_window_return",
        "slow_window_latency_return",
        "slow_window_avg_latency",
        "slow_window_p95_latency",
        "slow_tail_latency_cost",
        "slow_deployment_memory_cost",
        "slow_deployment_storage_cost",
        "slow_migration_cost",
        "slow_deployment_memory_fraction",
        "slow_deployment_storage_fraction",
        "slow_migration_fraction",
    ]

    aggregated: dict[str, float] = {
        "requests": float(requests.sum()),
        "request_events": float(request_events.sum()),
        "settlement_steps": float(sum(r["settlement_steps"] for r in rollouts)),
        "logical_steps": float(sum(r["logical_steps"] for r in rollouts)),
        "simulated_hours": float(sum(r["simulated_hours"] for r in rollouts)),
        "episode_complete": float(rollouts[-1]["episode_complete"]),
        "valid_requests": float(valid_requests.sum()),
        "invalid_actions": float(sum(r["invalid_actions"] for r in rollouts)),
        "deployment_updates": float(deployment_updates.sum()),
    }
    aggregated["temporal_sampling_fraction"] = float(
        aggregated["settlement_steps"] / max(aggregated["logical_steps"], 1.0)
    )
    for key in request_weighted:
        values = np.asarray([r[key] for r in rollouts], dtype=np.float64)
        finite = np.isfinite(values)
        aggregated[key] = float(np.average(values[finite], weights=requests[finite])) if finite.any() else float("nan")
    for key in valid_weighted:
        values = np.asarray([r[key] for r in rollouts], dtype=np.float64)
        finite = np.isfinite(values) & (valid_requests > 0)
        aggregated[key] = float(np.average(values[finite], weights=valid_requests[finite])) if finite.any() else float("nan")
    for key in simple_mean:
        values = np.asarray([r[key] for r in rollouts], dtype=np.float64)
        aggregated[key] = float(np.nanmean(values))
    return aggregated


def _resource_reward_components(
    env: EdgeComputingEnv,
    args: argparse.Namespace,
) -> dict[str, float]:
    compute_load = np.asarray(env.node_compute_load, dtype=np.float64)
    max_compute = float(np.max(compute_load)) if compute_load.size else 0.0
    compute_hotspot_excess = max(0.0, max_compute - float(args.compute_hotspot_threshold))

    finite_links = np.isfinite(env.scenario.bandwidth_mb_s) & env.scenario.adjacency if env.scenario is not None else None
    if finite_links is not None:
        np.fill_diagonal(finite_links, False)
        link_load = np.asarray(env.link_load[finite_links], dtype=np.float64)
        link_load = link_load[np.isfinite(link_load)]
    else:
        link_load = np.asarray([], dtype=np.float64)
    max_link = float(np.max(link_load)) if link_load.size else 0.0
    link_hotspot_excess = max(0.0, max_link - float(args.link_hotspot_threshold))

    deployed_idle_rate = 0.0
    if env.deployment is not None and env.scenario is not None:
        memory_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        storage_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        for service in env.scenario.services:
            for stage in service.stages:
                placed = env.deployment[service.service_id, stage.stage_id]
                memory_used += placed * stage.memory_gb
                storage_used += placed * stage.storage_gb
        deployed_nodes = np.logical_or(memory_used > 0.0, storage_used > 0.0)
        if deployed_nodes.any():
            deployed_idle_rate = float(
                np.mean(compute_load[deployed_nodes] <= float(args.resource_active_load_threshold))
            )

    compute_imbalance = float(np.std(compute_load)) if compute_load.size else 0.0
    link_imbalance = float(np.std(link_load)) if link_load.size else 0.0
    return {
        "compute_hotspot_penalty": float(args.compute_hotspot_coef) * compute_hotspot_excess,
        "link_hotspot_penalty": float(args.link_hotspot_coef) * link_hotspot_excess,
        "compute_imbalance_penalty": float(args.compute_imbalance_coef) * compute_imbalance,
        "link_imbalance_penalty": float(args.link_imbalance_coef) * link_imbalance,
        "idle_deployed_node_penalty": float(args.idle_deployed_node_coef) * deployed_idle_rate,
        "compute_hotspot_excess": compute_hotspot_excess,
        "link_hotspot_excess": link_hotspot_excess,
    }


def _training_reward_components(
    policy_info: dict[str, object],
    env: EdgeComputingEnv,
    args: argparse.Namespace,
) -> dict[str, float]:
    latency_cost = float(policy_info["latency_s"])
    components = _resource_reward_components(env, args)
    resource_penalty = (
        components["compute_hotspot_penalty"]
        + components["link_hotspot_penalty"]
        + components["compute_imbalance_penalty"]
        + components["link_imbalance_penalty"]
        + components["idle_deployed_node_penalty"]
    )
    reward = -latency_cost
    return {
        "train_reward": reward,
        "train_latency_cost_s": latency_cost,
        "train_resource_penalty": 0.0,
        "diagnostic_resource_penalty": resource_penalty,
        **components,
    }


def _training_reward(
    policy_info: dict[str, object],
    env: EdgeComputingEnv,
    args: argparse.Namespace,
) -> float:
    return _training_reward_components(policy_info, env, args)["train_reward"]


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
        reset_env = True
        if rollout_unit == "window":
            eval_env.reset()
            start_env_at_minute(eval_env, rollout_start_minute(args, seed_idx, eval_mode=True))
            reset_env = False
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
                reset_env=reset_env,
            )
        )
    avg_latencies = np.array([r["avg_latency_s"] for r in runs], dtype=np.float64)
    p95_latencies = np.array([r["p95_latency_s"] for r in runs], dtype=np.float64)
    avg_valid_latencies = np.array([r["avg_valid_latency_s"] for r in runs], dtype=np.float64)
    p95_valid_latencies = np.array([r["p95_valid_latency_s"] for r in runs], dtype=np.float64)
    avg_penalty_latencies = np.array([r["avg_penalty_latency_s"] for r in runs], dtype=np.float64)
    penalty_latency_shares = np.array([r["penalty_latency_share"] for r in runs], dtype=np.float64)
    invalid_actions = np.array([r["invalid_actions"] for r in runs], dtype=np.float64)
    invalid_action_rates = np.array([r["invalid_action_rate"] for r in runs], dtype=np.float64)
    violation_rates = np.array([r["deadline_violation_rate"] for r in runs], dtype=np.float64)
    deployment_updates = np.array([r["deployment_updates"] for r in runs], dtype=np.float64)
    request_events = np.array([r["request_events"] for r in runs], dtype=np.float64)
    settlement_steps = np.array([r["settlement_steps"] for r in runs], dtype=np.float64)
    logical_steps = np.array([r["logical_steps"] for r in runs], dtype=np.float64)
    avg_replicas = np.array([r["avg_replicas_per_stage"] for r in runs], dtype=np.float64)
    single_replica_rates = np.array([r["single_replica_stage_rate"] for r in runs], dtype=np.float64)
    total_replicas = np.array([r["total_deployed_replicas"] for r in runs], dtype=np.float64)
    first_window_latencies = np.array([r["first_window_avg_latency_s"] for r in runs], dtype=np.float64)
    last_window_latencies = np.array([r["last_window_avg_latency_s"] for r in runs], dtype=np.float64)
    avg_node_compute_load = np.array([r["avg_node_compute_load"] for r in runs], dtype=np.float64)
    max_node_compute_load = np.array([r["max_node_compute_load"] for r in runs], dtype=np.float64)
    p95_node_compute_load = np.array([r["p95_node_compute_load"] for r in runs], dtype=np.float64)
    std_node_compute_load = np.array([r["std_node_compute_load"] for r in runs], dtype=np.float64)
    active_node_rate = np.array([r["active_node_rate"] for r in runs], dtype=np.float64)
    hot_node_rate = np.array([r["hot_node_rate"] for r in runs], dtype=np.float64)
    avg_link_load = np.array([r["avg_link_load"] for r in runs], dtype=np.float64)
    max_link_load = np.array([r["max_link_load"] for r in runs], dtype=np.float64)
    p95_link_load = np.array([r["p95_link_load"] for r in runs], dtype=np.float64)
    std_link_load = np.array([r["std_link_load"] for r in runs], dtype=np.float64)
    active_link_rate = np.array([r["active_link_rate"] for r in runs], dtype=np.float64)
    hot_link_rate = np.array([r["hot_link_rate"] for r in runs], dtype=np.float64)
    avg_node_memory_util = np.array([r["avg_node_memory_util"] for r in runs], dtype=np.float64)
    max_node_memory_util = np.array([r["max_node_memory_util"] for r in runs], dtype=np.float64)
    avg_node_storage_util = np.array([r["avg_node_storage_util"] for r in runs], dtype=np.float64)
    max_node_storage_util = np.array([r["max_node_storage_util"] for r in runs], dtype=np.float64)
    deployed_node_rate = np.array([r["deployed_node_rate"] for r in runs], dtype=np.float64)
    idle_deployed_node_rate = np.array([r["idle_deployed_node_rate"] for r in runs], dtype=np.float64)
    scheduled_stage_invocations = np.array([r["scheduled_stage_invocations"] for r in runs], dtype=np.float64)
    used_replica_rate = np.array([r["used_replica_rate"] for r in runs], dtype=np.float64)
    idle_replica_rate = np.array([r["idle_replica_rate"] for r in runs], dtype=np.float64)
    used_replicas_per_stage = np.array([r["used_replicas_per_stage"] for r in runs], dtype=np.float64)
    avg_replica_use_entropy = np.array([r["avg_replica_use_entropy"] for r in runs], dtype=np.float64)
    avg_replica_use_top1_share = np.array([r["avg_replica_use_top1_share"] for r in runs], dtype=np.float64)
    cross_node_stage_transition_rate = np.array([r["cross_node_stage_transition_rate"] for r in runs], dtype=np.float64)
    return {
        "eval_avg_latency_s": float(avg_latencies.mean()),
        "eval_avg_latency_std": float(avg_latencies.std()),
        "eval_p95_latency_s": float(p95_latencies.mean()),
        "eval_avg_valid_latency_s": float(avg_valid_latencies.mean()),
        "eval_p95_valid_latency_s": float(p95_valid_latencies.mean()),
        "eval_avg_penalty_latency_s": float(avg_penalty_latencies.mean()),
        "eval_penalty_latency_share": float(penalty_latency_shares.mean()),
        "eval_invalid_actions": float(invalid_actions.mean()),
        "eval_invalid_action_rate": float(invalid_action_rates.mean()),
        "eval_deadline_violation_rate": float(violation_rates.mean()),
        "eval_deployment_updates": float(deployment_updates.mean()),
        "eval_request_events": float(request_events.mean()),
        "eval_settlement_steps": float(settlement_steps.mean()),
        "eval_logical_steps": float(logical_steps.mean()),
        "eval_temporal_sampling_fraction": float(settlement_steps.sum() / max(logical_steps.sum(), 1.0)),
        "eval_avg_replicas_per_stage": float(avg_replicas.mean()),
        "eval_single_replica_stage_rate": float(single_replica_rates.mean()),
        "eval_total_deployed_replicas": float(total_replicas.mean()),
        "eval_first_window_avg_latency_s": float(np.nanmean(first_window_latencies)),
        "eval_last_window_avg_latency_s": float(np.nanmean(last_window_latencies)),
        "eval_window_latency_delta_s": float(np.nanmean(last_window_latencies - first_window_latencies)),
        "eval_avg_node_compute_load": float(np.nanmean(avg_node_compute_load)),
        "eval_max_node_compute_load": float(np.nanmean(max_node_compute_load)),
        "eval_p95_node_compute_load": float(np.nanmean(p95_node_compute_load)),
        "eval_std_node_compute_load": float(np.nanmean(std_node_compute_load)),
        "eval_active_node_rate": float(np.nanmean(active_node_rate)),
        "eval_hot_node_rate": float(np.nanmean(hot_node_rate)),
        "eval_avg_link_load": float(np.nanmean(avg_link_load)),
        "eval_max_link_load": float(np.nanmean(max_link_load)),
        "eval_p95_link_load": float(np.nanmean(p95_link_load)),
        "eval_std_link_load": float(np.nanmean(std_link_load)),
        "eval_active_link_rate": float(np.nanmean(active_link_rate)),
        "eval_hot_link_rate": float(np.nanmean(hot_link_rate)),
        "eval_avg_node_memory_util": float(np.nanmean(avg_node_memory_util)),
        "eval_max_node_memory_util": float(np.nanmean(max_node_memory_util)),
        "eval_avg_node_storage_util": float(np.nanmean(avg_node_storage_util)),
        "eval_max_node_storage_util": float(np.nanmean(max_node_storage_util)),
        "eval_deployed_node_rate": float(np.nanmean(deployed_node_rate)),
        "eval_idle_deployed_node_rate": float(np.nanmean(idle_deployed_node_rate)),
        "eval_scheduled_stage_invocations": float(np.nanmean(scheduled_stage_invocations)),
        "eval_used_replica_rate": float(np.nanmean(used_replica_rate)),
        "eval_idle_replica_rate": float(np.nanmean(idle_replica_rate)),
        "eval_used_replicas_per_stage": float(np.nanmean(used_replicas_per_stage)),
        "eval_avg_replica_use_entropy": float(np.nanmean(avg_replica_use_entropy)),
        "eval_avg_replica_use_top1_share": float(np.nanmean(avg_replica_use_top1_share)),
        "eval_cross_node_stage_transition_rate": float(np.nanmean(cross_node_stage_transition_rate)),
    }


EVAL_STAT_KEYS = [
    "eval_avg_latency_s",
    "eval_avg_latency_std",
    "eval_p95_latency_s",
    "eval_avg_valid_latency_s",
    "eval_p95_valid_latency_s",
    "eval_avg_penalty_latency_s",
    "eval_penalty_latency_share",
    "eval_invalid_actions",
    "eval_invalid_action_rate",
    "eval_deadline_violation_rate",
    "eval_deployment_updates",
    "eval_request_events",
    "eval_settlement_steps",
    "eval_logical_steps",
    "eval_temporal_sampling_fraction",
    "eval_avg_replicas_per_stage",
    "eval_single_replica_stage_rate",
    "eval_total_deployed_replicas",
    "eval_first_window_avg_latency_s",
    "eval_last_window_avg_latency_s",
    "eval_window_latency_delta_s",
    "eval_avg_node_compute_load",
    "eval_max_node_compute_load",
    "eval_p95_node_compute_load",
    "eval_std_node_compute_load",
    "eval_active_node_rate",
    "eval_hot_node_rate",
    "eval_avg_link_load",
    "eval_max_link_load",
    "eval_p95_link_load",
    "eval_std_link_load",
    "eval_active_link_rate",
    "eval_hot_link_rate",
    "eval_avg_node_memory_util",
    "eval_max_node_memory_util",
    "eval_avg_node_storage_util",
    "eval_max_node_storage_util",
    "eval_deployed_node_rate",
    "eval_idle_deployed_node_rate",
    "eval_scheduled_stage_invocations",
    "eval_used_replica_rate",
    "eval_idle_replica_rate",
    "eval_used_replicas_per_stage",
    "eval_avg_replica_use_entropy",
    "eval_avg_replica_use_top1_share",
    "eval_cross_node_stage_transition_rate",
]


def prefix_eval_stats(stats: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}{key.removeprefix('eval_')}": value
        for key, value in ((key, stats.get(key, float("nan"))) for key in EVAL_STAT_KEYS)
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
    deterministic_latency_weights: list[float] = []
    stochastic_latencies: list[float] = []
    slow_policy = SlowGreedyDeploymentPolicy()

    for seed_idx in range(args.eval_seeds):
        env = build_env(args, seed_offset=seed_base + seed_idx)
        env.reset()
        if rollout_unit == "window":
            start_env_at_minute(env, rollout_start_minute(args, seed_idx, eval_mode=True))
        det_target_requests = max_requests if rollout_unit == "requests" else estimate_episode_requests(env)
        det_progress = RolloutProgress(
            label=f"diag-det seed={seed_idx + 1:02d}/{args.eval_seeds:02d}",
            target_requests=det_target_requests,
            interval_seconds=getattr(args, "progress_interval_seconds", 0.0),
            episode_hours=env.config.episode_hours,
            rollout_unit=rollout_unit,
            deployment_interval_minutes=env.config.deployment_interval_minutes,
        )
        det_stop_time_minute = None
        if rollout_unit == "window":
            det_stop_time_minute = min(
                env.current_time_minute + float(env.config.deployment_interval_minutes),
                float(env.config.episode_hours * 60),
            )
        while _rollout_active(
            env,
            max_requests=max_requests,
            rollout_unit=rollout_unit,
            stop_time_minute=det_stop_time_minute,
        ):
            requests = list(env.current_requests)
            if train_mode == "fast-only":
                if env.needs_deployment_update:
                    env.apply_deployment(slow_policy.act(env))
            else:
                agent.maybe_update_deployment(env, deterministic=True, record=False)
            actions: list[list[int]] = []
            diagnostics_by_group: list[list[dict[str, float | int]]] = []
            for request in requests:
                action, stage_stats = agent.fast_agent.schedule_with_diagnostics(env, request)
                actions.append(action)
                diagnostics_by_group.append(stage_stats)
            _, _, _, batch_info = env.step(actions)
            for request, action, stage_stats, info in zip(
                requests,
                actions,
                diagnostics_by_group,
                batch_info["group_infos"],
            ):
                deterministic_latencies.append(float(info["latency_s"]))
                deterministic_latency_weights.append(float(request.request_count))
                deterministic_actions.extend(action)
                entropies.extend(float(s["entropy"]) for s in stage_stats)
                top1_probs.extend(float(s["top1_prob"]) for s in stage_stats)
                top1_margins.extend(float(s["top1_margin"]) for s in stage_stats)
            det_progress.maybe_print(env)
        det_progress.finish(env)

        stochastic_env = build_env(args, seed_offset=seed_base + seed_idx)
        stochastic_reset_env = True
        if rollout_unit == "window":
            stochastic_env.reset()
            start_env_at_minute(stochastic_env, rollout_start_minute(args, seed_idx, eval_mode=True))
            stochastic_reset_env = False
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
            reset_env=stochastic_reset_env,
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
        "eval_deterministic_avg_latency_s": (
            float(np.average(deterministic_latencies, weights=deterministic_latency_weights))
            if deterministic_latencies
            else float("nan")
        ),
    }
    return diagnostics, deterministic_actions


def rollout_baseline(env: EdgeComputingEnv, max_requests: int) -> dict[str, float]:
    agent = build_baseline_agent()
    env.reset()
    rewards: list[float] = []
    latencies: list[float] = []
    invalid = 0
    while not env.done and env.metrics["requests"] < max_requests:
        requests = list(env.current_requests)
        actions = agent.act_batch(env)
        _, _, _, batch_info = env.step(actions)
        for request, info in zip(requests, batch_info["group_infos"]):
            rewards.extend([float(info["reward"])] * int(request.request_count))
            latencies.extend([float(info["latency_s"])] * int(request.request_count))
            invalid += int(not info["valid"]) * int(request.request_count)
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
            tick_requests = list(env.current_requests)
            expert_batch: list[list[int]] = []
            for request in tick_requests:
                expert_nodes = expert.act(env, request)
                expert_batch.append(expert_nodes)
                partial_nodes: list[int] = []
                for stage_id, node_id in enumerate(expert_nodes):
                    state = agent.fast_agent._build_state(env, request, stage_id, partial_nodes)
                    mask = agent.fast_agent._build_mask(env, request, stage_id, partial_nodes)
                    if mask[node_id]:
                        states.append(state)
                        masks.append(mask)
                        actions.append(node_id)
                    partial_nodes.append(node_id)
            env.step(expert_batch)
            collected_requests += int(sum(request.request_count for request in tick_requests))

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
            "slow_count_agent": agent.slow_agent.count_ppo.policy.state_dict(),
            "slow_placement_agent": agent.slow_agent.placement_ppo.policy.state_dict(),
            "slow_window_critic": agent.slow_agent.window_critic.state_dict(),
            "fast_agent": agent.fast_agent.ppo.policy.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(agent: HierarchicalPPOAgent, path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=agent.fast_agent.ppo.device)
    if "slow_count_agent" in checkpoint:
        agent.slow_agent.count_ppo.policy.load_state_dict(checkpoint["slow_count_agent"])
    if "slow_placement_agent" in checkpoint:
        agent.slow_agent.placement_ppo.policy.load_state_dict(checkpoint["slow_placement_agent"])
    elif "slow_agent" in checkpoint:
        agent.slow_agent.placement_ppo.policy.load_state_dict(checkpoint["slow_agent"])
    if "slow_window_critic" in checkpoint:
        agent.slow_agent.window_critic.load_state_dict(checkpoint["slow_window_critic"])
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
    replica_action_dim = effective_replicas_per_stage(args)
    agent = HierarchicalPPOAgent.from_env(
        env,
        device=args.device,
        replicas_per_stage=replica_action_dim,
        slow_lr=args.slow_lr,
        fast_lr=args.fast_lr,
        slow_k_epochs=args.slow_k_epochs,
        fast_k_epochs=args.fast_k_epochs,
        slow_entropy_coef=args.slow_entropy_coef,
        slow_count_entropy_coef=args.slow_count_entropy_coef,
        slow_placement_entropy_coef=args.slow_placement_entropy_coef,
        fast_entropy_coef=args.fast_entropy_coef,
        slow_value_coef=args.slow_value_coef,
        slow_critic_lr=args.slow_critic_lr,
        slow_critic_k_epochs=args.slow_critic_k_epochs,
        fast_value_coef=args.fast_value_coef,
        slow_target_kl=args.slow_target_kl,
        fast_target_kl=args.fast_target_kl,
        slow_minibatch_size=args.slow_minibatch_size,
        fast_minibatch_size=args.fast_minibatch_size,
        fast_policy_kind=args.fast_policy_kind,
        slow_reward_scale=args.reward_scale,
        slow_tail_latency_coef=args.slow_tail_latency_coef,
        slow_deployment_memory_coef=args.slow_deployment_memory_coef,
        slow_deployment_storage_coef=args.slow_deployment_storage_coef,
        slow_migration_coef=args.slow_migration_coef,
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
    print("  reference_style=DRL-AC-Allocation masked PPO with joint per-second settlement")
    print(f"  users={args.num_users}, nodes={args.num_edge_nodes}, services={args.num_service_types}")
    print(
        "  traffic avg={:.2f}/s min={:.2f}/s peak={:.2f}/s expected_episode({}h)={:.0f}".format(
            traffic["avg_requests_per_second"],
            traffic["min_requests_per_second"],
            traffic["max_requests_per_second"],
            args.episode_hours,
            traffic["expected_requests_per_episode"],
        )
    )
    print(f"  train_mode={args.train_mode}")
    print(f"  fast_policy_kind={args.fast_policy_kind}")
    print(f"  rollout_unit={args.rollout_unit}")
    print(f"  eval_rollout_unit={eval_rollout_unit}")
    print(f"  physical_seed={args.seed if args.physical_seed is None else args.physical_seed}")
    print(f"  pressure_profile={args.pressure_profile}")
    print(f"  demand_sampling_mode={args.demand_sampling_mode}")
    print(f"  fast_windows_per_update={args.fast_windows_per_update}")
    print(f"  slow_windows_per_update={args.slow_windows_per_update}")
    print(f"  synchronized_window_block={args.synchronized_window_block or 'disabled'}")
    print(f"  slow_tail_latency_coef={args.slow_tail_latency_coef:.2f} (P95 weight)")
    deployment_windows = max(
        int(np.ceil(args.episode_hours * 60.0 / args.deployment_interval_minutes)),
        1,
    )
    print(f"  episode_horizon={args.episode_hours}h deployment_windows={deployment_windows}")
    print("  environment_step=1s request_granularity=individual batch_settlement=joint")
    window_seconds = args.deployment_interval_minutes * 60
    effective_sampled_seconds = (
        window_seconds
        if args.sampled_seconds_per_window == 0
        else min(args.sampled_seconds_per_window, window_seconds)
    )
    print(
        f"  temporal_sampling train={effective_sampled_seconds}/{window_seconds}s "
        f"({effective_sampled_seconds / window_seconds:.1%}) eval=full"
    )
    print(f"  arrival_profile={args.arrival_profile}")
    if args.arrival_profile == "stationary":
        print("  rollout_start_mode=beginning (stationary demand; requested start modes have no effect)")
    else:
        print(f"  rollout_start_mode={args.rollout_start_mode} eval_rollout_start_mode={args.eval_rollout_start_mode}")
    print(f"  load_multipliers={args.load_multipliers}")
    print(f"  scenario_refresh_episodes={args.scenario_refresh_episodes} demand_only=true")
    print(f"  reward_mode={args.reward_mode}")
    print(f"  optimizer_reward_scale={args.reward_scale}")
    print(
        f"  max_replicas_per_stage={replica_action_dim} "
        f"actual_replica_count=learned_by_count_ppo artificial_cap={'none' if args.replicas_per_stage == 0 else 'explicit'}"
    )
    print(f"  service_resource_fraction={args.service_resource_fraction:.2f} of fixed node memory/storage")
    print(
        "  load_scales compute_task={} data_task={} node_capacity={} wired_bandwidth={}".format(
            args.task_compute_scale,
            args.task_data_scale,
            args.node_compute_capacity_scale,
            args.wired_link_bandwidth_scale,
        )
    )
    print(
        "  resource_diagnostics compute_hotspot_coef={} link_hotspot_coef={} compute_imbalance_coef={} link_imbalance_coef={} idle_deployed_node_coef={}".format(
            args.compute_hotspot_coef,
            args.link_hotspot_coef,
            args.compute_imbalance_coef,
            args.link_imbalance_coef,
            args.idle_deployed_node_coef,
        )
    )
    print(
        "  slow_window_cost memory_coef={} storage_coef={} migration_coef={} critic_lr={} critic_epochs={}".format(
            args.slow_deployment_memory_coef,
            args.slow_deployment_storage_coef,
            args.slow_migration_coef,
            args.slow_critic_lr,
            args.slow_critic_k_epochs,
        )
    )
    print(
        "  ppo slow_lr={} fast_lr={} slow_entropy={} slow_count_entropy={} slow_placement_entropy={} fast_entropy={} slow_value_coef={}".format(
            args.slow_lr,
            args.fast_lr,
            args.slow_entropy_coef,
            args.slow_count_entropy_coef if args.slow_count_entropy_coef is not None else args.slow_entropy_coef,
            args.slow_placement_entropy_coef if args.slow_placement_entropy_coef is not None else args.slow_entropy_coef,
            args.fast_entropy_coef,
            args.slow_value_coef,
        )
    )
    print(f"  ppo_minibatch slow={args.slow_minibatch_size} fast={args.fast_minibatch_size}")
    print(f"  slow_agent=service deployment every {args.deployment_interval_minutes} minutes")
    print("  fast_agent=independent request scheduling within each one-second batch")
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

    if args.eval_before_training and args.eval_interval:
        eval_stats = evaluate_agent(
            args,
            agent,
            seed_base=30_000,
            max_requests=args.eval_requests,
            train_mode=args.train_mode,
            rollout_unit=eval_rollout_unit,
        )
        seen_eval_stats: dict[str, float] = {}
        diagnostic_stats: dict[str, float] = {}
        initial_row = {
            "update": 0,
            "episode": 0,
            "demand_seed": scenario_seed_for_offset(args, 0),
            "demand_seed_end": scenario_seed_for_offset(args, 0),
            "load_multiplier": load_multiplier_for_rollout(args, 0),
            "load_multiplier_end": load_multiplier_for_rollout(args, 0),
            "start_minute": rollout_start_minute(args, 0),
            "start_minute_end": rollout_start_minute(args, 0),
            "rollouts_collected": 0,
            "window_in_episode": 0,
            "requests": 0,
            "request_events": 0,
            "settlement_steps": 0,
            "logical_steps": 0,
            "temporal_sampling_fraction": np.nan,
            "simulated_hours": np.nan,
            "episode_fraction": np.nan,
            "episode_complete": 0,
            "avg_reward": np.nan,
            "avg_train_reward": np.nan,
            "avg_train_latency_cost_s": np.nan,
            "avg_train_resource_penalty": np.nan,
            "avg_diagnostic_resource_penalty": np.nan,
            "avg_compute_hotspot_penalty": np.nan,
            "avg_link_hotspot_penalty": np.nan,
            "avg_compute_imbalance_penalty": np.nan,
            "avg_link_imbalance_penalty": np.nan,
            "avg_idle_deployed_node_penalty": np.nan,
            "avg_latency_s": np.nan,
            "p95_latency_s": np.nan,
            "avg_valid_latency_s": np.nan,
            "p95_valid_latency_s": np.nan,
            "valid_requests": 0,
            "avg_penalty_latency_s": np.nan,
            "penalty_latency_share": np.nan,
            "invalid_actions": 0,
            "invalid_action_rate": np.nan,
            "deadline_violation_rate": np.nan,
            "deployment_updates": 0,
            "avg_replicas_per_stage": np.nan,
            "min_replicas_per_stage": np.nan,
            "max_replicas_per_stage": np.nan,
            "single_replica_stage_rate": np.nan,
            "total_deployed_replicas": np.nan,
            "avg_node_compute_load": np.nan,
            "max_node_compute_load": np.nan,
            "p95_node_compute_load": np.nan,
            "std_node_compute_load": np.nan,
            "active_node_rate": np.nan,
            "hot_node_rate": np.nan,
            "avg_link_load": np.nan,
            "max_link_load": np.nan,
            "p95_link_load": np.nan,
            "std_link_load": np.nan,
            "active_link_rate": np.nan,
            "hot_link_rate": np.nan,
            "avg_node_memory_util": np.nan,
            "max_node_memory_util": np.nan,
            "avg_node_storage_util": np.nan,
            "max_node_storage_util": np.nan,
            "avg_service_memory_util": np.nan,
            "max_service_memory_util": np.nan,
            "avg_service_storage_util": np.nan,
            "max_service_storage_util": np.nan,
            "deployed_node_rate": np.nan,
            "idle_deployed_node_rate": np.nan,
            "scheduled_stage_invocations": np.nan,
            "used_replica_rate": np.nan,
            "idle_replica_rate": np.nan,
            "used_replicas_per_stage": np.nan,
            "avg_replica_use_entropy": np.nan,
            "avg_replica_use_top1_share": np.nan,
            "cross_node_stage_transition_rate": np.nan,
            "first_window_avg_latency_s": np.nan,
            "last_window_avg_latency_s": np.nan,
            "window_latency_delta_s": np.nan,
            "slow_window_return": np.nan,
            "slow_window_latency_return": np.nan,
            "slow_window_avg_latency": np.nan,
            "slow_window_p95_latency": np.nan,
            "slow_tail_latency_cost": np.nan,
            "slow_deployment_memory_cost": np.nan,
            "slow_deployment_storage_cost": np.nan,
            "slow_migration_cost": np.nan,
            "slow_deployment_memory_fraction": np.nan,
            "slow_deployment_storage_fraction": np.nan,
            "slow_migration_fraction": np.nan,
            "slow_loss": 0.0,
            "slow_updated": 0,
            "slow_windows_available": 0,
            "slow_windows_buffered": 0,
            "slow_policy_loss": 0.0,
            "slow_value_loss": 0.0,
            "slow_approx_kl": 0.0,
            "slow_window_count": 0.0,
            "slow_window_return_mean": np.nan,
            "slow_window_return_std": np.nan,
            "slow_advantage_mean": np.nan,
            "slow_advantage_std": np.nan,
            "slow_critic_explained_variance": np.nan,
            "slow_count_loss": 0.0,
            "slow_count_entropy": 0.0,
            "slow_count_approx_kl": 0.0,
            "slow_placement_loss": 0.0,
            "slow_placement_entropy": 0.0,
            "slow_placement_approx_kl": 0.0,
            "fast_loss": 0.0,
            "fast_policy_loss": 0.0,
            "fast_value_loss": 0.0,
            "fast_approx_kl": 0.0,
            "eval_avg_latency_s": eval_stats["eval_avg_latency_s"],
            "eval_avg_latency_std": eval_stats["eval_avg_latency_std"],
            "eval_p95_latency_s": eval_stats["eval_p95_latency_s"],
            "eval_avg_valid_latency_s": eval_stats["eval_avg_valid_latency_s"],
            "eval_p95_valid_latency_s": eval_stats["eval_p95_valid_latency_s"],
            "eval_avg_penalty_latency_s": eval_stats["eval_avg_penalty_latency_s"],
            "eval_penalty_latency_share": eval_stats["eval_penalty_latency_share"],
            "eval_invalid_actions": eval_stats["eval_invalid_actions"],
            "eval_invalid_action_rate": eval_stats["eval_invalid_action_rate"],
            "eval_deadline_violation_rate": eval_stats["eval_deadline_violation_rate"],
            "eval_deployment_updates": eval_stats["eval_deployment_updates"],
            "eval_request_events": eval_stats["eval_request_events"],
            "eval_settlement_steps": eval_stats["eval_settlement_steps"],
            "eval_logical_steps": eval_stats["eval_logical_steps"],
            "eval_temporal_sampling_fraction": eval_stats["eval_temporal_sampling_fraction"],
            "eval_avg_replicas_per_stage": eval_stats["eval_avg_replicas_per_stage"],
            "eval_single_replica_stage_rate": eval_stats["eval_single_replica_stage_rate"],
            "eval_total_deployed_replicas": eval_stats["eval_total_deployed_replicas"],
            "eval_avg_node_compute_load": eval_stats["eval_avg_node_compute_load"],
            "eval_max_node_compute_load": eval_stats["eval_max_node_compute_load"],
            "eval_p95_node_compute_load": eval_stats["eval_p95_node_compute_load"],
            "eval_std_node_compute_load": eval_stats["eval_std_node_compute_load"],
            "eval_active_node_rate": eval_stats["eval_active_node_rate"],
            "eval_hot_node_rate": eval_stats["eval_hot_node_rate"],
            "eval_avg_link_load": eval_stats["eval_avg_link_load"],
            "eval_max_link_load": eval_stats["eval_max_link_load"],
            "eval_p95_link_load": eval_stats["eval_p95_link_load"],
            "eval_std_link_load": eval_stats["eval_std_link_load"],
            "eval_active_link_rate": eval_stats["eval_active_link_rate"],
            "eval_hot_link_rate": eval_stats["eval_hot_link_rate"],
            "eval_avg_node_memory_util": eval_stats["eval_avg_node_memory_util"],
            "eval_max_node_memory_util": eval_stats["eval_max_node_memory_util"],
            "eval_avg_node_storage_util": eval_stats["eval_avg_node_storage_util"],
            "eval_max_node_storage_util": eval_stats["eval_max_node_storage_util"],
            "eval_deployed_node_rate": eval_stats["eval_deployed_node_rate"],
            "eval_idle_deployed_node_rate": eval_stats["eval_idle_deployed_node_rate"],
            "eval_scheduled_stage_invocations": eval_stats["eval_scheduled_stage_invocations"],
            "eval_used_replica_rate": eval_stats["eval_used_replica_rate"],
            "eval_idle_replica_rate": eval_stats["eval_idle_replica_rate"],
            "eval_used_replicas_per_stage": eval_stats["eval_used_replicas_per_stage"],
            "eval_avg_replica_use_entropy": eval_stats["eval_avg_replica_use_entropy"],
            "eval_avg_replica_use_top1_share": eval_stats["eval_avg_replica_use_top1_share"],
            "eval_cross_node_stage_transition_rate": eval_stats["eval_cross_node_stage_transition_rate"],
            "eval_first_window_avg_latency_s": eval_stats["eval_first_window_avg_latency_s"],
            "eval_last_window_avg_latency_s": eval_stats["eval_last_window_avg_latency_s"],
            "eval_window_latency_delta_s": eval_stats["eval_window_latency_delta_s"],
            **prefix_eval_stats(seen_eval_stats, "seen_eval_"),
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

    train_env: EdgeComputingEnv | None = None
    train_episode_idx = 0
    total_windows = max(
        int(np.ceil(args.episode_hours * 60.0 / args.deployment_interval_minutes)),
        1,
    )
    for update in range(args.updates):
        rollout_stats: list[dict[str, float]] = []
        demand_seeds: list[int] = []
        load_multipliers: list[float] = []
        start_minutes: list[float] = []
        episode_numbers: list[int] = []
        window_numbers: list[int] = []
        for rollout_in_update in range(args.fast_windows_per_update):
            rollout_idx = update * args.fast_windows_per_update + rollout_in_update
            start_minute = rollout_start_minute(args, rollout_idx)
            load_multiplier = load_multiplier_for_rollout(args, rollout_idx)
            batch_suffix = (
                ""
                if args.fast_windows_per_update <= 1
                else f" rollout={rollout_in_update + 1:02d}/{args.fast_windows_per_update:02d}"
            )
            if args.rollout_unit == "window":
                if args.demand_sampling_mode == "rollout":
                    train_env = build_training_env(args, rollout_idx=rollout_idx, episode_idx=rollout_idx)
                    train_env.reset()
                    start_env_at_minute(train_env, start_minute)
                    episode_number = rollout_idx + 1
                else:
                    if train_env is None or train_env.done:
                        train_env = build_training_env(args, rollout_idx=rollout_idx, episode_idx=train_episode_idx)
                        train_env.reset()
                    episode_number = train_episode_idx + 1
                env = train_env
                if args.demand_sampling_mode == "episode":
                    # Keep the fixed physical scenario and user preferences, while
                    # cycling demand intensity at each window boundary. This makes
                    # a synchronized four-window block expose distinct load cases.
                    previous_multiplier = float(env.config.demand_load_multiplier)
                    env.config.demand_load_multiplier = float(load_multiplier)
                    if not np.isclose(previous_multiplier, load_multiplier):
                        env.current_requests = env._generate_current_second_requests()
                        env.current_request = env.current_requests[0] if env.current_requests else None
                window_in_episode = min(int(env.current_time_minute // env.config.deployment_interval_minutes) + 1, total_windows)
                demand_seed = demand_seed_for_training_rollout(
                    args,
                    rollout_idx,
                    train_episode_idx if args.demand_sampling_mode == "episode" else rollout_idx,
                )
                progress_label = (
                    f"update={update + 1:03d}/{args.updates:03d}{batch_suffix} "
                    f"ep={episode_number:03d} win={window_in_episode:02d}/{total_windows:02d}"
                )
                one_stats = rollout(
                    env,
                    agent,
                    args.requests_per_update,
                    args=args,
                    reward_scale=args.reward_scale,
                    train_mode=args.train_mode,
                    progress_label=progress_label,
                    progress_interval_seconds=args.progress_interval_seconds,
                    rollout_unit=args.rollout_unit,
                    reset_env=False,
                )
                if args.rollout_unit == "window" and one_stats["episode_complete"]:
                    train_episode_idx += 1
            else:
                env = build_training_env(args, rollout_idx=rollout_idx, episode_idx=rollout_idx)
                episode_number = rollout_idx + 1
                demand_seed = demand_seed_for_training_rollout(args, rollout_idx, rollout_idx)
                one_stats = rollout(
                    env,
                    agent,
                    args.requests_per_update,
                    args=args,
                    reward_scale=args.reward_scale,
                    train_mode=args.train_mode,
                    progress_label=f"update={update + 1:03d}/{args.updates:03d}{batch_suffix}",
                    progress_interval_seconds=args.progress_interval_seconds,
                    rollout_unit=args.rollout_unit,
                )
                window_in_episode = int(one_stats["deployment_updates"])
            rollout_stats.append(one_stats)
            demand_seeds.append(demand_seed)
            load_multipliers.append(load_multiplier)
            start_minutes.append(start_minute)
            episode_numbers.append(episode_number)
            window_numbers.append(window_in_episode)

        stats = aggregate_rollout_stats(rollout_stats)
        episode_start_number = episode_numbers[0]
        episode_number = episode_numbers[-1]
        window_in_episode = window_numbers[-1]
        demand_seed = demand_seeds[0]
        demand_seed_end = demand_seeds[-1]
        load_multiplier = load_multipliers[0]
        load_multiplier_end = load_multipliers[-1]
        start_minute = start_minutes[0]
        start_minute_end = start_minutes[-1]
        fast_metrics = agent.update_fast(
            progress_label=f"update={update + 1:03d}/{args.updates:03d} fast PPO",
            progress_interval_seconds=args.progress_interval_seconds,
        )
        slow_updated = False
        slow_windows_available = agent.completed_slow_windows
        if args.train_mode == "fast-only":
            slow_metrics = agent.slow_agent.empty_update_metrics()
            agent.slow_agent.count_ppo.buffer.clear()
            agent.slow_agent.placement_ppo.buffer.clear()
            agent.window_reward = 0.0
            agent.window_steps = 0
        elif slow_windows_available >= args.slow_windows_per_update:
            slow_metrics = agent.update_slow(
                progress_label=f"update={update + 1:03d}/{args.updates:03d} slow PPO",
                progress_interval_seconds=args.progress_interval_seconds,
            )
            slow_updated = True
        else:
            slow_metrics = agent.slow_agent.empty_update_metrics()
        losses = {"slow": slow_metrics, "fast": fast_metrics}
        slow_windows_buffered = agent.completed_slow_windows

        eval_stats = {}
        seen_eval_stats = {}
        if args.eval_interval and (update + 1) % args.eval_interval == 0:
            eval_stats = evaluate_agent(
                args,
                agent,
                seed_base=30_000,
                max_requests=args.eval_requests,
                train_mode=args.train_mode,
                rollout_unit=eval_rollout_unit,
            )
        diagnostic_stats = {}
        log_row = {
            "update": update + 1,
            "episode": episode_number,
            "demand_seed": demand_seed,
            "demand_seed_end": demand_seed_end,
            "load_multiplier": load_multiplier,
            "load_multiplier_end": load_multiplier_end,
            "start_minute": start_minute,
            "start_minute_end": start_minute_end,
            "rollouts_collected": len(rollout_stats),
            "window_in_episode": window_in_episode,
            "requests": int(stats["requests"]),
            "request_events": int(stats["request_events"]),
            "settlement_steps": int(stats["settlement_steps"]),
            "logical_steps": int(round(stats["logical_steps"])),
            "temporal_sampling_fraction": stats["temporal_sampling_fraction"],
            "simulated_hours": stats["simulated_hours"],
            "episode_fraction": stats["episode_fraction"],
            "episode_complete": int(stats["episode_complete"]),
            "avg_reward": stats["avg_reward"],
            "avg_train_reward": stats["avg_train_reward"],
            "avg_train_latency_cost_s": stats["avg_train_latency_cost_s"],
            "avg_train_resource_penalty": stats["avg_train_resource_penalty"],
            "avg_diagnostic_resource_penalty": stats["avg_diagnostic_resource_penalty"],
            "avg_compute_hotspot_penalty": stats["avg_compute_hotspot_penalty"],
            "avg_link_hotspot_penalty": stats["avg_link_hotspot_penalty"],
            "avg_compute_imbalance_penalty": stats["avg_compute_imbalance_penalty"],
            "avg_link_imbalance_penalty": stats["avg_link_imbalance_penalty"],
            "avg_idle_deployed_node_penalty": stats["avg_idle_deployed_node_penalty"],
            "avg_latency_s": stats["avg_latency_s"],
            "p95_latency_s": stats["p95_latency_s"],
            "avg_valid_latency_s": stats["avg_valid_latency_s"],
            "p95_valid_latency_s": stats["p95_valid_latency_s"],
            "valid_requests": int(stats["valid_requests"]),
            "avg_penalty_latency_s": stats["avg_penalty_latency_s"],
            "penalty_latency_share": stats["penalty_latency_share"],
            "invalid_actions": int(stats["invalid_actions"]),
            "invalid_action_rate": stats["invalid_action_rate"],
            "deadline_violation_rate": stats["deadline_violation_rate"],
            "deployment_updates": int(stats["deployment_updates"]),
            "avg_replicas_per_stage": stats["avg_replicas_per_stage"],
            "min_replicas_per_stage": stats["min_replicas_per_stage"],
            "max_replicas_per_stage": stats["max_replicas_per_stage"],
            "single_replica_stage_rate": stats["single_replica_stage_rate"],
            "total_deployed_replicas": stats["total_deployed_replicas"],
            "avg_node_compute_load": stats["avg_node_compute_load"],
            "max_node_compute_load": stats["max_node_compute_load"],
            "p95_node_compute_load": stats["p95_node_compute_load"],
            "std_node_compute_load": stats["std_node_compute_load"],
            "active_node_rate": stats["active_node_rate"],
            "hot_node_rate": stats["hot_node_rate"],
            "avg_link_load": stats["avg_link_load"],
            "max_link_load": stats["max_link_load"],
            "p95_link_load": stats["p95_link_load"],
            "std_link_load": stats["std_link_load"],
            "active_link_rate": stats["active_link_rate"],
            "hot_link_rate": stats["hot_link_rate"],
            "avg_node_memory_util": stats["avg_node_memory_util"],
            "max_node_memory_util": stats["max_node_memory_util"],
            "avg_node_storage_util": stats["avg_node_storage_util"],
            "max_node_storage_util": stats["max_node_storage_util"],
            "avg_service_memory_util": stats["avg_service_memory_util"],
            "max_service_memory_util": stats["max_service_memory_util"],
            "avg_service_storage_util": stats["avg_service_storage_util"],
            "max_service_storage_util": stats["max_service_storage_util"],
            "deployed_node_rate": stats["deployed_node_rate"],
            "idle_deployed_node_rate": stats["idle_deployed_node_rate"],
            "scheduled_stage_invocations": stats["scheduled_stage_invocations"],
            "used_replica_rate": stats["used_replica_rate"],
            "idle_replica_rate": stats["idle_replica_rate"],
            "used_replicas_per_stage": stats["used_replicas_per_stage"],
            "avg_replica_use_entropy": stats["avg_replica_use_entropy"],
            "avg_replica_use_top1_share": stats["avg_replica_use_top1_share"],
            "cross_node_stage_transition_rate": stats["cross_node_stage_transition_rate"],
            "first_window_avg_latency_s": stats["first_window_avg_latency_s"],
            "last_window_avg_latency_s": stats["last_window_avg_latency_s"],
            "window_latency_delta_s": stats["window_latency_delta_s"],
            "slow_window_return": stats["slow_window_return"],
            "slow_window_latency_return": stats["slow_window_latency_return"],
            "slow_window_avg_latency": stats["slow_window_avg_latency"],
            "slow_window_p95_latency": stats["slow_window_p95_latency"],
            "slow_tail_latency_cost": stats["slow_tail_latency_cost"],
            "slow_deployment_memory_cost": stats["slow_deployment_memory_cost"],
            "slow_deployment_storage_cost": stats["slow_deployment_storage_cost"],
            "slow_migration_cost": stats["slow_migration_cost"],
            "slow_deployment_memory_fraction": stats["slow_deployment_memory_fraction"],
            "slow_deployment_storage_fraction": stats["slow_deployment_storage_fraction"],
            "slow_migration_fraction": stats["slow_migration_fraction"],
            "slow_loss": losses["slow"]["loss"],
            "slow_updated": int(slow_updated),
            "slow_windows_available": int(slow_windows_available),
            "slow_windows_buffered": int(slow_windows_buffered),
            "slow_policy_loss": losses["slow"]["policy_loss"],
            "slow_value_loss": losses["slow"]["value_loss"],
            "slow_approx_kl": losses["slow"].get("approx_kl", 0.0),
            "slow_window_count": losses["slow"].get("window_count", 0.0),
            "slow_window_return_mean": losses["slow"].get("window_return_mean", np.nan),
            "slow_window_return_std": losses["slow"].get("window_return_std", np.nan),
            "slow_advantage_mean": losses["slow"].get("advantage_mean", np.nan),
            "slow_advantage_std": losses["slow"].get("advantage_std", np.nan),
            "slow_critic_explained_variance": losses["slow"].get("critic_explained_variance", np.nan),
            "slow_count_loss": losses["slow"].get("count_loss", np.nan),
            "slow_count_entropy": losses["slow"].get("count_entropy", np.nan),
            "slow_count_approx_kl": losses["slow"].get("count_approx_kl", np.nan),
            "slow_placement_loss": losses["slow"].get("placement_loss", np.nan),
            "slow_placement_entropy": losses["slow"].get("placement_entropy", np.nan),
            "slow_placement_approx_kl": losses["slow"].get("placement_approx_kl", np.nan),
            "fast_loss": losses["fast"]["loss"],
            "fast_policy_loss": losses["fast"]["policy_loss"],
            "fast_value_loss": losses["fast"]["value_loss"],
            "fast_approx_kl": losses["fast"].get("approx_kl", 0.0),
            "eval_avg_latency_s": eval_stats.get("eval_avg_latency_s", np.nan),
            "eval_avg_latency_std": eval_stats.get("eval_avg_latency_std", np.nan),
            "eval_p95_latency_s": eval_stats.get("eval_p95_latency_s", np.nan),
            "eval_avg_valid_latency_s": eval_stats.get("eval_avg_valid_latency_s", np.nan),
            "eval_p95_valid_latency_s": eval_stats.get("eval_p95_valid_latency_s", np.nan),
            "eval_avg_penalty_latency_s": eval_stats.get("eval_avg_penalty_latency_s", np.nan),
            "eval_penalty_latency_share": eval_stats.get("eval_penalty_latency_share", np.nan),
            "eval_invalid_actions": eval_stats.get("eval_invalid_actions", np.nan),
            "eval_invalid_action_rate": eval_stats.get("eval_invalid_action_rate", np.nan),
            "eval_deadline_violation_rate": eval_stats.get("eval_deadline_violation_rate", np.nan),
            "eval_deployment_updates": eval_stats.get("eval_deployment_updates", np.nan),
            "eval_request_events": eval_stats.get("eval_request_events", np.nan),
            "eval_settlement_steps": eval_stats.get("eval_settlement_steps", np.nan),
            "eval_logical_steps": eval_stats.get("eval_logical_steps", np.nan),
            "eval_temporal_sampling_fraction": eval_stats.get("eval_temporal_sampling_fraction", np.nan),
            "eval_avg_replicas_per_stage": eval_stats.get("eval_avg_replicas_per_stage", np.nan),
            "eval_single_replica_stage_rate": eval_stats.get("eval_single_replica_stage_rate", np.nan),
            "eval_total_deployed_replicas": eval_stats.get("eval_total_deployed_replicas", np.nan),
            "eval_avg_node_compute_load": eval_stats.get("eval_avg_node_compute_load", np.nan),
            "eval_max_node_compute_load": eval_stats.get("eval_max_node_compute_load", np.nan),
            "eval_p95_node_compute_load": eval_stats.get("eval_p95_node_compute_load", np.nan),
            "eval_std_node_compute_load": eval_stats.get("eval_std_node_compute_load", np.nan),
            "eval_active_node_rate": eval_stats.get("eval_active_node_rate", np.nan),
            "eval_hot_node_rate": eval_stats.get("eval_hot_node_rate", np.nan),
            "eval_avg_link_load": eval_stats.get("eval_avg_link_load", np.nan),
            "eval_max_link_load": eval_stats.get("eval_max_link_load", np.nan),
            "eval_p95_link_load": eval_stats.get("eval_p95_link_load", np.nan),
            "eval_std_link_load": eval_stats.get("eval_std_link_load", np.nan),
            "eval_active_link_rate": eval_stats.get("eval_active_link_rate", np.nan),
            "eval_hot_link_rate": eval_stats.get("eval_hot_link_rate", np.nan),
            "eval_avg_node_memory_util": eval_stats.get("eval_avg_node_memory_util", np.nan),
            "eval_max_node_memory_util": eval_stats.get("eval_max_node_memory_util", np.nan),
            "eval_avg_node_storage_util": eval_stats.get("eval_avg_node_storage_util", np.nan),
            "eval_max_node_storage_util": eval_stats.get("eval_max_node_storage_util", np.nan),
            "eval_deployed_node_rate": eval_stats.get("eval_deployed_node_rate", np.nan),
            "eval_idle_deployed_node_rate": eval_stats.get("eval_idle_deployed_node_rate", np.nan),
            "eval_scheduled_stage_invocations": eval_stats.get("eval_scheduled_stage_invocations", np.nan),
            "eval_used_replica_rate": eval_stats.get("eval_used_replica_rate", np.nan),
            "eval_idle_replica_rate": eval_stats.get("eval_idle_replica_rate", np.nan),
            "eval_used_replicas_per_stage": eval_stats.get("eval_used_replicas_per_stage", np.nan),
            "eval_avg_replica_use_entropy": eval_stats.get("eval_avg_replica_use_entropy", np.nan),
            "eval_avg_replica_use_top1_share": eval_stats.get("eval_avg_replica_use_top1_share", np.nan),
            "eval_cross_node_stage_transition_rate": eval_stats.get("eval_cross_node_stage_transition_rate", np.nan),
            "eval_first_window_avg_latency_s": eval_stats.get("eval_first_window_avg_latency_s", np.nan),
            "eval_last_window_avg_latency_s": eval_stats.get("eval_last_window_avg_latency_s", np.nan),
            "eval_window_latency_delta_s": eval_stats.get("eval_window_latency_delta_s", np.nan),
            "eval_policy_entropy": diagnostic_stats.get("eval_policy_entropy", np.nan),
            "eval_top1_prob": diagnostic_stats.get("eval_top1_prob", np.nan),
            "eval_top1_margin": diagnostic_stats.get("eval_top1_margin", np.nan),
            "eval_action_change_rate": diagnostic_stats.get("eval_action_change_rate", np.nan),
            "eval_stochastic_avg_latency_s": diagnostic_stats.get("eval_stochastic_avg_latency_s", np.nan),
            "eval_deterministic_avg_latency_s": diagnostic_stats.get("eval_deterministic_avg_latency_s", np.nan),
            **prefix_eval_stats(seen_eval_stats, "seen_eval_"),
        }
        append_log(log_path, log_row)
        episode_label = (
            f"episode={episode_number:03d}"
            if episode_start_number == episode_number
            else f"episodes={episode_start_number:03d}-{episode_number:03d}"
        )
        print(
            "update={:03d} {} demand_seed={}-{} load={:.2f}-{:.2f} start_min={:.0f}-{:.0f} rollouts={} complete={} window={:02d} requests={} request_events={} sampled_steps={}/{} sim_hours={:.2f} episode_frac={:.1%} "
            "avg_reward={:.4f} avg_latency={:.4f}s valid_latency={:.4f}s penalty_latency={:.4f}s train_reward={:.4f} diag_res={:.4f} slowR={:.4f} invalid={} invalid_rate={:.2%} deployments={} "
            "replicas={:.2f}/{:.0f}-{:.0f} single={:.1%} "
            "used_replica={:.1%} idle_replica={:.1%} use_entropy={:.3f} top1_use={:.1%} cross_stage={:.1%} "
            "node_load={:.1%}/{:.1%} active={:.1%} hot={:.1%} link_load={:.2%}/{:.1%} active_link={:.1%} hot_link={:.1%} mem={:.1%}/{:.1%} storage={:.1%}/{:.1%} service_mem={:.1%}/{:.1%} service_storage={:.1%}/{:.1%} idle_deployed={:.1%} "
            "slow_update={} slow_buffer={}/{} slow_loss={:.4f} slow_windows={:.0f} critic_ev={:.3f} fast_loss={:.4f}".format(
                update + 1,
                episode_label,
                demand_seed,
                demand_seed_end,
                load_multiplier,
                load_multiplier_end,
                start_minute,
                start_minute_end,
                len(rollout_stats),
                int(stats["episode_complete"]),
                window_in_episode,
                int(stats["requests"]),
                int(stats["request_events"]),
                int(stats["settlement_steps"]),
                int(round(stats["logical_steps"])),
                stats["simulated_hours"],
                stats["episode_fraction"],
                stats["avg_reward"],
                stats["avg_latency_s"],
                stats["avg_valid_latency_s"],
                stats["avg_penalty_latency_s"],
                stats["avg_train_reward"],
                stats["avg_diagnostic_resource_penalty"],
                stats["slow_window_return"],
                int(stats["invalid_actions"]),
                stats["invalid_action_rate"],
                int(stats["deployment_updates"]),
                stats["avg_replicas_per_stage"],
                stats["min_replicas_per_stage"],
                stats["max_replicas_per_stage"],
                stats["single_replica_stage_rate"],
                stats["used_replica_rate"],
                stats["idle_replica_rate"],
                stats["avg_replica_use_entropy"],
                stats["avg_replica_use_top1_share"],
                stats["cross_node_stage_transition_rate"],
                stats["avg_node_compute_load"],
                stats["max_node_compute_load"],
                stats["active_node_rate"],
                stats["hot_node_rate"],
                stats["avg_link_load"],
                stats["max_link_load"],
                stats["active_link_rate"],
                stats["hot_link_rate"],
                stats["avg_node_memory_util"],
                stats["max_node_memory_util"],
                stats["avg_node_storage_util"],
                stats["max_node_storage_util"],
                stats["avg_service_memory_util"],
                stats["max_service_memory_util"],
                stats["avg_service_storage_util"],
                stats["max_service_storage_util"],
                stats["idle_deployed_node_rate"],
                int(slow_updated),
                int(slow_windows_buffered),
                args.slow_windows_per_update,
                losses["slow"]["loss"],
                losses["slow"].get("window_count", 0.0),
                losses["slow"].get("critic_explained_variance", 0.0),
                losses["fast"]["loss"],
            )
        )
        if eval_stats:
            print(
                "  eval_mean_latency={:.4f}s eval_valid_latency={:.4f}s eval_penalty_latency={:.4f}s eval_std={:.4f}s eval_p95={:.4f}s invalid={:.2f} "
                "eval_replicas={:.2f} single={:.1%} used_replica={:.1%} use_entropy={:.3f} top1_use={:.1%} cross_stage={:.1%} "
                "node_load={:.1%}/{:.1%} hot={:.1%} link_load={:.2%}/{:.1%} hot_link={:.1%}".format(
                    eval_stats["eval_avg_latency_s"],
                    eval_stats["eval_avg_valid_latency_s"],
                    eval_stats["eval_avg_penalty_latency_s"],
                    eval_stats["eval_avg_latency_std"],
                    eval_stats["eval_p95_latency_s"],
                    eval_stats["eval_invalid_actions"],
                    eval_stats["eval_avg_replicas_per_stage"],
                    eval_stats["eval_single_replica_stage_rate"],
                    eval_stats["eval_used_replica_rate"],
                    eval_stats["eval_avg_replica_use_entropy"],
                    eval_stats["eval_avg_replica_use_top1_share"],
                    eval_stats["eval_cross_node_stage_transition_rate"],
                    eval_stats["eval_avg_node_compute_load"],
                    eval_stats["eval_max_node_compute_load"],
                    eval_stats["eval_hot_node_rate"],
                    eval_stats["eval_avg_link_load"],
                    eval_stats["eval_max_link_load"],
                    eval_stats["eval_hot_link_rate"],
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
            "eval seeds={} requests_per_seed={} avg_latency={:.4f}s valid_latency={:.4f}s penalty_latency={:.4f}s std={:.4f}s p95_latency={:.4f}s invalid={:.2f} violation_rate={:.4f} replicas={:.2f} single={:.1%}".format(
                args.eval_seeds,
                args.eval_requests,
                stats["eval_avg_latency_s"],
                stats["eval_avg_valid_latency_s"],
                stats["eval_avg_penalty_latency_s"],
                stats["eval_avg_latency_std"],
                stats["eval_p95_latency_s"],
                stats["eval_invalid_actions"],
                stats["eval_deadline_violation_rate"],
                stats["eval_avg_replicas_per_stage"],
                stats["eval_single_replica_stage_rate"],
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
