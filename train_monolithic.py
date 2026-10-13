from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.comparison.checkpoint import edge_config_from_checkpoint, load_checkpoint_configuration
from edge_drl.comparison.monolithic import (
    MONOLITHIC_FAST_DESIGN,
    specialize_monolithic_fast_agent,
)
from edge_drl.comparison.monolithic_training_env import (
    MONOLITHIC_REQUEST_GENERATOR,
    MonolithicTrainingEnv,
)
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.env.environment import EdgeComputingEnv
from train_dual_ppo import (
    _load_probabilities_for_args,
    _load_strata_for_args,
    SlowLearningRateController,
    TrainingRandomStreams,
    aggregate_rollout_stats,
    append_log,
    build_episode_metrics_row,
    build_rollout_metrics_row,
    checkpoint_reference_load,
    fast_counterfactual_credit_coefficient,
    load_assignments_for_update,
    load_adjusted_rolling_latency,
    load_checkpoint,
    load_probability_for_group,
    rollout,
    save_checkpoint,
)


# This field is retained for legacy V3 checkpoint compatibility. V4 inherits
# Proposed's PPO hyperparameters and does not use temperature scaling.
MONOLITHIC_FAST_LOGIT_TEMPERATURE_MAX = 3.0

# Full feasible resource use should not be confused with reward-free waste.
# Capacity occupancy itself remains free, while these small coefficients retain
# credit for replicas that are deployed but never help serve a request.
MONOLITHIC_IDLE_REPLICA_COEF = 0.01
MONOLITHIC_COUNT_UNUSED_REPLICA_COEF = 0.005
MONOLITHIC_PLACEMENT_IDLE_COEF = 0.005


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Monolithic Service Orchestration (MSO) PPO")
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help="Proposed run metadata used only as the common physical/training configuration source",
    )
    parser.add_argument("--updates", type=int, default=320)
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help=(
            "Resume from a Monolithic latest/last/periodic checkpoint. "
            "--updates is the number of additional updates to run."
        ),
    )
    parser.add_argument(
        "--episode-minutes",
        type=int,
        default=None,
        help="Episode length; defaults to the Proposed checkpoint value",
    )
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument(
        "--fast-reservation-microbatch-size",
        type=int,
        default=1,
        help=(
            "Requests decided before updating the Fast reservation ledger. "
            "Monolithic defaults to strict per-request accounting (1)."
        ),
    )
    parser.add_argument(
        "--allow-full-resource-use",
        action="store_true",
        help=(
            "Remove memory/storage occupancy costs so Monolithic may use all "
            "physically feasible capacity; weak penalties for replicas that "
            "remain unused are retained for Slow credit assignment"
        ),
    )
    parser.add_argument(
        "--sampled-seconds-per-window",
        type=int,
        default=None,
        help=(
            "Training-only settlement samples per 10-minute deployment window; "
            "defaults to the Proposed checkpoint value; 0 restores the full window"
        ),
    )
    parser.add_argument(
        "--scenario-family",
        choices=("request_load", "compute_capacity", "wired_bandwidth", "intermediate_data", "stage_heterogeneity"),
        default="request_load",
    )
    parser.add_argument("--scenario-value", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for the synchronized random streams; defaults to the Proposed checkpoint value",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=10.0,
        help="Print in-episode rollout progress every N seconds; 0 disables it",
    )
    return parser


def _format_duration(seconds: float) -> str:
    """Format wall-clock seconds for compact progress messages."""

    if not np.isfinite(seconds):
        return "--:--"
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _resume_offsets(
    metadata: dict[str, object],
    *,
    episodes_per_update: int,
    windows_per_update: int,
) -> tuple[int, int, int]:
    """Return validated update/episode/rollout offsets for boundary resume."""

    updates = int(metadata.get("completed_updates", metadata.get("update", 0)))
    episodes = int(metadata.get("completed_episodes", updates * episodes_per_update))
    rollouts = int(metadata.get("completed_rollouts", updates * windows_per_update))
    if updates < 0 or episodes < 0 or rollouts < 0:
        raise ValueError("resume checkpoint counters must be non-negative")
    expected_episodes = updates * episodes_per_update
    expected_rollouts = updates * windows_per_update
    if episodes != expected_episodes or rollouts != expected_rollouts:
        raise ValueError(
            "Monolithic resume requires an update-boundary checkpoint: "
            f"updates={updates}, episodes={episodes}/{expected_episodes}, "
            f"rollouts={rollouts}/{expected_rollouts}"
        )
    return updates, episodes, rollouts


def _print_training_progress(
    *,
    completed_episodes: int,
    total_episodes: int,
    update: int,
    updates: int,
    episode: int,
    episodes_per_update: int,
    avg_latency_s: float,
    avg_reward: float,
    started_at: float,
    update_elapsed: float | None = None,
) -> None:
    """Report aggregate Monolithic training progress and a wall-clock ETA."""

    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = completed_episodes / elapsed
    eta = (total_episodes - completed_episodes) / rate if rate > 0.0 else float("nan")
    suffix = ""
    if update_elapsed is not None:
        suffix = f" update_time={_format_duration(update_elapsed)}"
    print(
        f"[monolithic] episodes={completed_episodes}/{total_episodes} "
        f"({100.0 * completed_episodes / max(total_episodes, 1):5.1f}%) "
        f"update={update}/{updates} episode={episode}/{episodes_per_update} "
        f"latency={avg_latency_s * 1000.0:.2f}ms reward={avg_reward:.5f} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}{suffix}",
        flush=True,
    )


def _agent_from_config(env: TraceReplayEnv, args: dict[str, object], device: str) -> HierarchicalPPOAgent:
    from inspect import signature

    kwargs: dict[str, object] = {}
    for name in signature(HierarchicalPPOAgent.from_env).parameters:
        if name in {"cls", "env"}:
            continue
        if name in args and args[name] is not None:
            kwargs[name] = args[name]
    kwargs["device"] = device
    kwargs["replicas_per_stage"] = env.config.num_edge_nodes
    # The Proposed metadata calls this field reward_scale; the agent API uses
    # slow_reward_scale. Keep the mapping local to this comparison trainer.
    if "reward_scale" in args:
        kwargs["slow_reward_scale"] = args["reward_scale"]
    if bool(args.get("fast_load_balanced_updates", False)):
        sampling_args = argparse.Namespace(**args)
        kwargs["fast_load_group_weights"] = _load_probabilities_for_args(
            sampling_args,
            len(_load_strata_for_args(sampling_args)),
        )
    agent = HierarchicalPPOAgent.from_env(env, **kwargs)
    agent.window_replica_ablation_samples = int(
        args.get("slow_replica_ablation_samples", 0) or 0
    )
    return specialize_monolithic_fast_agent(
        agent,
        str(args.get("monolithic_fast_design", "")) or None,
        logit_temperature_max=float(
            args.get(
                "monolithic_fast_logit_temperature_max",
                MONOLITHIC_FAST_LOGIT_TEMPERATURE_MAX,
            )
        ),
    )


def _optimized_monolithic_args(
    base_args: dict[str, object],
    *,
    reservation_microbatch_size: int,
    allow_full_resource_use: bool,
) -> dict[str, object]:
    """Return explicit Monolithic tuning without changing the physical scenario.

    A one-stage request concentrates all service work in one decision, so its
    reservation ledger is updated after every request rather than after the
    Proposed controller's larger stage-wise microbatch. PPO learning and
    entropy controls are inherited from Proposed for an auditable comparison.
    Resource-use tuning removes capacity-occupancy costs only when explicitly
    requested, while hard memory/storage feasibility checks remain unchanged.
    """

    if reservation_microbatch_size <= 0:
        raise ValueError("fast-reservation-microbatch-size must be positive")
    values = dict(base_args)
    values["fast_reservation_microbatch_size"] = int(reservation_microbatch_size)
    values["monolithic_fast_design"] = MONOLITHIC_FAST_DESIGN
    values["monolithic_request_generator"] = MONOLITHIC_REQUEST_GENERATOR
    # Retain the field for checkpoint schema compatibility. V4 does not use a
    # temperature-scaled policy; only V3 checkpoint loading consumes it.
    values.setdefault(
        "monolithic_fast_logit_temperature_max",
        MONOLITHIC_FAST_LOGIT_TEMPERATURE_MAX,
    )
    values["monolithic_allow_full_resource_use"] = bool(allow_full_resource_use)
    if allow_full_resource_use:
        values["slow_deployment_memory_coef"] = 0.0
        values["slow_deployment_storage_coef"] = 0.0
        values["slow_idle_replica_coef"] = MONOLITHIC_IDLE_REPLICA_COEF
        values["slow_count_unused_replica_coef"] = (
            MONOLITHIC_COUNT_UNUSED_REPLICA_COEF
        )
        values["slow_placement_idle_coef"] = MONOLITHIC_PLACEMENT_IDLE_COEF
    return values


def _synchronized_training_args(
    base_args: dict[str, object], *, seed: int, episode_minutes: int
) -> argparse.Namespace:
    """Build the same sampling configuration consumed by the Proposed trainer."""

    values = dict(base_args)
    values.update(
        {
            "seed": int(seed),
            "episode_minutes": int(episode_minutes),
            "episode_hours": None,
            "scenario_refresh_episodes": int(values.get("scenario_refresh_episodes", 1)),
            "demand_sampling_mode": str(values.get("demand_sampling_mode", "episode")),
            "demand_scenario_schedule": str(values.get("demand_scenario_schedule", "stream")),
            "load_sampling_mode": str(values.get("load_sampling_mode", "cyclic")),
            "load_multipliers": str(values.get("load_multipliers", "1.0")),
            "load_strata": str(values.get("load_strata", "")),
            "load_stratum_probabilities": str(values.get("load_stratum_probabilities", "")),
        }
    )
    return argparse.Namespace(**values)


def _monolithic_rollout_args(
    base_args: dict[str, object],
    *,
    sampled_seconds_per_window: int,
    fast_counterfactual_credit_coef: float,
) -> argparse.Namespace:
    """Build rollout controls shared with the source Proposed experiment.

    Monolithic differs from Proposed only through stage collapse. Reward-side
    controls introduced after this trainer was first written must therefore be
    forwarded explicitly instead of silently falling back to rollout defaults.
    """

    slow_counterfactual_coef = base_args.get("slow_counterfactual_credit_coef", 0.0)
    return argparse.Namespace(
        reward_mode="latency",
        compute_hotspot_threshold=0.60,
        link_hotspot_threshold=0.60,
        resource_active_load_threshold=0.01,
        compute_hotspot_coef=0.0,
        link_hotspot_coef=0.0,
        compute_imbalance_coef=0.0,
        link_imbalance_coef=0.0,
        idle_deployed_node_coef=0.0,
        fast_congestion_credit_coef=1.0,
        fast_counterfactual_credit_coef=float(fast_counterfactual_credit_coef),
        slow_counterfactual_credit_coef=float(
            0.0 if slow_counterfactual_coef is None else slow_counterfactual_coef
        ),
        slow_replica_ablation_samples=int(
            base_args.get("slow_replica_ablation_samples", 0) or 0
        ),
        fast_controllable_latency_credit=bool(
            base_args.get("fast_controllable_latency_credit", False)
        ),
        # Oracle search is a Proposed diagnostic, not a training reward term.
        # Keep it disabled here so Monolithic pays no avoidable rollout cost.
        fast_oracle_diagnostic_requests=0,
        fast_oracle_beam_width=int(base_args.get("fast_oracle_beam_width", 32)),
        fast_oracle_candidates_per_stage=int(
            base_args.get("fast_oracle_candidates_per_stage", 8)
        ),
        sampled_seconds_per_window=int(sampled_seconds_per_window),
        sampling_max_interval_seconds=float(base_args.get("sampling_max_interval_seconds", 0.0)),
    )


def _synchronized_episode(
    *,
    base_args: dict[str, object],
    physical_seed: int,
    environment_seed: int,
    demand_seed: int,
    episode_minutes: int,
    load_assignments: list[tuple[float, int]],
    windows_per_episode: int,
    point: ExperimentPoint,
) -> MonolithicTrainingEnv:
    """Create one Monolithic episode from Proposed's seeds and load schedule.

    Use Proposed's on-demand generator and inherited temporal sampler. Collapse
    only the execution view, not the catalogue used to draw individual tasks.
    """

    if len(load_assignments) != windows_per_episode:
        raise ValueError("load assignment count must equal windows per episode")
    source_args = dict(base_args)
    # edge_config_from_checkpoint derives physical_seed from checkpoint_args
    # seed. Keep the Proposed physical seed even when a caller overrides the
    # master seed for a separate experiment.
    source_args["seed"] = int(physical_seed)
    first_multiplier, first_group = load_assignments[0]
    config = edge_config_from_checkpoint(
        source_args,
        episode_minutes=episode_minutes,
        environment_seed=environment_seed,
        demand_seed=demand_seed,
        demand_load_multiplier=float(first_multiplier),
    )
    config.demand_load_group = int(first_group)

    env = MonolithicTrainingEnv(config, point)
    env.reset()
    return env


def main() -> None:
    cli = build_parser().parse_args()
    if cli.updates <= 0 or cli.episodes_per_update <= 0:
        raise ValueError("updates and episodes-per-update must be positive")
    if cli.fast_reservation_microbatch_size <= 0:
        raise ValueError("fast-reservation-microbatch-size must be positive")
    if cli.progress_interval_seconds < 0.0:
        raise ValueError("progress-interval-seconds must be non-negative")
    base_path, base_args, _ = load_checkpoint_configuration(cli.base_checkpoint)
    base_args = _optimized_monolithic_args(
        base_args,
        reservation_microbatch_size=cli.fast_reservation_microbatch_size,
        allow_full_resource_use=cli.allow_full_resource_use,
    )
    base_seed = int(base_args.get("seed", 2026))
    effective_seed = base_seed if cli.seed is None else int(cli.seed)
    resume_path: Path | None = None
    resume_args: dict[str, object] = {}
    resume_internal: dict[str, object] = {}
    if cli.load_checkpoint:
        resume_path, resume_args, resume_internal = load_checkpoint_configuration(
            cli.load_checkpoint
        )
        if resume_args.get("training_scheme") not in {"Monolithic", "MSO"}:
            raise ValueError("--load-checkpoint must be an MSO training checkpoint")
        if resume_internal.get("resumable") is False:
            raise ValueError(
                "this Monolithic checkpoint is evaluation-only; resume from latest.pt, "
                "last.pt, or a periodic update checkpoint"
            )
    effective_episode_minutes = int(
        base_args.get("episode_minutes", 60) if cli.episode_minutes is None else cli.episode_minutes
    )
    effective_sampled_seconds = int(
        base_args.get("sampled_seconds_per_window", 60)
        if cli.sampled_seconds_per_window is None
        else cli.sampled_seconds_per_window
    )
    if effective_episode_minutes <= 0:
        raise ValueError("episode-minutes must be positive")
    if effective_sampled_seconds < 0:
        raise ValueError("sampled-seconds-per-window must be non-negative")
    # Match the Proposed entrypoint: seed global model initialization before
    # constructing either the environment-facing networks or their optimizers.
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)
    if effective_seed != base_seed:
        print(
            f"[monolithic] warning: --seed={effective_seed} differs from Proposed checkpoint seed={base_seed}; "
            "demand streams will intentionally differ",
            flush=True,
        )
    synchronized_args = _synchronized_training_args(
        base_args,
        seed=effective_seed,
        episode_minutes=effective_episode_minutes,
    )
    if synchronized_args.demand_sampling_mode != "episode":
        raise ValueError(
            "synchronized Monolithic training currently requires demand_sampling_mode=episode"
        )
    physical_seed = int(
        base_args.get("physical_seed")
        if base_args.get("physical_seed") is not None
        else base_seed
    )
    source_args = dict(base_args)
    source_args["seed"] = physical_seed
    config = edge_config_from_checkpoint(
        source_args,
        episode_minutes=effective_episode_minutes,
        environment_seed=effective_seed + 100_000,
        demand_seed=effective_seed,
        demand_load_multiplier=1.0,
    )
    point = ExperimentPoint(cli.scenario_family, cli.scenario_value, f"{cli.scenario_family}:{cli.scenario_value:g}")
    base_env = MonolithicTrainingEnv(config, point)
    base_env.reset()
    # Persist resolved defaults as well as explicit flags, so a repaired run can
    # resume without silently changing its sampler or physical constraints.
    base_args.update({
        "network_constraint_mode": config.network_constraint_mode,
        "service_calibration_mode": config.service_calibration_mode,
        "temporal_sampling_mode": config.temporal_sampling_mode,
        "sampling_max_interval_seconds": float(base_args.get("sampling_max_interval_seconds", 0.0)),
    })
    agent = _agent_from_config(base_env, base_args, cli.device)
    total_windows_per_episode = max(
        int(np.ceil(effective_episode_minutes / max(config.deployment_interval_minutes, 1))),
        1,
    )
    window_seconds = int(config.deployment_interval_minutes * 60)
    effective_sampled_seconds = (
        window_seconds
        if effective_sampled_seconds == 0
        else min(effective_sampled_seconds, window_seconds)
    )
    sampling_cap = float(base_args.get("sampling_max_interval_seconds", 0.0))
    if sampling_cap > 0:
        effective_sampled_seconds = max(effective_sampled_seconds, int(np.ceil(window_seconds / sampling_cap)))
    windows_per_update = total_windows_per_episode * cli.episodes_per_update
    if resume_path is not None:
        expected_resume = {
            "seed": effective_seed,
            "episode_minutes": effective_episode_minutes,
            "episodes_per_update": cli.episodes_per_update,
            "sampled_seconds_per_window": effective_sampled_seconds,
            "fast_reservation_microbatch_size": cli.fast_reservation_microbatch_size,
            "fast_lr": float(base_args["fast_lr"]),
            "fast_entropy_target": float(base_args["fast_entropy_target"]),
            "fast_entropy_max_coef": float(base_args["fast_entropy_max_coef"]),
            "fast_entropy_adaptation_rate": float(
                base_args["fast_entropy_adaptation_rate"]
            ),
            "monolithic_fast_design": base_args["monolithic_fast_design"],
            "monolithic_request_generator": MONOLITHIC_REQUEST_GENERATOR,
            "network_constraint_mode": base_env.config.network_constraint_mode,
            "service_calibration_mode": base_env.config.service_calibration_mode,
            "temporal_sampling_mode": base_env.config.temporal_sampling_mode,
            "sampling_max_interval_seconds": base_args.get("sampling_max_interval_seconds", 0.0),
            "monolithic_fast_logit_temperature_max": float(
                base_args["monolithic_fast_logit_temperature_max"]
            ),
            "monolithic_allow_full_resource_use": cli.allow_full_resource_use,
            "comparison_scenario_family": cli.scenario_family,
        }
        for key, expected in expected_resume.items():
            actual = resume_args.get(key)
            if actual != expected:
                raise ValueError(
                    f"resume configuration mismatch for {key}: "
                    f"checkpoint={actual!r}, requested={expected!r}"
                )
        actual_value = float(resume_args.get("comparison_scenario_value", np.nan))
        if not np.isclose(actual_value, float(cli.scenario_value)):
            raise ValueError(
                "resume configuration mismatch for comparison_scenario_value: "
                f"checkpoint={actual_value!r}, requested={cli.scenario_value!r}"
            )
    training_random_streams = TrainingRandomStreams(effective_seed)
    if resume_path is not None:
        resume_internal = load_checkpoint(agent, resume_path, training_random_streams)
        if resume_internal.get("deterministic_initialization") is not True:
            print(
                "[monolithic] warning: resume checkpoint predates deterministic "
                "initialization metadata; do not use it as a new formal-training seed",
                flush=True,
            )
    resume_update_offset, resume_episode_offset, training_rollout_idx = _resume_offsets(
        resume_internal,
        episodes_per_update=cli.episodes_per_update,
        windows_per_update=windows_per_update,
    )
    target_update = resume_update_offset + cli.updates
    run_episodes = cli.updates * cli.episodes_per_update
    best_checkpoint_window = int(base_args.get("best_checkpoint_window", 10))
    checkpoint_interval = int(base_args.get("checkpoint_interval", 20))
    reference_checkpoint_load = checkpoint_reference_load(synchronized_args)
    best_checkpoint_score = float(
        resume_internal.get("best_checkpoint_score", float("inf"))
    )
    best_checkpoint_ready = bool(resume_internal.get("best_checkpoint_ready", False))
    checkpoint_selection_history: list[dict[str, float]] = []
    raw_selection_history = resume_internal.get("checkpoint_selection_history", [])
    if isinstance(raw_selection_history, list):
        for item in raw_selection_history:
            if isinstance(item, dict):
                checkpoint_selection_history.append(
                    {
                        "latency_s": float(item.get("latency_s", np.nan)),
                        "load_multiplier": float(item.get("load_multiplier", np.nan)),
                    }
                )
    slow_lr_controller = SlowLearningRateController(
        enabled=bool(base_args.get("slow_lr_decay", False)),
        patience=int(base_args.get("slow_lr_decay_patience", 10)),
        factor=float(base_args.get("slow_lr_decay_factor", 0.5)),
        min_delta=float(base_args.get("slow_lr_decay_min_delta", 5e-4)),
        min_lr=float(base_args.get("slow_min_lr", 1e-5)),
        count_min_lr=float(
            base_args.get(
                "slow_count_min_lr",
                base_args.get("slow_min_lr", 1e-5),
            )
        ),
        placement_min_lr=float(
            base_args.get(
                "slow_placement_min_lr",
                base_args.get("slow_min_lr", 1e-5),
            )
        ),
        kl_floor_fraction=float(base_args.get("slow_lr_kl_floor_fraction", 0.10)),
        max_reductions=int(base_args.get("slow_lr_max_reductions", 1)),
        cooldown_updates=int(base_args.get("slow_lr_cooldown_updates", 20)),
        state=resume_internal.get("slow_lr_controller_state"),
    )
    slow_lr_floor_restored = slow_lr_controller.enforce_minimum_lrs(agent)
    if slow_lr_floor_restored:
        print(
            "[monolithic] restored Slow LR floors "
            f"count={SlowLearningRateController.optimizer_lr(agent.slow_agent.count_ppo.optimizer):.3g} "
            f"placement={SlowLearningRateController.optimizer_lr(agent.slow_agent.placement_ppo.optimizer):.3g}",
            flush=True,
        )
    run_name = cli.run_name or f"mso_ppo_{cli.scenario_family}_{cli.scenario_value:g}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cli.run_root) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    update_log_path = log_dir / "training.csv"
    episode_log_path = log_dir / "episode_metrics.csv"
    rollout_log_path = log_dir / "rollout_metrics.csv"
    inherited_best_checkpoint: str | None = None
    if resume_path is not None:
        prior_best = resume_path.parent / "best.pt"
        if prior_best.is_file():
            destination = checkpoint_dir / "best.pt"
            shutil.copy2(prior_best, destination)
            inherited_best_checkpoint = str(prior_best)
    history: list[dict[str, float]] = []
    synchronization = {
        "source_proposed_checkpoint": str(base_path),
        "master_seed": effective_seed,
        "physical_seed": physical_seed,
        "demand_sampling_mode": synchronized_args.demand_sampling_mode,
        "demand_scenario_schedule": synchronized_args.demand_scenario_schedule,
        "load_sampling_mode": synchronized_args.load_sampling_mode,
        "load_strata": synchronized_args.load_strata,
        "load_stratum_probabilities": synchronized_args.load_stratum_probabilities,
        "fast_windows_per_update": total_windows_per_episode * cli.episodes_per_update,
        "slow_windows_per_update": total_windows_per_episode * cli.episodes_per_update,
        "sampled_seconds_per_window": effective_sampled_seconds,
        "best_checkpoint_window": best_checkpoint_window,
        "checkpoint_reference_load": reference_checkpoint_load,
        "slow_lr_decay": slow_lr_controller.enabled,
        "slow_lr_decay_patience": slow_lr_controller.patience,
        "slow_lr_decay_factor": slow_lr_controller.factor,
        "slow_lr_decay_min_delta": slow_lr_controller.min_delta,
        "slow_min_lr": slow_lr_controller.min_lr,
        "resume_checkpoint": None if resume_path is None else str(resume_path),
        "resume_update_offset": resume_update_offset,
        "resume_episode_offset": resume_episode_offset,
        "resume_rollout_offset": training_rollout_idx,
        "inherited_best_checkpoint": inherited_best_checkpoint,
        "request_trace_rng": "Proposed environment_seed stream",
        "request_generator": MONOLITHIC_REQUEST_GENERATOR,
        "trace_pre_generated": False,
        "window_load_observation_sync": True,
        "structural_difference": (
            "Monolithic complete-service stage collapse with batch-trajectory "
            "Fast credit and remaining-demand observation"
        ),
        "monolithic_fast_design": base_args["monolithic_fast_design"],
        "monolithic_fast_logit_temperature_max": float(
            base_args["monolithic_fast_logit_temperature_max"]
        ),
    }
    training_manifest: dict[str, object] = {
        "source_proposed_checkpoint": str(base_path),
        "synchronization": synchronization,
        "updates": [],
    }
    manifest_path = run_dir / "training_manifest.json"
    synchronization["training_manifest"] = str(manifest_path)
    training_manifest["synchronization"] = synchronization
    manifest_path.write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    training_started_at = time.monotonic()
    print(
        f"[monolithic] start run={run_name} additional_updates={cli.updates} "
        f"resume_update={resume_update_offset} target_update={target_update} "
        f"episodes_per_update={cli.episodes_per_update} episode_minutes={effective_episode_minutes} "
        f"windows_per_episode={total_windows_per_episode} "
        f"sampled_seconds_per_window={effective_sampled_seconds}/{window_seconds} "
        f"fast_reservation_microbatch={cli.fast_reservation_microbatch_size} "
        f"fast_lr={float(base_args['fast_lr']):g} "
        f"fast_entropy_target={float(base_args['fast_entropy_target']):g} "
        f"fast_entropy_max={float(base_args['fast_entropy_max_coef']):g} "
        f"fast_design={base_args['monolithic_fast_design']} "
        f"request_generator={MONOLITHIC_REQUEST_GENERATOR} "
        f"fast_temperature=disabled-v4 "
        f"allow_full_resource_use={cli.allow_full_resource_use} "
        f"slow_counterfactual_credit={float(base_args.get('slow_counterfactual_credit_coef', 0.0) or 0.0):g} "
        f"run_episodes={run_episodes} device={cli.device}",
        flush=True,
    )

    # Reuse Proposed's seeds, load assignments and on-demand RNG advancement.
    for local_update in range(1, cli.updates + 1):
        update = resume_update_offset + local_update
        fast_counterfactual_coef = fast_counterfactual_credit_coefficient(
            synchronized_args,
            update - 1,
        )
        update_started_at = time.monotonic()
        episode_stats_for_update: list[dict[str, float]] = []
        update_episode_manifest: list[dict[str, object]] = []
        windows_this_update = total_windows_per_episode * cli.episodes_per_update
        update_load_assignments = load_assignments_for_update(
            synchronized_args,
            update - 1,
            windows_this_update,
            rollout_start_idx=training_rollout_idx,
            rng=training_random_streams.load_rng,
        )
        if point.family == "request_load":
            update_load_assignments = [
                (float(multiplier) * point.value, group)
                for multiplier, group in update_load_assignments
            ]
        for episode_offset in range(cli.episodes_per_update):
            episode_index = (
                resume_episode_offset
                + (local_update - 1) * cli.episodes_per_update
                + episode_offset
            )
            demand_seed = training_random_streams.demand_seed_for_episode(
                episode_index,
                synchronized_args.scenario_refresh_episodes,
            )
            environment_seed = training_random_streams.next_environment_seed()
            assignment_start = episode_offset * total_windows_per_episode
            episode_assignments = update_load_assignments[
                assignment_start : assignment_start + total_windows_per_episode
            ]
            env = _synchronized_episode(
                base_args=base_args,
                physical_seed=physical_seed,
                environment_seed=environment_seed,
                demand_seed=demand_seed,
                episode_minutes=effective_episode_minutes,
                load_assignments=episode_assignments,
                windows_per_episode=total_windows_per_episode,
                point=point,
            )
            episode_config = env.config
            update_episode_manifest.append(
                {
                    "episode": episode_index + 1,
                    "demand_seed": int(demand_seed),
                    "environment_seed": int(environment_seed),
                    "physical_seed": int(
                        episode_config.physical_seed
                        if episode_config.physical_seed is not None
                        else physical_seed
                    ),
                    "load_multipliers": [float(value) for value, _ in episode_assignments],
                    "load_groups": [int(group) for _, group in episode_assignments],
                    "request_generator": MONOLITHIC_REQUEST_GENERATOR,
                    "trace_pre_generated": False,
                    "logical_steps": effective_episode_minutes * 60,
                }
            )
            window_stats: list[dict[str, float]] = []
            window_contexts: list[dict[str, float | int | str]] = []
            for window_offset in range(total_windows_per_episode):
                load_multiplier, load_group = episode_assignments[window_offset]
                env.set_window_load(load_multiplier, load_group)
                one_stats = rollout(
                    env,
                    agent,
                    max_requests=0,
                    args=_monolithic_rollout_args(
                        base_args,
                        sampled_seconds_per_window=effective_sampled_seconds,
                        fast_counterfactual_credit_coef=fast_counterfactual_coef,
                    ),
                    reward_scale=float(base_args.get("reward_scale", 1.0)),
                    train_mode="joint",
                    rollout_unit="window",
                    reset_env=False,
                    progress_label=(
                        f"monolithic update {update}/{target_update} "
                        f"episode {episode_offset + 1}/{cli.episodes_per_update} "
                        f"window {window_offset + 1}/{total_windows_per_episode}"
                    ),
                    progress_interval_seconds=cli.progress_interval_seconds,
                )
                window_stats.append(one_stats)
                context: dict[str, float | int | str] = {
                    "rollout": training_rollout_idx + 1,
                    "update": update,
                    "episode": episode_index + 1,
                    "window": window_offset + 1,
                    "training_phase": "joint_1to1",
                    "demand_seed": int(demand_seed),
                    "environment_seed": int(environment_seed),
                    "load_multiplier": float(load_multiplier),
                    "load_group": int(load_group),
                    "load_target_probability": load_probability_for_group(
                        synchronized_args,
                        int(load_group),
                    ),
                    "start_minute": float(window_offset * config.deployment_interval_minutes),
                }
                window_contexts.append(context)
                append_log(
                    rollout_log_path,
                    build_rollout_metrics_row(
                        one_stats,
                        context,
                        episode_offset * total_windows_per_episode + window_offset + 1,
                    ),
                )
                training_rollout_idx += 1
            episode_stats = aggregate_rollout_stats(window_stats)
            update_episode_manifest[-1].update({
                "generated_request_batches": env.generated_request_batches,
                "generated_request_events": env.generated_request_events,
                "sampled_settlement_steps": int(episode_stats["settlement_steps"]),
            })
            episode_stats_for_update.append(episode_stats)
            append_log(
                episode_log_path,
                build_episode_metrics_row(window_stats, window_contexts),
            )
            completed_run_episodes = (
                (local_update - 1) * cli.episodes_per_update + episode_offset + 1
            )
            _print_training_progress(
                completed_episodes=completed_run_episodes,
                total_episodes=run_episodes,
                update=update,
                updates=target_update,
                episode=episode_offset + 1,
                episodes_per_update=cli.episodes_per_update,
                avg_latency_s=float(episode_stats.get("avg_latency_s", 0.0)),
                avg_reward=float(episode_stats.get("avg_reward", 0.0)),
                started_at=training_started_at,
            )
        training_manifest["updates"].append(
            {
                "update": update,
                "episodes": update_episode_manifest,
            }
        )
        manifest_path.write_text(
            json.dumps(training_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stats = aggregate_rollout_stats(episode_stats_for_update)
        load_multiplier_mean = float(
            np.mean([multiplier for multiplier, _ in update_load_assignments])
        )
        checkpoint_selection_history.append(
            {
                "latency_s": float(stats.get("avg_latency_s", np.nan)),
                "load_multiplier": load_multiplier_mean,
            }
        )
        checkpoint_score, checkpoint_load_slope = load_adjusted_rolling_latency(
            checkpoint_selection_history,
            best_checkpoint_window,
            reference_checkpoint_load,
        )
        checkpoint_ready = len(checkpoint_selection_history) >= best_checkpoint_window
        should_replace_best = (
            (checkpoint_ready and not best_checkpoint_ready)
            or (
                checkpoint_ready == best_checkpoint_ready
                and checkpoint_score < best_checkpoint_score
            )
        )
        # The rollout score belongs to the policy *before* this update. Save
        # that exact policy now; otherwise best.pt would contain parameters
        # that have never produced the score used to select them.
        if should_replace_best:
            best_checkpoint_score = checkpoint_score
            best_checkpoint_ready = checkpoint_ready
            save_checkpoint(
                agent,
                checkpoint_dir / "best.pt",
                {
                    "args": {
                        **base_args,
                        "seed": effective_seed,
                        "physical_seed": physical_seed,
                        "episode_minutes": effective_episode_minutes,
                        "episodes_per_update": cli.episodes_per_update,
                        "sampled_seconds_per_window": effective_sampled_seconds,
                        "updates": target_update,
                        "additional_updates": cli.updates,
                        "load_checkpoint": (
                            None if resume_path is None else str(resume_path)
                        ),
                        "progress_interval_seconds": cli.progress_interval_seconds,
                        "run_name": run_name,
                        "training_scheme": "MSO",
                        "comparison_scenario_family": cli.scenario_family,
                        "comparison_scenario_value": cli.scenario_value,
                    },
                    "synchronization": synchronization,
                    "update": update,
                    "policy_completed_updates": update - 1,
                    "evaluated_by_rollout_update": update,
                    "resumable": False,
                    "selection_kind": "load_adjusted_rolling_train_pre_update",
                    "best_checkpoint_score": best_checkpoint_score,
                    "best_checkpoint_ready": best_checkpoint_ready,
                    "checkpoint_score": checkpoint_score,
                    "checkpoint_score_load_slope": checkpoint_load_slope,
                    "checkpoint_reference_load": reference_checkpoint_load,
                    "checkpoint_selection_window": best_checkpoint_window,
                    "checkpoint_selection_history": checkpoint_selection_history,
                },
                training_random_streams,
            )
        fast_metrics = agent.update_fast(
            progress_label=f"monolithic update {update} fast PPO",
            progress_interval_seconds=cli.progress_interval_seconds,
        )
        slow_windows_available = agent.completed_slow_windows
        if slow_windows_available < windows_this_update:
            raise RuntimeError(
                "Monolithic 1:1 update expected "
                f"{windows_this_update} completed Slow windows, got {slow_windows_available}"
            )
        slow_metrics = agent.update_slow(
            progress_label=f"monolithic update {update} slow PPO",
            progress_interval_seconds=cli.progress_interval_seconds,
        )
        update_metrics = {"fast": fast_metrics, "slow": slow_metrics}
        slow_updated = True
        slow_lr_decayed = slow_lr_controller.observe(
            checkpoint_score,
            ready=checkpoint_ready,
            slow_updated=slow_updated,
            agent=agent,
            count_approx_kl=slow_metrics.get("count_approx_kl"),
            placement_approx_kl=slow_metrics.get("placement_approx_kl"),
            count_target_kl=float(base_args.get("slow_count_target_kl", 0.01)),
            placement_target_kl=float(
                base_args.get("slow_placement_target_kl", 0.012)
            ),
        )
        slow_count_lr = SlowLearningRateController.optimizer_lr(
            agent.slow_agent.count_ppo.optimizer
        )
        slow_placement_lr = SlowLearningRateController.optimizer_lr(
            agent.slow_agent.placement_ppo.optimizer
        )
        row = {
            "update": float(update),
            "avg_latency_s": float(stats.get("avg_latency_s", 0.0)),
            "p95_latency_s": float(stats.get("p95_latency_s", 0.0)),
            "avg_reward": float(stats.get("avg_reward", 0.0)),
            "load_multiplier_mean": load_multiplier_mean,
            "checkpoint_score": checkpoint_score,
            "checkpoint_score_load_slope": checkpoint_load_slope,
            "checkpoint_ready": float(checkpoint_ready),
            "slow_count_lr": slow_count_lr,
            "slow_placement_lr": slow_placement_lr,
            "slow_lr_bad_updates": float(slow_lr_controller.bad_updates),
            "slow_lr_reductions": float(slow_lr_controller.reductions),
            "slow_lr_decayed": float(slow_lr_decayed),
            "slow_count_lr_bad_updates": float(
                slow_lr_controller.actor_metric("count", "bad_updates")
            ),
            "slow_placement_lr_bad_updates": float(
                slow_lr_controller.actor_metric("placement", "bad_updates")
            ),
            "slow_count_lr_reductions": float(
                slow_lr_controller.actor_metric("count", "reductions")
            ),
            "slow_placement_lr_reductions": float(
                slow_lr_controller.actor_metric("placement", "reductions")
            ),
            "fast_counterfactual_credit_coef": fast_counterfactual_coef,
            **{f"fast_{key}": float(value) for key, value in update_metrics.get("fast", {}).items() if isinstance(value, (int, float))},
            **{f"slow_{key}": float(value) for key, value in update_metrics.get("slow", {}).items() if isinstance(value, (int, float))},
        }
        history.append(row)
        append_log(update_log_path, row)
        metadata = {
            "args": {
                **base_args,
                "seed": effective_seed,
                "physical_seed": physical_seed,
                "episode_minutes": effective_episode_minutes,
                "episodes_per_update": cli.episodes_per_update,
                "sampled_seconds_per_window": effective_sampled_seconds,
                "updates": target_update,
                "additional_updates": cli.updates,
                "load_checkpoint": None if resume_path is None else str(resume_path),
                "progress_interval_seconds": cli.progress_interval_seconds,
                "run_name": run_name,
                "training_scheme": "MSO",
                "comparison_scenario_family": cli.scenario_family,
                "comparison_scenario_value": cli.scenario_value,
            },
            "synchronization": synchronization,
            "update": update,
            "completed_updates": update,
            "completed_rollouts": training_rollout_idx,
            "completed_episodes": (
                resume_episode_offset + local_update * cli.episodes_per_update
            ),
            "deterministic_initialization": True,
            "best_checkpoint_score": best_checkpoint_score,
            "best_checkpoint_ready": best_checkpoint_ready,
            "checkpoint_score": checkpoint_score,
            "checkpoint_score_load_slope": checkpoint_load_slope,
            "checkpoint_reference_load": reference_checkpoint_load,
            "checkpoint_selection_window": best_checkpoint_window,
            "checkpoint_selection_history": checkpoint_selection_history,
            "slow_lr_controller_state": slow_lr_controller.state_dict(),
            "history": row,
        }
        save_checkpoint(
            agent,
            checkpoint_dir / "latest.pt",
            {**metadata, "selection_kind": "latest"},
            training_random_streams,
        )
        if checkpoint_interval > 0 and update % checkpoint_interval == 0:
            save_checkpoint(
                agent,
                checkpoint_dir / f"update_{update:04d}.pt",
                {**metadata, "selection_kind": "periodic_snapshot"},
                training_random_streams,
            )
        (run_dir / "training_log.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        score = float(stats.get("avg_latency_s", float("inf")))
        _print_training_progress(
            completed_episodes=local_update * cli.episodes_per_update,
            total_episodes=run_episodes,
            update=update,
            updates=target_update,
            episode=cli.episodes_per_update,
            episodes_per_update=cli.episodes_per_update,
            avg_latency_s=score,
            avg_reward=float(stats.get("avg_reward", 0.0)),
            started_at=training_started_at,
            update_elapsed=time.monotonic() - update_started_at,
        )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "args": {
                    **base_args,
                    "seed": effective_seed,
                    "physical_seed": physical_seed,
                    "episode_minutes": effective_episode_minutes,
                    "episodes_per_update": cli.episodes_per_update,
                    "sampled_seconds_per_window": effective_sampled_seconds,
                    "updates": target_update,
                    "additional_updates": cli.updates,
                    "load_checkpoint": None if resume_path is None else str(resume_path),
                    "progress_interval_seconds": cli.progress_interval_seconds,
                    "run_name": run_name,
                    "training_scheme": "MSO",
                    "comparison_scenario_family": cli.scenario_family,
                    "comparison_scenario_value": cli.scenario_value,
                },
                "synchronization": synchronization,
                "best_checkpoint_score": best_checkpoint_score,
                "best_checkpoint_ready": best_checkpoint_ready,
                "checkpoint_reference_load": reference_checkpoint_load,
                "checkpoint_selection_window": best_checkpoint_window,
                "slow_lr_controller_state": slow_lr_controller.state_dict(),
                "completed_updates": target_update,
                "completed_episodes": (
                    resume_episode_offset + run_episodes
                ),
                "completed_rollouts": training_rollout_idx,
                "deterministic_initialization": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if history:
        save_checkpoint(
            agent,
            checkpoint_dir / "last.pt",
            {
                "args": {
                    **base_args,
                    "seed": effective_seed,
                    "physical_seed": physical_seed,
                    "episode_minutes": effective_episode_minutes,
                    "episodes_per_update": cli.episodes_per_update,
                    "sampled_seconds_per_window": effective_sampled_seconds,
                    "updates": target_update,
                    "additional_updates": cli.updates,
                    "load_checkpoint": None if resume_path is None else str(resume_path),
                    "run_name": run_name,
                    "training_scheme": "MSO",
                    "comparison_scenario_family": cli.scenario_family,
                    "comparison_scenario_value": cli.scenario_value,
                },
                "synchronization": synchronization,
                "update": target_update,
                "completed_updates": target_update,
                "completed_rollouts": training_rollout_idx,
                "completed_episodes": resume_episode_offset + run_episodes,
                "deterministic_initialization": True,
                "best_checkpoint_score": best_checkpoint_score,
                "best_checkpoint_ready": best_checkpoint_ready,
                "checkpoint_score": checkpoint_score,
                "checkpoint_score_load_slope": checkpoint_load_slope,
                "checkpoint_reference_load": reference_checkpoint_load,
                "checkpoint_selection_window": best_checkpoint_window,
                "checkpoint_selection_history": checkpoint_selection_history,
                "slow_lr_controller_state": slow_lr_controller.state_dict(),
                "history": history[-1],
                "selection_kind": "last",
            },
            training_random_streams,
        )
    print(f"monolithic checkpoints: {checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
