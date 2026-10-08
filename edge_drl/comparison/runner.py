from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import platform
import json
import sys
import traceback
from typing import Any

import numpy as np
import scipy
import torch

from edge_drl.comparison.checkpoint import edge_config_from_checkpoint, load_checkpoint_configuration
from edge_drl.comparison.algorithmic import DPMSScheme, HybridScheme
from edge_drl.comparison.heuristic import (
    DARPScheme,
    FitDeployScheme,
    UniformFirstFitDeployScheme,
    PopularityEFTScheme,
    PopularityNearestScheme,
)
from edge_drl.comparison.classic import HEFTScheme
from edge_drl.comparison.io import write_csv, write_json
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.monolithic import MSOScheme
from edge_drl.comparison.names import LEGACY_SCHEME_IDS, SCHEME_LABELS, canonical_scheme_id, display_scheme_name
from edge_drl.comparison.proposed import ProposedScheme
from edge_drl.comparison.unified import UnifiedPPOScheme
from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.scenario_transforms import formal_experiment_points, transform_scenario, transform_trace
from edge_drl.comparison.statistics import paired_differences, summarize_seed_rows
from edge_drl.comparison.trace import generate_comparison_trace
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.env.environment import EdgeComputingEnv


FORMAL_SCHEMES = ("Proposed", "MSO", "Hybrid", "DPMS")
AVAILABLE_SCHEMES = FORMAL_SCHEMES + (
    "PopularityEFT",
    "PopularityNearest",
    "HEFT",
    "DARP",
    "FitDeploy",
    "UniformFirstFitDeploy",
    "UnifiedPPO",
)
ACCEPTED_SCHEMES = AVAILABLE_SCHEMES + tuple(LEGACY_SCHEME_IDS)
FORMAL_SCENARIO_FAMILIES = (
    "request_load",
    "compute_capacity",
    "wired_bandwidth",
    "intermediate_data",
    "stage_heterogeneity",
)


def phase_design(phase: int) -> tuple[int, list[int], list[ExperimentPoint], int]:
    if phase == 1:
        return 10, [2026], [ExperimentPoint("request_load", 1.0, "nominal")], 1
    if phase == 2:
        return 60, [2026, 2027, 2028], formal_experiment_points(), 1
    if phase == 3:
        return 60, list(range(2026, 2046)), formal_experiment_points(), 1
    raise ValueError("phase must be 1, 2, or 3")


def configured_phase_design(
    phase: int,
    *,
    eval_seeds_override: tuple[int, ...] | None = None,
    scenario_families: tuple[str, ...] | None = None,
) -> tuple[int, list[int], list[ExperimentPoint], int, bool]:
    """Apply explicit evaluation-only filters without changing the formal defaults."""
    episode_minutes, eval_seeds, points, routing_repeats = phase_design(phase)
    is_standard_design = eval_seeds_override is None and scenario_families is None
    if eval_seeds_override is not None:
        eval_seeds = list(dict.fromkeys(int(seed) for seed in eval_seeds_override))
        if not eval_seeds:
            raise ValueError("eval_seeds_override must contain at least one seed")
    if scenario_families is not None:
        selected = tuple(dict.fromkeys(scenario_families))
        unknown = sorted(set(selected) - set(FORMAL_SCENARIO_FAMILIES))
        if unknown:
            raise ValueError(f"unknown scenario families: {unknown}")
        if not selected:
            raise ValueError("scenario_families must contain at least one family")
        points = [point for point in points if point.family in selected]
    return episode_minutes, eval_seeds, points, routing_repeats, is_standard_design


def run_comparison(
    *,
    checkpoint: str | Path,
    phase: int,
    device: str,
    output_root: str | Path = "results/comparison",
    run_id: str | None = None,
    schemes: tuple[str, ...] = FORMAL_SCHEMES,
    phase2_validation_run: str | Path | None = None,
    monolithic_checkpoint: str | Path | None = None,
    monolithic_checkpoint_mode: str = "fixed",
    unified_checkpoint: str | Path | None = None,
    eval_seeds_override: tuple[int, ...] | None = None,
    scenario_families: tuple[str, ...] | None = None,
    sampled_seconds_per_window: int = 0,
    learned_fast_eval_mode: str = "deterministic",
    routing_repeats_override: int | None = None,
) -> Path:
    schemes = tuple(canonical_scheme_id(name) for name in schemes)
    if phase == 3:
        _require_phase2_validation(phase2_validation_run, schemes=schemes)
    if "MSO" in schemes and monolithic_checkpoint is None:
        raise ValueError(
            "MSO is trainable and requires --mso-checkpoint; "
            "it is never replaced by an optimization baseline"
        )
    if "UnifiedPPO" in schemes and unified_checkpoint is None:
        raise ValueError(
            "UnifiedPPO is trainable and requires --unified-checkpoint"
        )
    if "UnifiedPPO" in schemes and not Path(str(unified_checkpoint)).is_file():
        raise ValueError("--unified-checkpoint must name one checkpoint file")
    if monolithic_checkpoint_mode not in {"fixed", "per-point"}:
        raise ValueError("monolithic_checkpoint_mode must be 'fixed' or 'per-point'")
    if sampled_seconds_per_window < 0:
        raise ValueError("sampled_seconds_per_window must be >= 0")
    if learned_fast_eval_mode not in {"deterministic", "stochastic"}:
        raise ValueError("learned_fast_eval_mode must be 'deterministic' or 'stochastic'")
    if routing_repeats_override is not None and routing_repeats_override <= 0:
        raise ValueError("routing_repeats_override must be positive")
    if (
        "MSO" in schemes
        and monolithic_checkpoint_mode == "fixed"
        and not Path(str(monolithic_checkpoint)).is_file()
    ):
        raise ValueError(
            "formal fixed-checkpoint comparison requires --mso-checkpoint "
            "to name one checkpoint file"
        )
    checkpoint_path, checkpoint_args, checkpoint_internal = load_checkpoint_configuration(checkpoint)
    unknown = sorted(set(schemes) - set(AVAILABLE_SCHEMES))
    if unknown:
        raise ValueError(f"formal comparison does not include schemes: {unknown}")
    if len(set(schemes)) != len(schemes):
        raise ValueError("comparison schemes must not be repeated")
    episode_minutes, eval_seeds, points, default_routing_repeats, is_standard_design = configured_phase_design(
        phase,
        eval_seeds_override=eval_seeds_override,
        scenario_families=scenario_families,
    )
    routing_repeats = (
        default_routing_repeats
        if routing_repeats_override is None
        else int(routing_repeats_override)
    )
    is_standard_design = (
        is_standard_design
        and sampled_seconds_per_window == 0
        and tuple(schemes) == FORMAL_SCHEMES
        and learned_fast_eval_mode == "deterministic"
        and routing_repeats_override is None
    )
    deployment_window_seconds = float(
        checkpoint_args.get("deployment_interval_minutes", 10) * 60
    )
    effective_sampled_seconds = (
        deployment_window_seconds
        if sampled_seconds_per_window == 0
        else min(float(sampled_seconds_per_window), deployment_window_seconds)
    )
    represented_seconds = deployment_window_seconds / max(effective_sampled_seconds, 1.0)
    selected_run_id = run_id or f"phase{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = Path(output_root) / selected_run_id
    output.mkdir(parents=True, exist_ok=False)
    diagnostics_dir = output / "diagnostics"
    diagnostics_dir.mkdir()
    raw_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    trace_manifest: list[dict[str, Any]] = []
    solver_diagnostics: list[dict[str, Any]] = []

    metadata = {
        "run_id": selected_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "formal_schemes": list(schemes),
        "scheme_display_names": {name: SCHEME_LABELS.get(name, name) for name in schemes},
        "greedy_in_main_results": False,
        "episode_minutes": episode_minutes,
        "sampled_seconds_per_window": sampled_seconds_per_window,
        "represented_seconds_per_settlement": represented_seconds,
        "learned_fast_eval_mode": learned_fast_eval_mode,
        "routing_repeats": routing_repeats,
        "routing_seed_protocol": "trace_request_seed + 10000 * (routing_repeat + 1)",
        "expected_settlement_steps_per_episode": int(
            round(episode_minutes * 60 / represented_seconds)
        ),
        "temporal_sampling_protocol": (
            "full_second_by_second"
            if sampled_seconds_per_window == 0
            else "training_equivalent_weighted_settlement"
        ),
        "logical_steps_per_episode": episode_minutes * 60,
        "eval_seeds": eval_seeds,
        "scenario_families": list(dict.fromkeys(point.family for point in points)),
        "scenario_point_count": len(points),
        "standard_phase_design": is_standard_design,
        "checkpoint": str(checkpoint_path),
        "monolithic_checkpoint": None if monolithic_checkpoint is None else str(monolithic_checkpoint),
        "unified_checkpoint": None if unified_checkpoint is None else str(unified_checkpoint),
        "learning_checkpoint_protocol": (
            "fixed_across_scenario_sweeps"
            if monolithic_checkpoint_mode == "fixed"
            else "exploratory_per_point"
        ),
        "monolithic_checkpoint_mode": monolithic_checkpoint_mode,
        "checkpoint_metadata": checkpoint_internal,
        "checkpoint_args": checkpoint_args,
        "proposed_offline_training_time_s": None,
        "proposed_offline_training_time_source": "unavailable in legacy checkpoint metadata",
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "labels": {
            "MSO": "separately trained dual-scale PPO with one aggregated stage per service",
            "Hybrid": "frozen Proposed Slow PPO with causal KKT marginal-cost path scheduling",
            "DPMS": "demand-aware binary MILP placement with marginal replicas and the same path scheduling",
            "PopularityEFT": "v1 demand/capacity/locality replica allocation with online work-based stage completion scheduling; project-specific heuristic",
            "PopularityNearest": "frozen popularity/resource placement with static nearest-reachable-replica scheduling",
            "HEFT": "frozen Proposed Slow PPO with HEFT-style upward-rank and earliest-finish Fast scheduling",
            "DARP": "fixed demand/capacity/locality placement with frozen Proposed Fast PPO and unchanged KKT settlement",
            "FitDeploy": "demand-only replica priority and ascending-node First-Fit placement, with coverage feasibility repair, frozen Proposed Fast PPO and unchanged KKT settlement",
            "UniformFirstFitDeploy": "equal-count stage replica allocation and ascending-node First-Fit placement with coverage feasibility repair, frozen Proposed Fast PPO and unchanged KKT settlement",
            "UnifiedPPO": "one shared PPO actor-critic for deployment, routing, and learned resource priorities; no KKT optimization",
        },
    }
    write_json(output / "metadata.json", metadata)

    for point in points:
        for eval_seed in eval_seeds:
            load_multiplier = point.value if point.family == "request_load" else 1.0
            config = edge_config_from_checkpoint(
                checkpoint_args,
                episode_minutes=episode_minutes,
                environment_seed=eval_seed + 100_000,
                demand_seed=eval_seed,
                demand_load_multiplier=load_multiplier,
            )
            base_env = EdgeComputingEnv(config)
            base_env.reset()
            assert base_env.scenario is not None
            base_scenario = deepcopy(base_env.scenario)
            request_seed = eval_seed + 1_000_000 + int(round(point.value * 1000)) + sum(map(ord, point.family))
            trace = generate_comparison_trace(
                scenario=base_scenario,
                logical_steps=episode_minutes * 60,
                requests_per_minute=base_env._arrival_rate_per_minute(),
                physical_seed=int(config.physical_seed if config.physical_seed is not None else config.seed),
                demand_seed=eval_seed,
                request_seed=request_seed,
                task_compute_scale=config.task_compute_scale,
                task_data_scale=config.task_data_scale,
            )
            scenario = transform_scenario(base_scenario, point)
            trace = transform_trace(trace, point)
            trace_manifest.append(
                {
                    "scenario_family": point.family,
                    "scenario_value": point.value,
                    "eval_seed": eval_seed,
                    "physical_seed": trace.physical_seed,
                    "demand_seed": trace.demand_seed,
                    "request_seed": trace.request_seed,
                    "trace_hash": trace.trace_hash,
                    "logical_steps": trace.logical_steps,
                    "request_count": trace.request_count,
                }
            )
            for scheme_name in schemes:
                scheme_display = display_scheme_name(scheme_name)
                for repeat in range(routing_repeats):
                    # Some schemes intentionally use a different resource
                    # settlement rule.  Give every scheme an independent
                    # config so that one adapter cannot contaminate the next
                    # scheme in the same paired comparison.
                    env = TraceReplayEnv(deepcopy(config), scenario, trace)
                    env.reset()
                    routing_seed = request_seed + 10_000 * (repeat + 1)
                    try:
                        scheme = _make_scheme(
                            scheme_name,
                            env,
                            checkpoint_path,
                            checkpoint_args,
                            device=device,
                            monolithic_checkpoint=monolithic_checkpoint,
                            monolithic_checkpoint_mode=monolithic_checkpoint_mode,
                            unified_checkpoint=unified_checkpoint,
                            point=point,
                            learned_fast_deterministic=learned_fast_eval_mode == "deterministic",
                        )
                        # Checkpoint construction consumes RNG state. Reseed only after
                        # loading so each learned scheduler receives the same explicit,
                        # reproducible routing stream for this repeat.
                        np.random.seed(routing_seed % (2**32))
                        torch.manual_seed(routing_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(routing_seed)
                        result = evaluate_scheme_episode(
                            scheme=scheme,
                            env=env,
                            point=point,
                            eval_seed=eval_seed,
                            routing_repeat=repeat,
                            sampled_seconds_per_window=sampled_seconds_per_window,
                        )
                        row = result.to_dict()
                        row.update(
                            {
                                "experiment": point.family,
                                "parameter": point.value,
                                "physical_seed": trace.physical_seed,
                                "demand_seed": trace.demand_seed,
                                "trace_seed": trace.request_seed,
                                "routing_seed": routing_seed,
                                "requests": result.request_count,
                                "mean_latency": result.mean_latency_ms,
                                "p95_latency": result.p95_latency_ms,
                                "planning_time_ms": result.planning_time_s * 1000.0,
                                "scheduling_time_ms": result.scheduling_time_s * 1000.0,
                                "settlement_time_ms": result.kkt_settlement_time_s * 1000.0,
                            }
                        )
                        raw_rows.append(row)
                        for diagnostic in scheme.diagnostics.planning:
                            solver_diagnostics.append(
                                {
                                    "scheme": scheme_display,
                                    "scenario_family": point.family,
                                    "scenario_value": point.value,
                                    "eval_seed": eval_seed,
                                    "routing_repeat": repeat,
                                    **diagnostic,
                                }
                            )
                        if result.failed:
                            failure_rows.append(row)
                    except Exception as error:
                        failure = {
                            "scheme": scheme_display,
                            "scenario_family": point.family,
                            "scenario_value": point.value,
                            "eval_seed": eval_seed,
                            "routing_repeat": repeat,
                            "trace_hash": trace.trace_hash,
                            "experiment": point.family,
                            "parameter": point.value,
                            "physical_seed": trace.physical_seed,
                            "demand_seed": trace.demand_seed,
                            "trace_seed": trace.request_seed,
                            "routing_seed": routing_seed,
                            "failed": True,
                            "failure_reason": str(error),
                            "traceback": traceback.format_exc(),
                        }
                        failure_rows.append(failure)
                        raw_rows.append(failure)
                    _write_incremental(output, raw_rows, failure_rows, trace_manifest, solver_diagnostics)
                    print(
                        f"[{scheme_name}] {point.label} seed={eval_seed} repeat={repeat} "
                        f"status={'failed' if failure_rows and failure_rows[-1].get('scheme') == scheme_display and failure_rows[-1].get('trace_hash') == trace.trace_hash else 'complete'}"
                    )

    _write_incremental(output, raw_rows, failure_rows, trace_manifest, solver_diagnostics)
    if (
        phase == 2
        and is_standard_design
        and not failure_rows
        and len(raw_rows) == len(points) * len(eval_seeds) * len(schemes) * routing_repeats
    ):
        write_json(
            output / "diagnostics" / "phase2_validated.json",
            {
                "validated": True,
                "phase": 2,
                "completed_rows": len(raw_rows),
                "expected_rows": len(points) * len(eval_seeds) * len(schemes) * routing_repeats,
                "schemes": list(schemes),
            },
        )
    return output


def _require_phase2_validation(
    run: str | Path | None, *, schemes: tuple[str, ...] = FORMAL_SCHEMES
) -> None:
    if run is None:
        raise ValueError("Phase 3 requires --phase2-validation-run from a successful complete Phase 2 run")
    marker = Path(run) / "diagnostics" / "phase2_validated.json"
    if not marker.is_file():
        raise ValueError(f"Phase 2 validation marker not found: {marker}")
    with marker.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("validated") is not True
        or payload.get("phase") != 2
        or [canonical_scheme_id(name) for name in payload.get("schemes", [])] != list(schemes)
    ):
        raise ValueError(f"invalid Phase 2 validation marker: {marker}")


def _make_scheme(
    name: str,
    env: TraceReplayEnv,
    checkpoint_path: Path,
    checkpoint_args: dict[str, Any],
    *,
    device: str,
    monolithic_checkpoint: str | Path | None = None,
    monolithic_checkpoint_mode: str = "fixed",
    unified_checkpoint: str | Path | None = None,
    point: ExperimentPoint | None = None,
    learned_fast_deterministic: bool = True,
):
    if name == "Proposed":
        return ProposedScheme(
            env,
            checkpoint_path,
            checkpoint_args,
            device=device,
            fast_deterministic=learned_fast_deterministic,
        )
    if name == "Hybrid":
        return HybridScheme(env, checkpoint_path, checkpoint_args, device=device)
    name = canonical_scheme_id(name)
    if name == "DPMS":
        return DPMSScheme()
    if name == "PopularityEFT":
        return PopularityEFTScheme()
    if name == "PopularityNearest":
        return PopularityNearestScheme()
    if name == "HEFT":
        return HEFTScheme(env, checkpoint_path, checkpoint_args, device=device)
    if name in {"DARP", "FitDeploy", "UniformFirstFitDeploy"}:
        scheme_class = {
            "DARP": DARPScheme,
            "FitDeploy": FitDeployScheme,
            "UniformFirstFitDeploy": UniformFirstFitDeployScheme,
        }[name]
        return scheme_class(
            env,
            checkpoint_path,
            checkpoint_args,
            device=device,
            fast_deterministic=learned_fast_deterministic,
        )
    if name == "MSO":
        assert monolithic_checkpoint is not None
        assert point is not None
        resolved = _resolve_monolithic_checkpoint(
            monolithic_checkpoint,
            point,
            mode=monolithic_checkpoint_mode,
        )
        return MSOScheme(
            env,
            resolved,
            device=device,
            fast_deterministic=learned_fast_deterministic,
        )
    if name == "UnifiedPPO":
        assert unified_checkpoint is not None
        return UnifiedPPOScheme(
            env,
            unified_checkpoint,
            device=device,
            deterministic=learned_fast_deterministic,
        )
    raise ValueError(name)


def _resolve_monolithic_checkpoint(
    path: str | Path,
    point: ExperimentPoint,
    *,
    mode: str = "per-point",
) -> Path:
    if mode == "fixed":
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)
    if mode != "per-point":
        raise ValueError("Monolithic checkpoint mode must be 'fixed' or 'per-point'")
    raw = str(path)
    if "{family}" in raw or "{value}" in raw:
        candidate = Path(raw.format(family=point.family, value=f"{point.value:g}"))
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        stem = f"{point.family}_{point.value:g}"
        options = (
            candidate / f"{stem}.pt",
            candidate / stem / "checkpoints" / "best.pt",
            candidate / f"{stem}_best.pt",
        )
        for option in options:
            if option.is_file():
                return option
        raise FileNotFoundError(f"no Monolithic checkpoint for {stem} under {candidate}")
    raise FileNotFoundError(candidate)


def _write_incremental(output, raw_rows, failure_rows, trace_manifest, solver_diagnostics) -> None:
    families = ("request_load", "compute_capacity", "wired_bandwidth", "intermediate_data", "stage_heterogeneity")
    for family in families:
        write_csv(output / "raw" / f"{family}.csv", [row for row in raw_rows if row.get("scenario_family") == family])
    complete_rows = [row for row in raw_rows if "mean_latency_ms" in row]
    write_csv(output / "summary" / "summary.csv", summarize_seed_rows(complete_rows))
    write_csv(output / "summary" / "paired_differences.csv", paired_differences(complete_rows))
    write_csv(output / "summary" / "failures.csv", failure_rows)
    write_csv(
        output / "summary" / "runtime.csv",
        [
            {key: row.get(key) for key in ("scheme", "scenario_family", "scenario_value", "eval_seed", "routing_repeat", "planning_time_s", "scheduling_time_s", "kkt_settlement_time_s", "total_runtime_s")}
            for row in complete_rows
        ],
    )
    write_csv(output / "diagnostics" / "trace_manifest.csv", trace_manifest)
    write_csv(output / "diagnostics" / "solver.csv", solver_diagnostics)
