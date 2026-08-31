from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
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
from edge_drl.comparison.dmdr import DMDRScheme
from edge_drl.comparison.io import write_csv, write_json
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.monolithic import MonolithicScheme
from edge_drl.comparison.proposed import ProposedScheme
from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.scenario_transforms import formal_experiment_points, transform_scenario, transform_trace
from edge_drl.comparison.sicp import SICPScheme
from edge_drl.comparison.statistics import paired_differences, summarize_seed_rows
from edge_drl.comparison.trace import generate_comparison_trace
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.env.environment import EdgeComputingEnv


FORMAL_SCHEMES = ("Proposed", "Monolithic", "DMDR", "SICP")


def phase_design(phase: int) -> tuple[int, list[int], list[ExperimentPoint], int]:
    if phase == 1:
        return 10, [2026], [ExperimentPoint("request_load", 1.0, "nominal")], 1
    if phase == 2:
        return 60, [2026, 2027, 2028], formal_experiment_points(), 1
    if phase == 3:
        return 60, list(range(2026, 2046)), formal_experiment_points(), 3
    raise ValueError("phase must be 1, 2, or 3")


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
) -> Path:
    if phase == 3:
        _require_phase2_validation(phase2_validation_run)
    if "Monolithic" in schemes and monolithic_checkpoint is None:
        raise ValueError(
            "Monolithic is trainable and requires --monolithic-checkpoint; "
            "it is never replaced by an optimization baseline"
        )
    if monolithic_checkpoint_mode not in {"fixed", "per-point"}:
        raise ValueError("monolithic_checkpoint_mode must be 'fixed' or 'per-point'")
    if (
        "Monolithic" in schemes
        and monolithic_checkpoint_mode == "fixed"
        and not Path(str(monolithic_checkpoint)).is_file()
    ):
        raise ValueError(
            "formal fixed-checkpoint comparison requires --monolithic-checkpoint "
            "to name one checkpoint file"
        )
    checkpoint_path, checkpoint_args, checkpoint_internal = load_checkpoint_configuration(checkpoint)
    unknown = sorted(set(schemes) - set(FORMAL_SCHEMES))
    if unknown:
        raise ValueError(f"formal comparison does not include schemes: {unknown}")
    episode_minutes, eval_seeds, points, dmdr_repeats = phase_design(phase)
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
        "greedy_in_main_results": False,
        "episode_minutes": episode_minutes,
        "represented_seconds": 1,
        "logical_steps_per_episode": episode_minutes * 60,
        "eval_seeds": eval_seeds,
        "dmdr_routing_repeats": dmdr_repeats,
        "checkpoint": str(checkpoint_path),
        "monolithic_checkpoint": None if monolithic_checkpoint is None else str(monolithic_checkpoint),
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
            "Monolithic": "separately trained dual-scale PPO with one aggregated stage per service",
            "DMDR": "native AES-JDR adaptation from Peng et al., IEEE TSC 2024",
            "SICP": "adapted JPS-CP; TSN gate scheduling omitted",
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
                repeat_count = dmdr_repeats if scheme_name == "DMDR" else 1
                dmdr_plan_cache: dict[int, dict[str, object]] = {}
                for repeat in range(repeat_count):
                    env = TraceReplayEnv(config, scenario, trace)
                    env.reset()
                    try:
                        scheme = _make_scheme(
                            scheme_name,
                            env,
                            checkpoint_path,
                            checkpoint_args,
                            device=device,
                            routing_seed=request_seed + 10_000 * (repeat + 1),
                            phase=phase,
                            dmdr_plan_cache=dmdr_plan_cache,
                            monolithic_checkpoint=monolithic_checkpoint,
                            monolithic_checkpoint_mode=monolithic_checkpoint_mode,
                            point=point,
                        )
                        result = evaluate_scheme_episode(
                            scheme=scheme,
                            env=env,
                            point=point,
                            eval_seed=eval_seed,
                            routing_repeat=repeat,
                        )
                        row = result.to_dict()
                        row.update(
                            {
                                "experiment": point.family,
                                "parameter": point.value,
                                "physical_seed": trace.physical_seed,
                                "demand_seed": trace.demand_seed,
                                "trace_seed": trace.request_seed,
                                "routing_seed": request_seed + 10_000 * (repeat + 1),
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
                                    "scheme": scheme_name,
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
                            "scheme": scheme_name,
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
                            "routing_seed": request_seed + 10_000 * (repeat + 1),
                            "failed": True,
                            "failure_reason": str(error),
                            "traceback": traceback.format_exc(),
                        }
                        failure_rows.append(failure)
                        raw_rows.append(failure)
                    _write_incremental(output, raw_rows, failure_rows, trace_manifest, solver_diagnostics)
                    print(
                        f"[{scheme_name}] {point.label} seed={eval_seed} repeat={repeat} "
                        f"status={'failed' if failure_rows and failure_rows[-1].get('scheme') == scheme_name and failure_rows[-1].get('trace_hash') == trace.trace_hash else 'complete'}"
                    )

    _write_incremental(output, raw_rows, failure_rows, trace_manifest, solver_diagnostics)
    if phase == 2 and not failure_rows and len(raw_rows) == len(points) * len(eval_seeds) * len(schemes):
        write_json(
            output / "diagnostics" / "phase2_validated.json",
            {
                "validated": True,
                "phase": 2,
                "completed_rows": len(raw_rows),
                "expected_rows": len(points) * len(eval_seeds) * len(schemes),
            },
        )
    return output


def _require_phase2_validation(run: str | Path | None) -> None:
    if run is None:
        raise ValueError("Phase 3 requires --phase2-validation-run from a successful complete Phase 2 run")
    marker = Path(run) / "diagnostics" / "phase2_validated.json"
    if not marker.is_file():
        raise ValueError(f"Phase 2 validation marker not found: {marker}")
    with marker.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("validated") is not True or payload.get("phase") != 2:
        raise ValueError(f"invalid Phase 2 validation marker: {marker}")


def _make_scheme(
    name: str,
    env: TraceReplayEnv,
    checkpoint_path: Path,
    checkpoint_args: dict[str, Any],
    *,
    device: str,
    routing_seed: int,
    phase: int,
    dmdr_plan_cache: dict[int, dict[str, object]] | None = None,
    monolithic_checkpoint: str | Path | None = None,
    monolithic_checkpoint_mode: str = "fixed",
    point: ExperimentPoint | None = None,
):
    if name == "Proposed":
        return ProposedScheme(env, checkpoint_path, checkpoint_args, device=device)
    if name == "Monolithic":
        assert monolithic_checkpoint is not None
        assert point is not None
        resolved = _resolve_monolithic_checkpoint(
            monolithic_checkpoint,
            point,
            mode=monolithic_checkpoint_mode,
        )
        return MonolithicScheme(env, resolved, device=device)
    if name == "SICP":
        return SICPScheme(solver_time_limit_s=30.0 if phase == 1 else 120.0)
    if name == "DMDR":
        return DMDRScheme(
            routing_seed=routing_seed,
            solver_max_iterations=120 if phase == 1 else 400,
            plan_cache=dmdr_plan_cache,
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
    write_csv(
        output / "diagnostics" / "dmdr_native.csv",
        [row for row in solver_diagnostics if row.get("scheme") == "DMDR"],
    )
