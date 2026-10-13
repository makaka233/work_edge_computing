"""Paired action-replay comparison for KKT versus equal-share allocation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from edge_drl.comparison.checkpoint import edge_config_from_checkpoint, load_checkpoint_configuration
from edge_drl.comparison.io import write_csv, write_json
from edge_drl.comparison.proposed import ProposedScheme
from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.runner import configured_phase_design
from edge_drl.comparison.window_feedback import SLOW_FEEDBACK_PROTOCOL
from edge_drl.comparison.scenario_transforms import transform_scenario, transform_trace
from edge_drl.comparison.statistics import paired_differences, summarize_seed_rows
from edge_drl.comparison.trace import generate_comparison_trace
from edge_drl.env.environment import EdgeComputingEnv


ALLOCATION_MODES = {
    "Proposed": ("kkt", "kkt"),
    "NoKKT-Compute": ("equal-share", "kkt"),
    "NoKKT-Link": ("kkt", "equal-share"),
    "NoKKT-All": ("equal-share", "equal-share"),
}


def _weighted_percentile(values: list[float], weights: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    threshold = percentile / 100.0 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _empty_accumulator() -> dict:
    return {
        "latencies": [], "weights": [], "requests": 0.0, "latency_total": 0.0,
        "compute": 0.0, "link": 0.0, "access": 0.0, "propagation": 0.0,
        "violations": 0.0, "invalid": 0.0, "settlement_steps": 0,
        "settlement_time_s": 0.0,
    }


def _accumulate(target: dict, requests, info: dict) -> None:
    target["settlement_steps"] += 1
    for request, group in zip(requests, info["group_infos"]):
        weight = float(group["request_count"])
        latency = float(group["latency_s"])
        target["latencies"].append(latency)
        target["weights"].append(weight)
        target["requests"] += weight
        target["latency_total"] += latency * weight
        target["compute"] += float(group["compute_delay_s"]) * weight
        target["link"] += float(group["link_delay_s"]) * weight
        target["access"] += float(group["access_delay_s"]) * weight
        target["propagation"] += float(group["propagation_delay_s"]) * weight
        target["violations"] += float(latency > request.deadline_s) * weight
        target["invalid"] += float(not group["valid"]) * weight


def _finalize_row(name: str, accumulator: dict, *, point, eval_seed: int, trace, action_hash: str) -> dict:
    requests = max(float(accumulator["requests"]), 1.0)
    total = float(accumulator["latency_total"])
    return {
        "scheme": name,
        "scenario_family": point.family,
        "scenario_value": float(point.value),
        "scenario_label": point.label,
        "eval_seed": int(eval_seed),
        "routing_repeat": 0,
        "trace_hash": trace.trace_hash,
        "action_trace_hash": action_hash,
        "logical_steps": int(trace.logical_steps),
        "settlement_steps": int(accumulator["settlement_steps"]),
        "request_count": int(accumulator["requests"]),
        "mean_latency_ms": total / requests * 1000.0,
        "p95_latency_ms": _weighted_percentile(
            accumulator["latencies"], accumulator["weights"], 95.0
        ) * 1000.0,
        "episode_total_latency_s": total,
        "mean_slot_total_latency_s": total / max(int(trace.logical_steps), 1),
        "deadline_violation_rate": float(accumulator["violations"]) / requests,
        "invalid_action_rate": float(accumulator["invalid"]) / requests,
        "mean_compute_delay_ms": float(accumulator["compute"]) / requests * 1000.0,
        "mean_link_delay_ms": float(accumulator["link"]) / requests * 1000.0,
        "mean_access_delay_ms": float(accumulator["access"]) / requests * 1000.0,
        "mean_propagation_delay_ms": float(accumulator["propagation"]) / requests * 1000.0,
        "kkt_settlement_time_s": float(accumulator["settlement_time_s"]),
        "failed": bool(accumulator["invalid"] > 0.0),
        "failure_reason": "invalid_action_rate > 0" if accumulator["invalid"] > 0.0 else "",
    }


def run_kkt_action_replay(
    *, checkpoint: str | Path, phase: int, device: str,
    output_root: str | Path = "results/kkt_comparison", run_id: str | None = None,
    eval_seeds_override: tuple[int, ...] | None = None,
    scenario_families: tuple[str, ...] | None = None,
    sampled_seconds_per_window: int = 0,
) -> Path:
    if sampled_seconds_per_window < 0:
        raise ValueError("sampled_seconds_per_window must be >= 0")
    checkpoint_path, checkpoint_args, checkpoint_internal = load_checkpoint_configuration(checkpoint)
    episode_minutes, eval_seeds, points, _, _ = configured_phase_design(
        phase, eval_seeds_override=eval_seeds_override, scenario_families=scenario_families
    )
    selected_id = run_id or f"phase{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = Path(output_root) / selected_id
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    manifests: list[dict] = []
    metadata = {
        "run_id": selected_id, "phase": phase, "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_internal, "checkpoint_args": checkpoint_args,
        "allocation_modes": {
            name: {"compute": compute, "link": link}
            for name, (compute, link) in ALLOCATION_MODES.items()
        },
        "protocol": "frozen Proposed KKT trajectory with identical deployment and Fast actions replayed",
        "slow_window_feedback_protocol": SLOW_FEEDBACK_PROTOCOL,
        "slow_window_feedback_source": "Proposed controller only; other allocations cannot affect frozen actions",
        "sampled_seconds_per_window": int(sampled_seconds_per_window),
        "eval_seeds": eval_seeds,
        "scenario_families": sorted({point.family for point in points}),
    }
    write_json(output / "metadata.json", metadata)

    for point in points:
        for eval_seed in eval_seeds:
            load = point.value if point.family == "request_load" else 1.0
            config = edge_config_from_checkpoint(
                checkpoint_args, episode_minutes=episode_minutes,
                environment_seed=eval_seed + 100_000, demand_seed=eval_seed,
                demand_load_multiplier=load,
            )
            base = EdgeComputingEnv(config)
            base.reset()
            scenario = deepcopy(base.scenario)
            request_seed = eval_seed + 1_000_000 + int(round(point.value * 1000)) + sum(map(ord, point.family))
            trace = generate_comparison_trace(
                scenario=scenario, logical_steps=episode_minutes * 60,
                requests_per_minute=base._arrival_rate_per_minute(),
                physical_seed=int(config.physical_seed if config.physical_seed is not None else config.seed),
                demand_seed=eval_seed, request_seed=request_seed,
                task_compute_scale=config.task_compute_scale,
                task_data_scale=config.task_data_scale,
            )
            scenario = transform_scenario(scenario, point)
            trace = transform_trace(trace, point)
            environments = {
                name: TraceReplayEnv(
                    replace(config, compute_allocation_mode=compute, link_allocation_mode=link),
                    scenario, trace,
                )
                for name, (compute, link) in ALLOCATION_MODES.items()
            }
            for env in environments.values():
                env.reset()
            controller = environments["Proposed"]
            scheme = ProposedScheme(controller, checkpoint_path, checkpoint_args, device=device)
            accumulators = {name: _empty_accumulator() for name in ALLOCATION_MODES}
            action_digest = hashlib.sha256()
            window_seconds = float(config.deployment_interval_minutes * 60)
            represented = (
                window_seconds / float(sampled_seconds_per_window)
                if 0 < sampled_seconds_per_window < window_seconds
                else 1.0
            )
            while not controller.done:
                remaining = max(float(trace.logical_steps) - controller.current_time_minute * 60.0, 0.0)
                step_seconds = min(represented, remaining)
                if step_seconds < 1.0 - 1e-9:
                    break
                deployment_due = controller.needs_deployment_update
                scheme.maybe_plan(controller)
                if deployment_due:
                    deployment = controller.deployment.copy()
                    for name, env in environments.items():
                        if name != "Proposed":
                            env.apply_deployment(deployment)
                    action_digest.update(deployment.tobytes())
                requests = list(controller.current_requests)
                signatures = [(r.request_id, r.request_count, r.service_id) for r in requests]
                for env in environments.values():
                    other = [(r.request_id, r.request_count, r.service_id) for r in env.current_requests]
                    if other != signatures:
                        raise RuntimeError("action replay environments diverged from the immutable trace")
                schedules = scheme.schedule_batch(controller, requests)
                action_digest.update(json.dumps(schedules, separators=(",", ":")).encode("utf-8"))
                for name, env in environments.items():
                    started = time.perf_counter()
                    _, _, _, info = env.step(schedules, represented_seconds=step_seconds)
                    accumulators[name]["settlement_time_s"] += time.perf_counter() - started
                    _accumulate(accumulators[name], requests, info)
                    if name == "Proposed":
                        scheme.observe_step(env, requests, schedules, info)
            scheme.finish_evaluation_episode(controller)
            action_hash = action_digest.hexdigest()
            for name, accumulator in accumulators.items():
                rows.append(_finalize_row(
                    name, accumulator, point=point, eval_seed=eval_seed,
                    trace=trace, action_hash=action_hash,
                ))
            manifests.append({
                "scenario_family": point.family, "scenario_value": point.value,
                "eval_seed": eval_seed, "trace_hash": trace.trace_hash,
                "action_trace_hash": action_hash, "request_count": trace.request_count,
            })
            _write_outputs(output, rows, manifests)
            print(f"[KKT replay] {point.label} seed={eval_seed} complete")
    return output


def _write_outputs(output: Path, rows: list[dict], manifests: list[dict]) -> None:
    for family in sorted({row["scenario_family"] for row in rows}):
        write_csv(output / "raw" / f"{family}.csv", [row for row in rows if row["scenario_family"] == family])
    write_csv(output / "diagnostics" / "trace_manifest.csv", manifests)
    write_csv(output / "summary" / "summary.csv", summarize_seed_rows(rows))
    write_csv(output / "summary" / "paired_differences.csv", paired_differences(rows))
