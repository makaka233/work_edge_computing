"""Paired deployment comparison with one frozen Fast and public settlement."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from edge_drl.comparison.checkpoint import (
    CRITICAL_ENV_KEYS, assert_no_critical_conflicts, build_proposed_agent,
    edge_config_from_checkpoint, load_checkpoint_configuration,
)
from edge_drl.comparison.greedy import DemandGreedyPlacement
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.proposed import ProposedScheme
from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.trace import generate_comparison_trace
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.env.environment import EdgeComputingEnv


def policy_digest(fast) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(fast.ppo.policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FrozenFastDARP(BaseComparisonScheme):
    name = "DARP-v2"

    def __init__(self, fast):
        super().__init__()
        self.fast = deepcopy(fast)
        self.deployment_windows = []

    def maybe_plan(self, env):
        if env.needs_deployment_update:
            deployment, details = DemandGreedyPlacement().plan(env)
            env.apply_deployment(deployment)
            self.diagnostics.planning.append(details)
            self.deployment_windows.append(deployment_diagnostics(env))

    def schedule_batch(self, env, requests):
        env.current_requests = requests
        env.current_request = requests[0] if requests else None
        return self.fast.schedule_batch(env, requests, deterministic=True, record=False)


def deployment_diagnostics(env, replica_limit=None):
    """Physical opportunities at termination, not a claimed latency benefit."""
    memory = env.service_memory_capacities().copy()
    storage = env.service_storage_capacities().copy()
    stages = [(s.service_id, t) for s in env.scenario.services for t in s.stages]
    for service_id, stage in stages:
        selected = env.deployment[service_id, stage.stage_id]
        memory -= selected * stage.memory_gb
        storage -= selected * stage.storage_gb
    combinations = available_stages = 0
    limit = env.config.num_edge_nodes if replica_limit is None else replica_limit
    for service_id, stage in stages:
        selected = env.deployment[service_id, stage.stage_id]
        feasible = (~selected & (memory + 1e-9 >= stage.memory_gb)
                    & (storage + 1e-9 >= stage.storage_gb))
        count = min(int(feasible.sum()), max(limit - int(selected.sum()), 0))
        combinations += count
        available_stages += int(count > 0)
    return {"time_minute": float(env.current_time_minute),
        "replicas": int(env.deployment.sum()), "remaining_memory_gb": float(memory.sum()),
        "remaining_storage_gb": float(storage.sum()),
        "individually_feasible_additions": combinations,
        "stages_with_feasible_additions": available_stages}


class FrozenFastSlow(ProposedScheme):
    def maybe_plan(self, env):
        planning = env.needs_deployment_update
        super().maybe_plan(env)
        if planning:
            slow = self.agent.slow_agent
            self.deployment_windows.append(deployment_diagnostics(env, slow.replicas_per_stage))
            self.quota_actions.extend({"time_minute": float(env.current_time_minute), **row}
                                      for row in slow.last_stage_budget_diagnostics)


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_slow_components(*, fast_checkpoint, slow_checkpoints=(), quota_decoders=("checkpoint",),
                             seeds=(2030, 2031, 2032), loads=(.7, 1., 1.35), minutes=60,
                             samples=6, hold_cap=100., device="cpu", output=None):
    """Compare each Slow against DARP v2 with identical Fast/trace/sampling.

    Each scheme evolves its own load state across all deployment windows.
    Missing/failed points remain failures and never become zero-valued scores.
    """
    if (not seeds or not loads or len(set(seeds)) != len(seeds) or len(set(loads)) != len(loads)
            or any(not np.isfinite(x) or x <= 0 for x in loads)
            or minutes <= 0 or samples < 0 or not np.isfinite(hold_cap) or hold_cap < 0):
        raise ValueError("Use unique nonempty seeds/positive loads and valid evaluation budgets")
    if not quota_decoders or len(set(quota_decoders)) != len(quota_decoders):
        raise ValueError("Choose unique quota decoders")
    if set(quota_decoders) - {"checkpoint", "mean", "integer-mode", "categorical-mode", "categorical-mean"}:
        raise ValueError("Unknown quota decoder")
    output = Path(output or Path("results/slow_components") / datetime.now().strftime("paired_%Y%m%d_%H%M%S"))
    # Never overwrite another evaluation.
    output.mkdir(parents=True, exist_ok=False)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return _evaluate(output, fast_checkpoint, slow_checkpoints or (fast_checkpoint,), quota_decoders,
                         seeds, loads, minutes, samples, hold_cap, device)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _evaluate(output, fast_checkpoint, slow_checkpoints, decoders, seeds, loads, minutes, samples, hold_cap, device):
    fast_path, fast_args, fast_metadata = load_checkpoint_configuration(fast_checkpoint)
    cfg = edge_config_from_checkpoint(fast_args, episode_minutes=minutes,
        environment_seed=2_002_029, demand_seed=seeds[0], demand_load_multiplier=loads[0])
    physical = EdgeComputingEnv(cfg)
    physical.reset()
    reference, _ = build_proposed_agent(physical, fast_path, fast_args, device=device)
    fast_hash = policy_digest(reference.fast_agent)
    candidates, descriptions = [], []
    for index, raw_path in enumerate(slow_checkpoints, 1):
        path, saved_args, saved_metadata = load_checkpoint_configuration(raw_path)
        assert_no_critical_conflicts(fast_args, {k: saved_args.get(k) for k in CRITICAL_ENV_KEYS
                                               if k != "episode_minutes"})
        saved_physical = saved_args.get("physical_seed")
        saved_physical = saved_args.get("seed", 2026) if saved_physical is None else saved_physical
        if int(saved_physical) != int(cfg.physical_seed):
            raise ValueError("Slow and Fast checkpoints have different physical seeds")
        candidate, _ = build_proposed_agent(physical, path, saved_args, device=device)
        for decoder in decoders:
            slow = deepcopy(candidate.slow_agent)
            if decoder != "checkpoint":
                allowed = ({"mean", "integer-mode"} if slow.allocation_mode == "stage-budget-v3"
                           else {"categorical-mode", "categorical-mean"} if slow.integer_quotas else {"mean"})
                if decoder not in allowed:
                    raise ValueError(f"{decoder} is incompatible with {slow.allocation_mode}")
                # An explicit evaluation override happens AFTER exact checkpoint loading.
                slow.quota_decode = decoder
            label = f"Slow-{index:02d}-{decoder}"
            candidates.append((label, slow))
            descriptions.append({"label": label, "checkpoint": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "update": saved_metadata.get("update"), "allocation_mode": slow.allocation_mode,
                "deployment_context": slow.deployment_context, "training_seed": saved_args.get("seed"),
                "slow_initialization": saved_args.get("slow_initialization"),
                "fast_initialization": saved_args.get("fast_initialization"),
                "saved_quota_decode": candidate.slow_agent.quota_decode, "evaluated_quota_decode": slow.quota_decode})
    metadata = {"protocol": "slow-component-paired-v1", "fast_checkpoint": str(fast_path),
        "fast_checkpoint_sha256": hashlib.sha256(fast_path.read_bytes()).hexdigest(),
        "fast_update": fast_metadata.get("update"), "fast_policy_sha256": fast_hash,
        "fast_evaluation": "deterministic", "baseline_implementation": "demand-greedy-v2",
        "slow_candidates": descriptions, "seeds": list(seeds), "loads": list(loads),
        "episode_minutes": minutes, "sampled_seconds_per_window": samples,
        "sampling_max_interval_seconds": hold_cap, "sampling_protocol": "training-equivalent-v2",
        "equal_cell_summary": True, "environment": asdict(cfg)}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, services, failures, pairs, manifest = [], [], [], [], []
    deployment_windows, quota_actions = [], []
    for seed in seeds:
        for load in loads:
            cfg = edge_config_from_checkpoint(fast_args, episode_minutes=minutes,
                environment_seed=2_002_029, demand_seed=seed, demand_load_multiplier=load)
            physical = EdgeComputingEnv(cfg)
            physical.reset()
            trace = generate_comparison_trace(scenario=physical.scenario, logical_steps=minutes * 60,
                requests_per_minute=physical._arrival_rate_per_minute(), physical_seed=cfg.physical_seed,
                demand_seed=seed, request_seed=seed + 3_000_017,
                task_compute_scale=cfg.task_compute_scale, task_data_scale=cfg.task_data_scale)
            manifest.append({"seed": seed, "load": load, "trace_hash": trace.trace_hash})
            point_rows = []
            for label, slow in [("DARP-v2", None), *candidates]:
                local_services = []
                try:
                    if slow is None:
                        scheme = FrozenFastDARP(reference.fast_agent)
                        checked_fast = scheme.fast
                    else:
                        scheme = FrozenFastSlow.__new__(FrozenFastSlow)
                        BaseComparisonScheme.__init__(scheme)
                        scheme.name = label
                        scheme.agent = deepcopy(reference)
                        scheme.agent.slow_agent = deepcopy(slow)
                        scheme.fast_deterministic = True
                        scheme.deployment_windows, scheme.quota_actions = [], []
                        scheme._initialize_window_feedback()
                        checked_fast = scheme.agent.fast_agent
                    assert policy_digest(checked_fast) == fast_hash
                    replay = TraceReplayEnv(deepcopy(cfg), deepcopy(physical.scenario), trace)
                    result = evaluate_scheme_episode(scheme=scheme, env=replay,
                        point=ExperimentPoint("request_load", load, "slow_component"), eval_seed=seed,
                        routing_repeat=0, sampled_seconds_per_window=samples,
                        sampling_protocol="training-equivalent-v2", sampling_max_interval_seconds=hold_cap,
                        service_latency_rows=local_services)
                    if result.failed or result.invalid_action_rate:
                        raise RuntimeError(f"Invalid evaluation: {result.failure_reason}")
                    assert policy_digest(checked_fast) == fast_hash, "Frozen Fast weights changed"
                    identifiers = {"seed": seed, "load": load, "candidate": label}
                    deployment_windows.extend({**identifiers, **record} for record in scheme.deployment_windows)
                    quota_actions.extend({**identifiers, **record} for record in getattr(scheme, "quota_actions", []))
                    row = {**asdict(result), "seed": seed, "load": load, "candidate": label,
                           "fast_policy_sha256": fast_hash, "evaluated_quota_decode":
                           "algorithm" if slow is None else slow.quota_decode}
                    services.extend(local_services)
                except Exception as error:
                    row = {"seed": seed, "load": load, "candidate": label, "trace_hash": trace.trace_hash,
                           "failed": True, "failure_reason": f"{type(error).__name__}: {error}"}
                    failures.append(row)
                rows.append(row)
                point_rows.append(row)
                print(f"seed={seed} load={load:g} {label}: "
                      f"{row.get('mean_latency_ms', float('nan')):.3f} ms failed={row['failed']}", flush=True)
            baseline = point_rows[0]
            for row in point_rows[1:]:
                failed = baseline["failed"] or row["failed"]
                if not failed:
                    assert row["trace_hash"] == baseline["trace_hash"]
                    assert row["request_count"] == baseline["request_count"]
                    assert row["settlement_steps"] == baseline["settlement_steps"]
                pairs.append({"seed": seed, "load": load, "candidate": row["candidate"], "failed": failed,
                    "trace_hash": trace.trace_hash,
                    "slow_latency_ms": None if failed else row["mean_latency_ms"],
                    "darp_latency_ms": None if failed else baseline["mean_latency_ms"],
                    "slow_minus_darp_ms": None if failed else row["mean_latency_ms"] - baseline["mean_latency_ms"]})
            _write_csv(output / "raw.csv", rows)
            _write_csv(output / "paired.csv", pairs)
            _write_csv(output / "per_service.csv", services)
            _write_csv(output / "deployment_windows.csv", deployment_windows)
            _write_csv(output / "quota_actions.csv", quota_actions)
            (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "trace_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = []
    for label, _ in candidates:
        subset = [p for p in pairs if p["candidate"] == label]
        complete = bool(subset) and not any(p["failed"] for p in subset)
        slow_mean = float(np.mean([p["slow_latency_ms"] for p in subset])) if complete else None
        darp_mean = float(np.mean([p["darp_latency_ms"] for p in subset])) if complete else None
        summary.append({"candidate": label, "complete": complete, "points": len(subset),
            "slow_mean_latency_ms": slow_mean, "darp_mean_latency_ms": darp_mean,
            "relative_gap_percent": 100 * (slow_mean / darp_mean - 1) if complete else None,
            "wins": sum(not p["failed"] and p["slow_minus_darp_ms"] < 0 for p in subset)})
    report = {"protocol": metadata["protocol"], "failed_evaluations": len(failures), "candidates": summary}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output, report
