"""Frozen-policy paired sampling audit and transparent resource-budget report.

No training, no edits to existing experiments, no claim of device profiling.
Run from the project root. A full-second replay can be expensive at full load.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.comparison.checkpoint import build_proposed_agent, edge_config_from_checkpoint, load_checkpoint_configuration
from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.trace import generate_comparison_trace, rehash_trace
from edge_drl.env.calibration import load_budget_document
from edge_drl.env.environment import EdgeComputingEnv
from train_dual_ppo import build_env, parse_args


def resource_report(env):
    assert env.scenario is not None
    services, nodes = env.scenario.services, env.scenario.nodes
    memory = env.service_memory_capacities()
    storage = env.service_storage_capacities()
    service_share = np.mean([u.service_weights for u in env.scenario.users], axis=0)
    rate = env._arrival_rate_per_minute() / 60.0
    lambdas = service_share * rate
    compute_lb = 0.0
    for phase in range(env.config.max_service_stages):
        first = second = 0.0
        for service, arrival in zip(services, lambdas):
            if phase >= len(service.stages):
                continue
            mean = service.stages[phase].compute_gcycles_mean * env.config.task_compute_scale
            sigma = .30
            if service.coherent_sampling:
                expected_c = mean
                expected_sqrt_c = np.sqrt(mean) * np.exp(-sigma ** 2 / 8)
            else:
                expected_c = mean * np.exp(sigma ** 2 / 2)
                expected_sqrt_c = np.sqrt(mean) * np.exp(sigma ** 2 / 8)
            first += arrival * expected_sqrt_c
            second += arrival * expected_c
        compute_lb += (first ** 2 + second) / sum(n.compute_gcycles_per_s for n in nodes) / max(rate, 1e-12)
    expected_input = sum(share * s.input_mb_mean * (1.0 if s.coherent_sampling else np.exp(.35 ** 2 / 2))
                         for s, share in zip(services, service_share)) * env.config.task_data_scale
    access = env.config.radio_rtt_ms / 1000 + expected_input / (env.config.wireless_uplink_mbps / 8)
    tau = env.config.load_ewma_tau_minutes * 60
    noise = [{"hold_seconds": hold,
              "iid_stationary_ewma_variance_ratio_vs_1s": float(np.tanh(hold / (2 * tau)) / np.tanh(1 / (2 * tau)))}
             for hold in (1, 5, 10, 15, 30, 100)]
    return {
        "status": "engineering_budget_audit_not_device_measurement",
        "expected_requests_per_second": rate,
        "independent_poisson_raw_capacity_relaxed_mean_bound_ms": 1000 * (compute_lb + access),
        "bound_ignores": ["deployment", "network", "EWMA capacity reduction", "queueing"],
        "node_nic_mb_s_per_direction": env.node_nic_capacities().tolist(),
        "service_memory_pool_gb": float(memory.sum()),
        "service_storage_pool_gb": float(storage.sum()),
        "sampling_analytic_diagnostic": {"ewma_tau_seconds": tau, "variance_ratios": noise,
            "assumptions": "IID stationary unclipped utilization; not a latency bias bound or dynamic guarantee"},
        "services": [{"name": service.name, "expected_share": float(service_share[service.service_id]),
                      "input_mb": service.input_mb_mean,
                      "stages": [{"compute_gcycle": stage.compute_gcycles_mean,
                                  "memory_gb": stage.memory_gb, "storage_gb": stage.storage_gb,
                                  "resident_memory_gb": stage.resident_memory_gb,
                                  "context_memory_gb": stage.context_memory_gb,
                                  "reserved_contexts": stage.reserved_contexts,
                                  "output_mb": stage.output_mb_mean,
                                  "feasible_nodes": int(np.count_nonzero((memory >= stage.memory_gb) & (storage >= stage.storage_gb)))}
                                 for stage in service.stages]} for service in services],
    }


def simulate_sampling(config, scenario, trace, agent, deployment, samples, sampling_seed, *, audit_seconds=None):
    env = TraceReplayEnv(replace(config), scenario, trace)
    env.reset()
    env.apply_deployment(deployment.copy())
    rng = np.random.default_rng(sampling_seed)
    seconds = trace.logical_steps if audit_seconds is None else int(audit_seconds)
    boundaries = np.linspace(0, seconds, min(samples, seconds) + 1, dtype=int)
    delay = requests = invalid = 0.0
    node_loads = []
    link_loads = []
    for start, stop in zip(boundaries, boundaries[1:]):
        sample = int(start) if stop - start <= 1 else int(rng.integers(start, stop))
        env.current_time_minute = start / 60.0
        env.current_requests = list(trace.slots[sample])
        env.current_request = env.current_requests[0] if env.current_requests else None
        paths = agent.fast_agent.schedule_batch(env, list(env.current_requests), record=False, deterministic=True)
        _, _, _, info = env.step(paths, represented_seconds=float(stop - start))
        for row in info["group_infos"]:
            requests += row["request_count"]
            delay += row["latency_s"] * row["request_count"]
            invalid += float(not row["valid"]) * row["request_count"]
        node_loads.append(env.node_compute_load.copy())
        link_loads.append(env.link_load.copy())
    return {"samples": len(boundaries) - 1, "estimated_requests": requests,
            "mean_latency_ms": 1000 * delay / max(requests, 1),
            "invalid_fraction": invalid / max(requests, 1),
            "final_node_load": node_loads[-1], "final_link_load": link_loads[-1]}


def paired_sampling_audit(config, scenario, agent, *, seconds, samples, repeats, seed):
    if seconds <= 0 or seconds % 60 or repeats <= 0 or any(n <= 0 for n in samples):
        raise ValueError("seconds must be positive whole minutes; samples/repeats must be positive")
    # Keep the original episode clock feature. Padding is not replayed and
    # must not make a 60-minute trained policy see a fake one-minute episode.
    config = replace(config, episode_hours=None)
    if seconds > config.episode_minutes * 60:
        raise ValueError("audit duration exceeds the checkpoint episode horizon")
    base = EdgeComputingEnv(config)
    base.reset()
    base.scenario = scenario
    deployment = agent.slow_agent.plan_deployment(base, deterministic=True, record=False)
    comparisons = []
    for repeat in range(repeats):
        request_seed = seed + repeat
        trace = generate_comparison_trace(scenario=scenario, logical_steps=seconds,
                  requests_per_minute=base._arrival_rate_per_minute(), physical_seed=config.seed,
                  demand_seed=config.seed, request_seed=request_seed,
                  task_compute_scale=config.task_compute_scale, task_data_scale=config.task_data_scale)
        if trace.logical_steps < config.episode_minutes * 60:
            trace = rehash_trace(trace, trace.slots + ((),) * (config.episode_minutes * 60 - trace.logical_steps))
        full = simulate_sampling(config, scenario, trace, agent, deployment, seconds, request_seed, audit_seconds=seconds)
        for count in samples:
            sampled = simulate_sampling(config, scenario, trace, agent, deployment, count, request_seed, audit_seconds=seconds)
            relative_latency = sampled["mean_latency_ms"] / max(full["mean_latency_ms"], 1e-12) - 1
            comparisons.append({"repeat": repeat, "samples": count,
                "full_mean_latency_ms": full["mean_latency_ms"], "sampled_mean_latency_ms": sampled["mean_latency_ms"],
                "relative_latency_error": relative_latency,
                "node_load_rmse": float(np.sqrt(np.mean((sampled["final_node_load"] - full["final_node_load"]) ** 2))),
                "link_load_rmse": float(np.sqrt(np.mean((sampled["final_link_load"] - full["final_link_load"]) ** 2))),
                "request_count_relative_error": sampled["estimated_requests"] / max(full["estimated_requests"], 1) - 1})
        print(f"paired sampling replay {repeat + 1}/{repeats} completed", flush=True)
    summaries = []
    for count in samples:
        rows = [row for row in comparisons if row["samples"] == count]
        mean_abs_error = float(np.mean([abs(row["relative_latency_error"]) for row in rows]))
        load_rmse = float(np.mean([row["node_load_rmse"] for row in rows]))
        summaries.append({"samples": count, "mean_absolute_relative_latency_error": mean_abs_error,
                          "mean_node_load_rmse": load_rmse,
                          "passes_screening_not_proof": mean_abs_error <= .05 and load_rmse <= .05})
    return {"frozen_policy": True, "stationary_trace": True, "seconds": seconds,
            "repeats": repeats, "method": "paired immutable demand trace, uniform temporal strata",
            "warning": "small replay is diagnostic; it does not certify 60-minute or trained-policy fidelity",
            "raw": comparisons, "summary": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replay-seconds", type=int, default=0)
    parser.add_argument("--samples", default="6,12,30,60")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--export-budget", type=Path)
    options, remaining = parser.parse_known_args()
    torch.set_num_threads(1)
    if options.output.exists() or (options.export_budget and options.export_budget.exists()):
        parser.error("output already exists; choose a new filename")
    if options.checkpoint:
        path, checkpoint_args, _ = load_checkpoint_configuration(options.checkpoint)
        config = edge_config_from_checkpoint(checkpoint_args, episode_minutes=60,
                    environment_seed=options.seed, demand_seed=options.seed, demand_load_multiplier=1.0)
        env = EdgeComputingEnv(config)
        env.reset()
        agent, _ = build_proposed_agent(env, path, checkpoint_args, device=options.device)
    else:
        sys.argv = [sys.argv[0], "--pressure-profile", "mec-latency200", "--load-multipliers", "1.0", *remaining]
        args = parse_args()
        env = build_env(args)
        env.reset()
        torch.manual_seed(options.seed)
        agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=env.config.num_edge_nodes,
                    fast_chain_context="chain-kkt-v3", slow_allocation_mode="budget-continuous-v2",
                    training_repair="batch-credit-v1", device=options.device)
    report = {"policy_source": str(options.checkpoint) if options.checkpoint else "untrained_reference",
              "resources": resource_report(env), "sampling": None}
    if options.replay_seconds:
        report["sampling"] = paired_sampling_audit(env.config, env.scenario, agent,
            seconds=options.replay_seconds, samples=[int(part) for part in options.samples.split(",")],
            repeats=options.repeats, seed=options.seed)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if options.export_budget:
        options.export_budget.parent.mkdir(parents=True, exist_ok=True)
        options.export_budget.write_text(json.dumps(load_budget_document(env.config.service_calibration_file),
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {options.output.resolve()}")


if __name__ == "__main__":
    main()
