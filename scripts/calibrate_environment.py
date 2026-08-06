from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_drl.agents.hierarchical import build_baseline_agent
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig


@dataclass(frozen=True)
class CalibrationProfile:
    traffic_scale: float
    task_compute_scale: float
    task_data_scale: float
    node_capacity_scale: float
    link_bandwidth_scale: float


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate MEC pressure with a fixed physical seed and deterministic heuristic policy."
    )
    parser.add_argument("--physical-seed", type=int, default=2026)
    parser.add_argument("--demand-seed", type=int, default=32026)
    parser.add_argument("--num-users", type=int, default=12000)
    parser.add_argument("--num-edge-nodes", type=int, default=32)
    parser.add_argument("--num-service-types", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--traffic-scales", type=parse_float_list, default=(1.0,))
    parser.add_argument("--task-compute-scales", type=parse_float_list, default=(1.0,))
    parser.add_argument("--task-data-scales", type=parse_float_list, default=(1.0,))
    parser.add_argument("--node-capacity-scales", type=parse_float_list, default=(1.0,))
    parser.add_argument("--link-bandwidth-scales", type=parse_float_list, default=(1.0,))
    parser.add_argument("--output", type=Path, default=Path("runs/calibration/environment_pressure.csv"))
    return parser.parse_args()


def weighted_percentile(values: list[float], weights: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values_np = np.asarray(values, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values_np)
    values_np = values_np[order]
    weights_np = weights_np[order]
    cutoff = percentile / 100.0 * float(weights_np.sum())
    index = min(int(np.searchsorted(np.cumsum(weights_np), cutoff, side="left")), len(values_np) - 1)
    return float(values_np[index])


def evaluate_profile(args: argparse.Namespace, profile: CalibrationProfile) -> dict[str, float]:
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=args.demand_seed,
            physical_seed=args.physical_seed,
            scenario_seed=args.demand_seed,
            num_users=args.num_users,
            num_edge_nodes=args.num_edge_nodes,
            num_service_types=args.num_service_types,
            episode_hours=1,
            deployment_interval_minutes=10,
            arrival_profile="stationary",
            traffic_scale=profile.traffic_scale,
            task_compute_scale=profile.task_compute_scale,
            task_data_scale=profile.task_data_scale,
            node_compute_capacity_scale=profile.node_capacity_scale,
            wired_link_bandwidth_scale=profile.link_bandwidth_scale,
        )
    )
    env.reset()
    agent = build_baseline_agent()
    latencies: list[float] = []
    weights: list[float] = []
    node_means: list[float] = []
    node_maxima: list[float] = []
    used_link_means: list[float] = []
    link_maxima: list[float] = []
    deadline_violations = 0.0

    for _ in range(args.seconds):
        requests = list(env.current_requests)
        actions = agent.act_batch(env)
        _, _, _, batch_info = env.step(actions)
        for request, info in zip(requests, batch_info["group_infos"]):
            weight = float(request.request_count)
            latency = float(info["latency_s"])
            latencies.append(latency)
            weights.append(weight)
            if latency > request.deadline_s:
                deadline_violations += weight
        node_means.append(float(np.mean(env.node_compute_load)))
        node_maxima.append(float(np.max(env.node_compute_load)))
        assert env.scenario is not None
        link_mask = env.scenario.adjacency.copy()
        np.fill_diagonal(link_mask, False)
        link_values = np.asarray(env.link_load[link_mask], dtype=np.float64)
        used_links = link_values[link_values > 1e-12]
        used_link_means.append(float(np.mean(used_links)) if used_links.size else 0.0)
        link_maxima.append(float(np.max(link_values)) if link_values.size else 0.0)

    weights_np = np.asarray(weights, dtype=np.float64)
    latencies_np = np.asarray(latencies, dtype=np.float64)
    total_requests = float(weights_np.sum())
    return {
        "traffic_scale": profile.traffic_scale,
        "task_compute_scale": profile.task_compute_scale,
        "task_data_scale": profile.task_data_scale,
        "node_capacity_scale": profile.node_capacity_scale,
        "link_bandwidth_scale": profile.link_bandwidth_scale,
        "requests": total_requests,
        "requests_per_second": total_requests / max(args.seconds, 1),
        "avg_latency_ms": float(np.average(latencies_np, weights=weights_np) * 1000.0),
        "p95_latency_ms": weighted_percentile(latencies, weights, 95.0) * 1000.0,
        "deadline_violation_rate": deadline_violations / max(total_requests, 1.0),
        "avg_node_compute_load": float(np.mean(node_means)),
        "max_node_compute_load": float(np.max(node_maxima)),
        "avg_used_link_load": float(np.mean(used_link_means)),
        "max_link_load": float(np.max(link_maxima)),
    }


def main() -> None:
    args = parse_args()
    profiles = [
        CalibrationProfile(*values)
        for values in product(
            args.traffic_scales,
            args.task_compute_scales,
            args.task_data_scales,
            args.node_capacity_scales,
            args.link_bandwidth_scales,
        )
    ]
    rows: list[dict[str, float]] = []
    for index, profile in enumerate(profiles, start=1):
        row = evaluate_profile(args, profile)
        rows.append(row)
        print(
            "profile={:02d}/{:02d} traffic={:.2f} compute={:.2f} data={:.2f} node_cap={:.2f} "
            "link_bw={:.2f} latency={:.1f}/{:.1f}ms node={:.1%}/{:.1%} used_link={:.2%}/{:.1%} deadline={:.1%}".format(
                index,
                len(profiles),
                profile.traffic_scale,
                profile.task_compute_scale,
                profile.task_data_scale,
                profile.node_capacity_scale,
                profile.link_bandwidth_scale,
                row["avg_latency_ms"],
                row["p95_latency_ms"],
                row["avg_node_compute_load"],
                row["max_node_compute_load"],
                row["avg_used_link_load"],
                row["max_link_load"],
                row["deadline_violation_rate"],
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
