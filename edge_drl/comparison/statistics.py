from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import t


PRIMARY_METRICS = (
    "mean_latency_ms",
    "p95_latency_ms",
    "episode_total_latency_s",
    "mean_slot_total_latency_s",
    "deadline_violation_rate",
)


def average_routing_repeats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scheme"],
            row["scenario_family"],
            float(row["scenario_value"]),
            int(row["eval_seed"]),
        )
        grouped[key].append(row)
    averaged = []
    for group in grouped.values():
        base = dict(group[0])
        numeric = [key for key, value in base.items() if isinstance(value, (int, float)) and key not in {"eval_seed", "routing_repeat"}]
        for key in numeric:
            base[key] = float(np.mean([float(row[key]) for row in group]))
        base["routing_repeat"] = -1
        base["routing_repeats_averaged"] = len(group)
        averaged.append(base)
    return averaged


def summarize_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed_rows = average_routing_repeats([row for row in rows if not row.get("failed", False)])
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["scheme"], row["scenario_family"], float(row["scenario_value"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (scheme, family, value), group in sorted(grouped.items()):
        for metric in PRIMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            mean, half = _student_interval(values)
            summary.append(
                {
                    "scheme": scheme,
                    "scenario_family": family,
                    "scenario_value": value,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": mean,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                }
            )
    return summary


def paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed_rows = average_routing_repeats([row for row in rows if not row.get("failed", False)])
    lookup = {
        (row["scheme"], row["scenario_family"], float(row["scenario_value"]), int(row["eval_seed"])): row
        for row in seed_rows
    }
    combinations = sorted({(row["scheme"], row["scenario_family"], float(row["scenario_value"])) for row in seed_rows if row["scheme"] != "Proposed"})
    result = []
    for scheme, family, value in combinations:
        seeds = sorted(
            seed
            for candidate_scheme, candidate_family, candidate_value, seed in lookup
            if candidate_scheme == scheme
            and candidate_family == family
            and candidate_value == value
            and ("Proposed", family, value, seed) in lookup
        )
        for metric in PRIMARY_METRICS:
            differences = np.asarray(
                [lookup[scheme, family, value, seed][metric] - lookup["Proposed", family, value, seed][metric] for seed in seeds],
                dtype=np.float64,
            )
            mean, half = _student_interval(differences)
            result.append(
                {
                    "baseline": scheme,
                    "scenario_family": family,
                    "scenario_value": value,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "mean_baseline_minus_proposed": mean,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                }
            )
    return result


def _student_interval(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, float(t.ppf(0.975, values.size - 1) * sem)
