from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dual PPO convergence from a training CSV.")
    parser.add_argument("log_csv", type=Path)
    parser.add_argument("--window", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.log_csv)
    if not rows:
        raise SystemExit("empty log")

    metric = "eval_avg_latency_s" if _has_finite(rows, "eval_avg_latency_s") else "avg_latency_s"
    values = [float(row[metric]) for row in rows if _is_finite(row.get(metric, ""))]
    if not values:
        raise SystemExit(f"no finite {metric} values")

    updates = [float(row["update"]) for row in rows if _is_finite(row.get(metric, ""))]
    first = values[0]
    last = values[-1]
    best = min(values)
    best_idx = values.index(best)
    best_update = updates[best_idx] if best_idx < len(updates) else best_idx
    improvement = (first - best) / max(first, 1e-9)
    tail = values[-args.window :] if len(values) >= args.window else values
    head = values[: len(tail)]
    tail_mean = sum(tail) / len(tail)
    head_mean = sum(head) / len(head)
    tail_improvement = (head_mean - tail_mean) / max(head_mean, 1e-9)
    slope = _linear_slope(updates, values)
    first_half = values[: max(len(values) // 2, 1)]
    second_half = values[max(len(values) // 2, 1) :]
    first_half_mean = sum(first_half) / len(first_half)
    second_half_mean = sum(second_half) / len(second_half) if second_half else values[-1]
    half_improvement = (first_half_mean - second_half_mean) / max(first_half_mean, 1e-9)
    invalid_rates = [
        float(row["invalid_actions"]) / max(float(row["requests"]), 1.0)
        for row in rows
        if _is_finite(row.get("invalid_actions", "")) and _is_finite(row.get("requests", ""))
    ]
    action_changes = [float(row["eval_action_change_rate"]) for row in rows if _is_finite(row.get("eval_action_change_rate", ""))]
    entropies = [float(row["eval_policy_entropy"]) for row in rows if _is_finite(row.get("eval_policy_entropy", ""))]
    margins = [float(row["eval_top1_margin"]) for row in rows if _is_finite(row.get("eval_top1_margin", ""))]
    stochastic = [float(row["eval_stochastic_avg_latency_s"]) for row in rows if _is_finite(row.get("eval_stochastic_avg_latency_s", ""))]

    print("Convergence report")
    print(f"  log={args.log_csv}")
    print(f"  metric={metric}")
    print(f"  updates={len(rows)}")
    print(f"  first={first:.6f}")
    print(f"  last={last:.6f}")
    print(f"  best={best:.6f}")
    print(f"  best_update={best_update:g}")
    print(f"  best_improvement={improvement:.2%}")
    print(f"  first_half_mean={first_half_mean:.6f}")
    print(f"  second_half_mean={second_half_mean:.6f}")
    print(f"  half_improvement={half_improvement:.2%}")
    print(f"  linear_slope={slope:.6f}")
    print(f"  recent_window={len(tail)}")
    print(f"  recent_vs_initial_window={tail_improvement:.2%}")
    if invalid_rates:
        print(f"  avg_invalid_rate={sum(invalid_rates) / len(invalid_rates):.2%}")
    if action_changes:
        print(f"  avg_action_change_rate={sum(action_changes) / len(action_changes):.2%}")
        print(f"  last_action_change_rate={action_changes[-1]:.2%}")
    if entropies:
        print(f"  first_entropy={entropies[0]:.6f}")
        print(f"  last_entropy={entropies[-1]:.6f}")
    if margins:
        print(f"  first_top1_margin={margins[0]:.6f}")
        print(f"  last_top1_margin={margins[-1]:.6f}")
    if stochastic:
        print(f"  last_stochastic_avg_latency={stochastic[-1]:.6f}")
    if len(values) < 5:
        print("  verdict=too_few_points")
    elif slope < -1e-4 and half_improvement > 0.02:
        print("  verdict=training_trend_improving")
    elif improvement > 0.05:
        print("  verdict=best_checkpoint_improved_but_not_monotonic")
    elif abs(slope) <= 1e-4:
        print("  verdict=plateau")
    else:
        print("  verdict=no_clear_training_improvement")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _has_finite(rows: list[dict[str, str]], key: str) -> bool:
    return any(_is_finite(row.get(key, "")) for row in rows)


def _is_finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _linear_slope(updates: list[float], values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.asarray(updates[: len(values)], dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if np.allclose(x, x[0]):
        x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


if __name__ == "__main__":
    main()
