from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize staged full-training results.")
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.run_root
    phase0 = root / "phase0_baseline_eval"
    phase1 = root / "phase1_fast_only"
    phase2 = root / "phase2_joint"

    baseline = _read_single_row(phase0 / "evaluation" / "baseline.csv")
    fast = _best_eval(phase1 / "logs" / "training.csv")
    joint = _best_eval(phase2 / "logs" / "training.csv")

    print("Full training summary")
    print(f"  root={root}")
    if baseline:
        print(
            "  baseline avg={:.6f}s p95={:.6f}s invalid={:.0f}".format(
                baseline["avg_latency_s"],
                baseline["p95_latency_s"],
                baseline["invalid_actions"],
            )
        )
    if fast:
        print(
            "  fast_only best_update={} eval_avg={:.6f}s eval_p95={:.6f}s invalid={:.2f}".format(
                int(fast["update"]),
                fast["eval_avg_latency_s"],
                fast["eval_p95_latency_s"],
                fast["eval_invalid_actions"],
            )
        )
    if joint:
        print(
            "  joint best_update={} eval_avg={:.6f}s eval_p95={:.6f}s invalid={:.2f}".format(
                int(joint["update"]),
                joint["eval_avg_latency_s"],
                joint["eval_p95_latency_s"],
                joint["eval_invalid_actions"],
            )
        )
    if baseline and joint:
        improvement = (baseline["avg_latency_s"] - joint["eval_avg_latency_s"]) / max(baseline["avg_latency_s"], 1e-9)
        print(f"  joint_vs_baseline_avg_improvement={improvement:.2%}")
    if fast and joint:
        delta = (fast["eval_avg_latency_s"] - joint["eval_avg_latency_s"]) / max(fast["eval_avg_latency_s"], 1e-9)
        print(f"  joint_vs_fast_best_delta={delta:.2%}")


def _read_single_row(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return _float_row(rows[0]) if rows else None


def _best_eval(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [_float_row(row) for row in csv.DictReader(handle)]
    eval_rows = [row for row in rows if _finite(row.get("eval_avg_latency_s", math.nan))]
    if not eval_rows:
        return None
    return min(eval_rows, key=lambda row: row["eval_avg_latency_s"])


def _float_row(row: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = math.nan
    return out


def _finite(value: float) -> bool:
    return math.isfinite(value)


if __name__ == "__main__":
    main()
