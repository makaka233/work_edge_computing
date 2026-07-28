from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def summarize(train_log: Path, heuristic_log: Path | None = None, tail: int = 5) -> dict[str, float | int | None]:
    rows = _read_rows(train_log)
    if not rows:
        raise RuntimeError(f"No rows found in {train_log}.")

    train_delays = [_finite_float(row.get("avg_delay", "")) for row in rows]
    train_delays = [value for value in train_delays if value is not None]
    val_rows = [row for row in rows if _finite_float(row.get("val_avg_delay", "")) is not None]
    val_delays = [float(row["val_avg_delay"]) for row in val_rows]
    val_episodes = [int(float(row["episode"])) for row in val_rows]

    result: dict[str, float | int | None] = {
        "episodes": len(rows),
        "train_avg_delay_best": min(train_delays) if train_delays else None,
        "train_avg_delay_last": train_delays[-1] if train_delays else None,
        "train_avg_delay_tail_mean": _mean(train_delays[-tail:]) if train_delays else None,
        "val_points": len(val_rows),
        "val_best_episode": val_episodes[val_delays.index(min(val_delays))] if val_delays else None,
        "val_avg_delay_best": min(val_delays) if val_delays else None,
        "val_avg_delay_last": val_delays[-1] if val_delays else None,
        "val_avg_delay_tail_mean": _mean(val_delays[-tail:]) if val_delays else None,
        "ppo_entropy_last": _finite_float(rows[-1].get("ppo_entropy", "")),
        "ppo_explained_variance_last": _finite_float(rows[-1].get("ppo_explained_variance", "")),
        "ppo_value_loss_last": _finite_float(rows[-1].get("ppo_value_loss", "")),
        "restore_count": sum(1 for row in rows if _finite_float(row.get("restored_best", "")) == 1.0),
        "restored_ppo_lr_last": _finite_float(rows[-1].get("restored_ppo_lr", "")),
    }

    if heuristic_log is not None:
        heuristic_rows = _read_rows(heuristic_log)
        heuristic_delays = [_finite_float(row.get("avg_delay", "")) for row in heuristic_rows]
        heuristic_delays = [value for value in heuristic_delays if value is not None]
        heuristic_delay = _mean(heuristic_delays) if heuristic_delays else None
        result["heuristic_avg_delay"] = heuristic_delay
        if heuristic_delay is not None and result["val_avg_delay_best"] is not None:
            result["best_gap_vs_heuristic"] = float(result["val_avg_delay_best"]) - heuristic_delay
        if heuristic_delay is not None and result["val_avg_delay_last"] is not None:
            result["last_gap_vs_heuristic"] = float(result["val_avg_delay_last"]) - heuristic_delay
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_log", type=Path)
    parser.add_argument("--heuristic-log", type=Path, default=None)
    parser.add_argument("--tail", type=int, default=5)
    args = parser.parse_args()
    result = summarize(args.train_log, args.heuristic_log, args.tail)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
