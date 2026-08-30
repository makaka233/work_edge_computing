from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEME_ORDER = ("Proposed", "Monolithic", "DMDR", "SICP")
COLORS = {
    "Proposed": "#1f77b4",
    "Monolithic": "#ff7f0e",
    "DMDR": "#2ca02c",
    "SICP": "#d62728",
}


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_run(run_dir: Path) -> list[Path]:
    rows = load_summary(run_dir / "summary" / "summary.csv")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    outputs = []
    families = sorted({row["scenario_family"] for row in rows})
    for family in families:
        for metric, ylabel, suffix in (
            ("mean_latency_ms", "Mean request latency (ms)", "mean_latency"),
            ("p95_latency_ms", "P95 request latency (ms)", "p95_latency"),
        ):
            figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
            for scheme in SCHEME_ORDER:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["scenario_family"] == family
                        and row["metric"] == metric
                        and row["scheme"] == scheme
                    ),
                    key=lambda row: float(row["scenario_value"]),
                )
                if not selected:
                    continue
                x = np.asarray([float(row["scenario_value"]) for row in selected])
                y = np.asarray([float(row["mean"]) for row in selected])
                low = np.asarray([float(row["ci95_low"]) for row in selected])
                high = np.asarray([float(row["ci95_high"]) for row in selected])
                axis.plot(x, y, marker="o", linewidth=2, label=scheme, color=COLORS[scheme])
                axis.fill_between(x, low, high, alpha=0.16, color=COLORS[scheme])
            axis.set_xlabel(family.replace("_", " ").title() + " multiplier")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, ncol=2)
            output = plot_dir / f"{family}_{suffix}.png"
            figure.savefig(output, dpi=220)
            plt.close(figure)
            outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot formal four-scheme comparison results")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for path in plot_run(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()
