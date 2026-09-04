from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEME_ORDER = ("Proposed", "Monolithic", "DMDR", "SICP")
COLORS = {
    "Proposed": "#0072B2",
    "Monolithic": "#D55E00",
    "DMDR": "#009E73",
    "SICP": "#CC79A7",
}
LINE_STYLES = {"Proposed": "-", "Monolithic": "--", "DMDR": "-.", "SICP": ":"}
MARKERS = {"Proposed": "o", "Monolithic": "s", "DMDR": "^", "SICP": "D"}
FAMILY_ORDER = (
    "request_load",
    "compute_capacity",
    "wired_bandwidth",
    "intermediate_data",
    "stage_heterogeneity",
)
FAMILY_LABELS = {
    "request_load": "Request load",
    "compute_capacity": "Compute capacity",
    "wired_bandwidth": "Wired bandwidth",
    "intermediate_data": "Intermediate data",
    "stage_heterogeneity": "Stage heterogeneity",
}


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_rows(
    rows: list[dict[str, str]], family: str, metric: str, scheme: str
) -> list[dict[str, str]]:
    return sorted(
        (
            row
            for row in rows
            if row["scenario_family"] == family
            and row["metric"] == metric
            and row["scheme"] == scheme
        ),
        key=lambda row: float(row["scenario_value"]),
    )


def _plot_series(axis, selected: list[dict[str, str]], scheme: str) -> None:
    if not selected:
        return
    x = np.asarray([float(row["scenario_value"]) for row in selected])
    y = np.asarray([float(row["mean"]) for row in selected])
    axis.plot(
        x,
        y,
        marker=MARKERS[scheme],
        markersize=5,
        linewidth=2,
        linestyle=LINE_STYLES[scheme],
        label=scheme,
        color=COLORS[scheme],
    )
    if any(int(row["n_seeds"]) > 1 for row in selected):
        low = np.asarray([float(row["ci95_low"]) for row in selected])
        high = np.asarray([float(row["ci95_high"]) for row in selected])
        axis.fill_between(x, low, high, alpha=0.14, color=COLORS[scheme])


def plot_five_scenario_overview(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: Path,
) -> list[Path]:
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.3), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, family in zip(flat_axes, FAMILY_ORDER):
        for scheme in SCHEME_ORDER:
            _plot_series(axis, _selected_rows(rows, family, metric, scheme), scheme)
        axis.set_title(FAMILY_LABELS[family])
        axis.set_xlabel("Scenario multiplier")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, which="major", alpha=0.28)
        axis.grid(True, which="minor", alpha=0.10)

    legend_axis = flat_axes[-1]
    legend_axis.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=12,
        handlelength=3.0,
    )
    figure.suptitle(title, fontsize=16)
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    figure.savefig(outputs[0], dpi=300)
    figure.savefig(outputs[1])
    plt.close(figure)
    return outputs


def plot_run(run_dir: Path) -> list[Path]:
    rows = load_summary(run_dir / "summary" / "summary.csv")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    outputs = []
    outputs.extend(
        plot_five_scenario_overview(
            rows,
            metric="mean_latency_ms",
            ylabel="Mean latency (ms, log scale)",
            title="Mean latency across five edge-computing scenarios (seed 2026)",
            output_stem=plot_dir / "five_scenario_mean_latency_all_schemes",
        )
    )
    outputs.extend(
        plot_five_scenario_overview(
            rows,
            metric="p95_latency_ms",
            ylabel="P95 latency (ms, log scale)",
            title="P95 latency across five edge-computing scenarios (seed 2026)",
            output_stem=plot_dir / "five_scenario_p95_latency_all_schemes",
        )
    )
    families = sorted({row["scenario_family"] for row in rows})
    for family in families:
        for metric, ylabel, suffix in (
            ("mean_latency_ms", "Mean request latency (ms)", "mean_latency"),
            ("p95_latency_ms", "P95 request latency (ms)", "p95_latency"),
        ):
            figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
            for scheme in SCHEME_ORDER:
                _plot_series(axis, _selected_rows(rows, family, metric, scheme), scheme)
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
