from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_drl.comparison.names import canonical_scheme_id

SCHEME_ORDER = ("Proposed", "MSO", "Hybrid", "DPMS", "PopularityEFT", "PopularityNearest", "HEFT", "DARP", "FitDeploy", "UniformFirstFitDeploy")
COLORS = {
    "Proposed": "#0072B2",
    "MSO": "#D55E00",
    "Hybrid": "#009E73",
    "DPMS": "#CC79A7",
    "PopularityEFT": "#8C564B",
    "PopularityNearest": "#7F7F7F",
    "HEFT": "#E69F00",
    "DARP": "#56B4E9",
    "FitDeploy": "#009E73",
    "UniformFirstFitDeploy": "#CC79A7",
}
LINE_STYLES = {"Proposed": "-", "MSO": "--", "Hybrid": "-.", "DPMS": ":"}
MARKERS = {"Proposed": "o", "MSO": "s", "Hybrid": "^", "DPMS": "D"}
LINE_STYLES["PopularityEFT"] = "--"
MARKERS["PopularityEFT"] = "v"
LINE_STYLES["PopularityNearest"] = ":"
MARKERS["PopularityNearest"] = "P"
LINE_STYLES["HEFT"] = "-."
MARKERS["HEFT"] = "X"
LINE_STYLES["DARP"] = "--"
MARKERS["DARP"] = "*"
LINE_STYLES["FitDeploy"] = ":"
MARKERS["FitDeploy"] = "^"
LINE_STYLES["UniformFirstFitDeploy"] = "--"
MARKERS["UniformFirstFitDeploy"] = "D"
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
            and canonical_scheme_id(row["scheme"]) == scheme
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
    linear_scale: bool = False,
) -> list[Path]:
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.3))
    flat_axes = axes.ravel()
    for axis, family in zip(flat_axes, FAMILY_ORDER):
        for scheme in SCHEME_ORDER:
            _plot_series(axis, _selected_rows(rows, family, metric, scheme), scheme)
        axis.set_title(FAMILY_LABELS[family])
        axis.set_xlabel("Scenario multiplier")
        axis.set_ylabel(ylabel)
        if not linear_scale:
            axis.set_yscale("log")
        axis.grid(True, which="major", alpha=0.28)
        axis.grid(True, which="minor", alpha=0.10)

    legend_axis = flat_axes[-1]
    legend_axis.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.5,
    )
    figure.suptitle(title, fontsize=16, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    figure.savefig(outputs[0], dpi=300)
    figure.savefig(outputs[1])
    plt.close(figure)
    return outputs


def plot_run(run_dir: Path, *, mean_only: bool = False, linear_scale: bool = False) -> list[Path]:
    rows = load_summary(run_dir / "summary" / "summary.csv")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    outputs = []
    outputs.extend(
        plot_five_scenario_overview(
            rows,
            metric="mean_latency_ms",
            ylabel="Mean latency (ms)" if linear_scale else "Mean latency (ms, log scale)",
            title="Mean latency across five edge-computing scenarios",
            output_stem=plot_dir / "five_scenario_mean_latency_all_schemes",
            linear_scale=linear_scale,
        )
    )
    if not mean_only:
        outputs.extend(
        plot_five_scenario_overview(
            rows,
            metric="p95_latency_ms",
            ylabel="P95 latency (ms, log scale)",
            title="P95 latency across five edge-computing scenarios",
            output_stem=plot_dir / "five_scenario_p95_latency_all_schemes",
        )
        )
    families = sorted({row["scenario_family"] for row in rows})
    for family in families:
        for metric, ylabel, suffix in (
            ("mean_latency_ms", "Mean request latency (ms)", "mean_latency"),
            ("p95_latency_ms", "P95 request latency (ms)", "p95_latency"),
        ):
            if mean_only and metric != "mean_latency_ms":
                continue
            figure, axis = plt.subplots(figsize=(7.2, 4.6))
            for scheme in SCHEME_ORDER:
                _plot_series(axis, _selected_rows(rows, family, metric, scheme), scheme)
            axis.set_xlabel(family.replace("_", " ").title() + " multiplier")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            axis.legend(
                frameon=False,
                ncol=3,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                fontsize=9,
                handlelength=2.4,
            )
            figure.subplots_adjust(bottom=0.15, left=0.14, right=0.97, top=0.78)
            output = plot_dir / f"{family}_{suffix}.png"
            figure.savefig(output, dpi=220)
            plt.close(figure)
            outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparison results with publication-facing scheme names")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mean-only", action="store_true", help="Only plot mean latency")
    parser.add_argument("--linear-scale", action="store_true", help="Use linear overview axes")
    args = parser.parse_args()
    for path in plot_run(args.run_dir, mean_only=args.mean_only, linear_scale=args.linear_scale):
        print(path)


if __name__ == "__main__":
    main()
