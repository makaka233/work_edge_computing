from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCHEME_ORDER = ("Proposed", "Monolithic")
COLORS = {"Proposed": "#0072B2", "Monolithic": "#D55E00"}
LINE_STYLES = {"Proposed": "-", "Monolithic": "--"}
FAMILY_ORDER = (
    "request_load",
    "compute_capacity",
    "wired_bandwidth",
    "intermediate_data",
    "stage_heterogeneity",
)
FAMILY_LABELS = {
    "request_load": ("Request load", "Request-load multiplier"),
    "compute_capacity": ("Compute capacity", "Compute-capacity multiplier"),
    "wired_bandwidth": ("Wired bandwidth", "Bandwidth multiplier"),
    "intermediate_data": ("Intermediate data", "Intermediate-data multiplier"),
    "stage_heterogeneity": ("Stage heterogeneity", "Heterogeneity level"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Proposed/Monolithic five-scenario evaluation and training curves."
    )
    parser.add_argument("--comparison-run", type=Path, required=True)
    parser.add_argument("--proposed-run", type=Path, required=True)
    parser.add_argument("--monolithic-run", type=Path, required=True)
    parser.add_argument("--rolling-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    figure.savefig(outputs[0], dpi=300, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    plt.close(figure)
    return outputs


def _load_comparison_rows(run_dir: Path) -> pd.DataFrame:
    raw_dir = run_dir / "raw"
    files = sorted(raw_dir.glob("*.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size > 0]
    if not frames:
        raise ValueError(f"no comparison rows found under {raw_dir}")
    data = pd.concat(frames, ignore_index=True)
    if bool(data.get("failed", pd.Series(False, index=data.index)).astype(bool).any()):
        raise ValueError("comparison results contain failed rows")
    expected = {
        (scheme, family)
        for scheme in SCHEME_ORDER
        for family in FAMILY_ORDER
    }
    observed = set(zip(data["scheme"], data["scenario_family"]))
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"comparison results are incomplete: {missing}")
    return data


def plot_five_scenario_metric(
    data: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: Path,
) -> list[Path]:
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
    axes_flat = list(axes.flat)
    for axis, family in zip(axes_flat, FAMILY_ORDER):
        family_title, xlabel = FAMILY_LABELS[family]
        selected_family = data[data["scenario_family"] == family]
        for scheme in SCHEME_ORDER:
            selected = selected_family[selected_family["scheme"] == scheme].sort_values(
                "scenario_value"
            )
            axis.plot(
                selected["scenario_value"],
                selected[metric],
                color=COLORS[scheme],
                linestyle=LINE_STYLES[scheme],
                marker="o" if scheme == "Proposed" else "s",
                markersize=5.2,
                linewidth=2.1,
                label=scheme,
            )
        axis.set_title(family_title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=12)
    figure.suptitle(title, fontsize=15)
    return _save_figure(figure, output_stem)


def _load_episode_log(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "logs" / "episode_metrics.csv"
    data = pd.read_csv(path).sort_values("episode").reset_index(drop=True)
    required = {
        "episode",
        "avg_train_reward",
        "slow_window_return",
        "avg_latency_s",
        "p95_latency_s",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return data


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(window // 4, 1)).mean()


def _plot_training_series(
    axis: plt.Axes,
    logs: dict[str, pd.DataFrame],
    *,
    column: str,
    scale: float,
    ylabel: str,
    title: str,
    rolling_episodes: int,
) -> None:
    for scheme in SCHEME_ORDER:
        data = logs[scheme]
        x = data["episode"]
        y = data[column] * scale
        axis.plot(
            x,
            y,
            color=COLORS[scheme],
            alpha=0.12,
            linewidth=0.65,
        )
        axis.plot(
            x,
            _rolling(y, rolling_episodes),
            color=COLORS[scheme],
            linestyle=LINE_STYLES[scheme],
            linewidth=2.2,
            label=f"{scheme} (rolling {rolling_episodes})",
        )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)


def plot_training_rewards(
    logs: dict[str, pd.DataFrame], output_stem: Path, rolling_episodes: int
) -> list[Path]:
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.6), sharex=True, constrained_layout=True)
    _plot_training_series(
        axes[0],
        logs,
        column="avg_train_reward",
        scale=1.0,
        ylabel="Fast reward per request",
        title="Fast scheduling reward",
        rolling_episodes=rolling_episodes,
    )
    _plot_training_series(
        axes[1],
        logs,
        column="slow_window_return",
        scale=1.0,
        ylabel="Slow return per window",
        title="Slow deployment reward",
        rolling_episodes=rolling_episodes,
    )
    axes[1].set_xlabel("Training episode (60 minutes each)")
    figure.suptitle("Proposed and Monolithic training rewards", fontsize=15)
    return _save_figure(figure, output_stem)


def plot_training_latency(
    logs: dict[str, pd.DataFrame], output_stem: Path, rolling_episodes: int
) -> list[Path]:
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.6), sharex=True, constrained_layout=True)
    _plot_training_series(
        axes[0],
        logs,
        column="avg_latency_s",
        scale=1000.0,
        ylabel="Mean latency (ms)",
        title="Mean request latency during training",
        rolling_episodes=rolling_episodes,
    )
    _plot_training_series(
        axes[1],
        logs,
        column="p95_latency_s",
        scale=1000.0,
        ylabel="P95 latency (ms)",
        title="Tail latency during training",
        rolling_episodes=rolling_episodes,
    )
    axes[1].set_xlabel("Training episode (60 minutes each)")
    figure.suptitle("Proposed and Monolithic training latency", fontsize=15)
    return _save_figure(figure, output_stem)


def main() -> None:
    args = parse_args()
    if args.rolling_episodes < 1:
        raise SystemExit("--rolling-episodes must be positive")
    output_dir = args.output_dir or args.comparison_run / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = _load_comparison_rows(args.comparison_run)
    logs = {
        "Proposed": _load_episode_log(args.proposed_run),
        "Monolithic": _load_episode_log(args.monolithic_run),
    }
    outputs: list[Path] = []
    outputs.extend(
        plot_five_scenario_metric(
            comparison,
            metric="mean_latency_ms",
            ylabel="Mean latency (ms)",
            title="Mean latency across five edge-computing scenarios (seed 2026)",
            output_stem=output_dir / "five_scenario_mean_latency",
        )
    )
    outputs.extend(
        plot_five_scenario_metric(
            comparison,
            metric="p95_latency_ms",
            ylabel="P95 latency (ms)",
            title="P95 latency across five edge-computing scenarios (seed 2026)",
            output_stem=output_dir / "five_scenario_p95_latency",
        )
    )
    outputs.extend(
        plot_training_rewards(
            logs,
            output_dir / "training_rewards_proposed_vs_monolithic",
            args.rolling_episodes,
        )
    )
    outputs.extend(
        plot_training_latency(
            logs,
            output_dir / "training_latency_proposed_vs_monolithic",
            args.rolling_episodes,
        )
    )
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
