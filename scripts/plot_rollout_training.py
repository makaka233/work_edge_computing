from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "raw": "#8FB7D9",
    "short": "#1F77B4",
    "long": "#0B3C5D",
    "fast": "#2E8B57",
    "slow": "#8C564B",
    "count": "#9467BD",
    "placement": "#FF7F0E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rollout-resolution dual-PPO training results.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rolling-rollouts", type=int, default=120)
    parser.add_argument("--long-rolling-rollouts", type=int, default=360)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(window // 4, 1)).mean()


def save_single_rollout_plot(
    data: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    output: Path,
    short_window: int,
    long_window: int,
    scale: float = 1.0,
    color: str = COLORS["short"],
) -> None:
    x = data["rollout"]
    y = data[column] * scale
    short = rolling(y, short_window)
    long = rolling(y, long_window)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.plot(x, y, color=COLORS["raw"], alpha=0.18, linewidth=0.65, label="Per rollout")
    ax.plot(x, short, color=color, linewidth=2.0, label=f"Rolling {short_window}")
    ax.plot(x, long, color=COLORS["long"], linewidth=2.4, label=f"Rolling {long_window}")
    finite = np.isfinite(short.to_numpy())
    if finite.any():
        valid_indices = np.flatnonzero(finite)
        best_index = int(valid_indices[np.argmin(short.to_numpy()[finite])]) if scale > 0 else int(valid_indices[0])
        if column in {"avg_latency_s", "p95_latency_s"}:
            ax.scatter(x.iloc[best_index], short.iloc[best_index], color="#D62728", s=45, zorder=5)
            ax.annotate(
                f"best rolling: {short.iloc[best_index]:.1f} @ r{int(x.iloc[best_index])}",
                (x.iloc[best_index], short.iloc[best_index]),
                xytext=(12, 14),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set_xlabel("Global rollout (one 10-minute window)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(ncol=3)
    ax.set_xlim(float(x.min()), float(x.max()))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_policy_stability(data: pd.DataFrame, output: Path) -> None:
    update = data["update"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True, sharex=True)

    axes[0].plot(update, data["fast_entropy"], color=COLORS["fast"], label="Fast")
    axes[0].plot(update, data["slow_count_entropy"], color=COLORS["count"], label="Slow Count")
    axes[0].plot(update, data["slow_placement_entropy"], color=COLORS["placement"], label="Slow Placement")
    axes[0].set_ylabel("Policy entropy")
    axes[0].set_title("Policy exploration")
    axes[0].legend(ncol=3)

    axes[1].plot(update, data["fast_approx_kl"], color=COLORS["fast"], label="Fast")
    axes[1].plot(update, data["slow_count_approx_kl"], color=COLORS["count"], label="Slow Count")
    axes[1].plot(update, data["slow_placement_approx_kl"], color=COLORS["placement"], label="Slow Placement")
    axes[1].axhline(0.015, color="#D62728", linestyle="--", linewidth=1.2, label="KL threshold")
    axes[1].set_yscale("symlog", linthresh=1e-6)
    axes[1].set_ylabel("Approximate KL")
    axes[1].set_title("PPO update magnitude")
    axes[1].legend(ncol=4)

    axes[2].plot(update, data["slow_count_lr"], color=COLORS["count"], label="Count LR")
    axes[2].plot(update, data["slow_placement_lr"], color=COLORS["placement"], label="Placement LR")
    decay = data["slow_lr_decayed"].fillna(0).astype(bool)
    axes[2].scatter(
        update[decay],
        data.loc[decay, "slow_count_lr"],
        color="#D62728",
        marker="v",
        s=55,
        label="LR reduction",
        zorder=5,
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Joint Fast/Slow PPO update")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Slow learning-rate controller")
    axes[2].legend(ncol=3)

    for axis in axes:
        axis.grid(True, alpha=0.22)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_deployment_diagnostics(data: pd.DataFrame, output: Path, window: int) -> None:
    x = data["rollout"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True, sharex=True)
    series = [
        ("cross_node_stage_transition_rate", "Cross-node transition rate", COLORS["placement"]),
        ("slow_count_redundant_replica_fraction", "Redundant replica fraction", COLORS["count"]),
        ("std_node_compute_load", "Node compute-load standard deviation", COLORS["fast"]),
    ]
    for axis, (column, label, color) in zip(axes, series):
        axis.plot(x, data[column], color=COLORS["raw"], alpha=0.14, linewidth=0.6)
        axis.plot(x, rolling(data[column], window), color=color, linewidth=2.2, label=f"Rolling {window}")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.22)
        axis.legend()
    axes[0].set_title("Deployment and load-balance diagnostics")
    axes[-1].set_xlabel("Global rollout (one 10-minute window)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.rolling_rollouts < 1 or args.long_rolling_rollouts < 1:
        raise SystemExit("rolling windows must be positive")
    log_dir = args.run_dir / "logs"
    rollout_path = log_dir / "rollout_metrics.csv"
    training_path = log_dir / "training.csv"
    rollout_data = pd.read_csv(rollout_path)
    training_data = pd.read_csv(training_path)
    output_dir = args.output_dir or args.run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        output_dir / "rollout_mean_latency.png",
        output_dir / "rollout_p95_latency.png",
        output_dir / "fast_reward_rollout.png",
        output_dir / "slow_reward_rollout.png",
        output_dir / "policy_stability_update.png",
        output_dir / "deployment_diagnostics_rollout.png",
    ]
    save_single_rollout_plot(
        rollout_data,
        column="avg_latency_s",
        ylabel="Mean latency (ms)",
        title="Mean latency by rollout",
        output=outputs[0],
        short_window=args.rolling_rollouts,
        long_window=args.long_rolling_rollouts,
        scale=1000.0,
    )
    save_single_rollout_plot(
        rollout_data,
        column="p95_latency_s",
        ylabel="P95 latency (ms)",
        title="P95 latency by rollout",
        output=outputs[1],
        short_window=args.rolling_rollouts,
        long_window=args.long_rolling_rollouts,
        scale=1000.0,
        color="#E67E22",
    )
    save_single_rollout_plot(
        rollout_data,
        column="avg_train_reward",
        ylabel="Fast reward per request",
        title="Fast scheduling reward by rollout",
        output=outputs[2],
        short_window=args.rolling_rollouts,
        long_window=args.long_rolling_rollouts,
        color=COLORS["fast"],
    )
    save_single_rollout_plot(
        rollout_data,
        column="slow_window_return",
        ylabel="Slow return per deployment window",
        title="Slow deployment reward by rollout",
        output=outputs[3],
        short_window=args.rolling_rollouts,
        long_window=args.long_rolling_rollouts,
        color=COLORS["slow"],
    )
    save_policy_stability(training_data, outputs[4])
    save_deployment_diagnostics(rollout_data, outputs[5], args.rolling_rollouts)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
