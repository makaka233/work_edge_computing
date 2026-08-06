from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hierarchical PPO training diagnostics.")
    parser.add_argument("log_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rolling-window", type=int, default=5)
    return parser.parse_args()


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    result = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except ValueError:
            value = float("nan")
        result.append(value)
    return np.asarray(result, dtype=np.float64)


def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(data, np.nan)
    finite = np.isfinite(data)
    for index in range(len(data)):
        start = max(index - window + 1, 0)
        chunk = data[start : index + 1]
        valid = chunk[np.isfinite(chunk)]
        if finite[index] and len(valid):
            result[index] = valid.mean()
    return result


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def save_reward_latency(rows: list[dict[str, str]], output: Path, window: int, run_name: str) -> None:
    update = values(rows, "update")
    train_latency = values(rows, "avg_latency_s") * 1000.0
    train_p95 = values(rows, "p95_latency_s") * 1000.0
    train_reward = values(rows, "avg_train_reward")
    slow_return = values(rows, "slow_window_return")
    eval_latency = values(rows, "eval_avg_latency_s") * 1000.0
    eval_std = values(rows, "eval_avg_latency_std") * 1000.0
    eval_p95 = values(rows, "eval_p95_latency_s") * 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(f"Training convergence: {run_name}", fontsize=15)

    train_mask = finite_mask(update, train_latency)
    axes[0].plot(update[train_mask], train_latency[train_mask], color="#4C78A8", alpha=0.38, linewidth=1.1, label="Train mean")
    axes[0].plot(update[train_mask], train_p95[train_mask], color="#F58518", alpha=0.25, linewidth=1.0, label="Train P95")
    axes[0].plot(update, rolling_mean(train_latency, window), color="#1F4E79", linewidth=2.4, label=f"Mean rolling {window}")
    axes[0].plot(update, rolling_mean(train_p95, window), color="#C45A00", linewidth=2.0, label=f"P95 rolling {window}")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Training latency")
    axes[0].legend(ncol=4)

    eval_mask = finite_mask(update, eval_latency, eval_std, eval_p95)
    eval_updates = update[eval_mask]
    eval_means = eval_latency[eval_mask]
    axes[1].errorbar(
        eval_updates,
        eval_means,
        yerr=eval_std[eval_mask],
        color="#E45756",
        marker="o",
        linewidth=2.0,
        capsize=3,
        label="Eval mean +/- seed std",
    )
    axes[1].plot(eval_updates, eval_p95[eval_mask], color="#B279A2", marker="s", linestyle="--", label="Eval P95")
    if len(eval_means):
        best_index = int(np.argmin(eval_means))
        axes[1].scatter(eval_updates[best_index], eval_means[best_index], s=90, color="#2E8B57", zorder=5, label="Best eval")
        axes[1].annotate(
            f"{eval_means[best_index]:.1f} ms @ u{int(eval_updates[best_index])}",
            (eval_updates[best_index], eval_means[best_index]),
            xytext=(-120, 18),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#2E8B57"},
        )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Latency (ms, log scale)")
    axes[1].set_title("Held-out evaluation latency")
    axes[1].legend(ncol=3)

    reward_mask = finite_mask(update, train_reward)
    axes[2].plot(update[reward_mask], train_reward[reward_mask], color="#59A14F", alpha=0.35, linewidth=1.1, label="Fast train reward")
    axes[2].plot(update, rolling_mean(train_reward, window), color="#237A3B", linewidth=2.4, label=f"Fast reward rolling {window}")
    axes[2].set_ylabel("Fast reward")
    axes[2].set_xlabel("PPO update")
    slow_axis = axes[2].twinx()
    slow_mask = finite_mask(update, slow_return)
    slow_axis.plot(update[slow_mask], slow_return[slow_mask], color="#9C755F", alpha=0.32, linewidth=1.0, label="Slow window return")
    slow_axis.plot(update, rolling_mean(slow_return, window), color="#6F4E37", linewidth=2.0, label=f"Slow return rolling {window}")
    slow_axis.set_ylabel("Slow return")
    lines, labels = axes[2].get_legend_handles_labels()
    slow_lines, slow_labels = slow_axis.get_legend_handles_labels()
    axes[2].legend(lines + slow_lines, labels + slow_labels, ncol=4)
    axes[2].set_title("Optimizer rewards")

    for axis in axes:
        axis.grid(True, alpha=0.22)
        axis.set_xlim(left=0)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def save_policy_resource(rows: list[dict[str, str]], output: Path, window: int, run_name: str) -> None:
    update = values(rows, "update")
    replicas = values(rows, "avg_replicas_per_stage")
    used_replicas = values(rows, "used_replica_rate") * 100.0
    avg_compute = values(rows, "avg_node_compute_load") * 100.0
    max_compute = values(rows, "max_node_compute_load") * 100.0
    avg_link = values(rows, "avg_link_load") * 100.0
    max_link = values(rows, "max_link_load") * 100.0
    critic_ev = values(rows, "slow_critic_explained_variance")
    count_entropy = values(rows, "slow_count_entropy")
    fast_kl = values(rows, "fast_approx_kl")
    slow_updated = values(rows, "slow_updated") > 0.5
    critic_ev[~slow_updated] = np.nan
    count_entropy[~slow_updated] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(f"Policy and resource diagnostics: {run_name}", fontsize=15)

    mask = finite_mask(update, replicas)
    axes[0].plot(update[mask], replicas[mask], color="#4C78A8", alpha=0.4, label="Replicas per stage")
    axes[0].plot(update, rolling_mean(replicas, window), color="#1F4E79", linewidth=2.3, label=f"Replicas rolling {window}")
    axes[0].set_ylabel("Average replicas/stage")
    use_axis = axes[0].twinx()
    use_axis.plot(update, rolling_mean(used_replicas, window), color="#E45756", linewidth=2.0, label=f"Used replicas rolling {window}")
    use_axis.set_ylabel("Used replicas (%)")
    lines, labels = axes[0].get_legend_handles_labels()
    use_lines, use_labels = use_axis.get_legend_handles_labels()
    axes[0].legend(lines + use_lines, labels + use_labels, ncol=3)
    axes[0].set_title("Deployment size and scheduler utilization")

    axes[1].plot(update, rolling_mean(avg_compute, window), color="#59A14F", linewidth=2.2, label="Average compute")
    axes[1].plot(update, rolling_mean(max_compute, window), color="#237A3B", linewidth=2.0, label="Maximum compute")
    axes[1].plot(update, rolling_mean(avg_link, window), color="#F2CF5B", linewidth=1.8, label="Average link")
    axes[1].plot(update, rolling_mean(max_link, window), color="#F58518", linewidth=2.0, label="Maximum link")
    axes[1].set_ylabel("Utilization (%)")
    axes[1].set_title("Resource pressure, rolling averages")
    axes[1].legend(ncol=4)

    axes[2].plot(
        update[slow_updated],
        critic_ev[slow_updated],
        color="#B279A2",
        alpha=0.45,
        label="Slow critic explained variance",
    )
    critic_rolling = rolling_mean(critic_ev, window)
    axes[2].plot(
        update[slow_updated],
        critic_rolling[slow_updated],
        color="#7B4F8C",
        linewidth=2.2,
        label=f"Critic EV rolling {window}",
    )
    entropy_axis = axes[2].twinx()
    count_entropy_rolling = rolling_mean(count_entropy, window)
    entropy_axis.plot(
        update[slow_updated],
        count_entropy_rolling[slow_updated],
        color="#9C755F",
        linewidth=2.0,
        label="Slow count entropy",
    )
    entropy_axis.plot(update, rolling_mean(fast_kl, window), color="#E45756", linewidth=1.8, label="Fast approximate KL")
    entropy_axis.axhline(0.03, color="#E45756", linestyle=":", alpha=0.7, label="Fast target KL")
    axes[2].set_xlabel("PPO update")
    axes[2].set_ylabel("Critic explained variance")
    entropy_axis.set_ylabel("Entropy / KL")
    lines, labels = axes[2].get_legend_handles_labels()
    entropy_lines, entropy_labels = entropy_axis.get_legend_handles_labels()
    axes[2].legend(lines + entropy_lines, labels + entropy_labels, ncol=3)
    axes[2].set_title("PPO stability")

    for axis in axes:
        axis.grid(True, alpha=0.22)
        axis.set_xlim(left=0)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with args.log_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("empty training log")
    if args.rolling_window < 1:
        raise SystemExit("--rolling-window must be positive")

    output_dir = args.output_dir or args.log_csv.parent.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.log_csv.parent.parent.name
    reward_latency = output_dir / "reward_latency_trend.png"
    policy_resource = output_dir / "policy_resource_diagnostics.png"
    save_reward_latency(rows, reward_latency, args.rolling_window, run_name)
    save_policy_resource(rows, policy_resource, args.rolling_window, run_name)
    print(reward_latency)
    print(policy_resource)


if __name__ == "__main__":
    main()
