from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.comparison.checkpoint import edge_config_from_checkpoint, load_checkpoint_configuration
from edge_drl.comparison.monolithic import collapse_scenario, collapse_trace
from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.trace import generate_comparison_trace
from train_dual_ppo import rollout, save_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the isolated Monolithic PPO comparison controller")
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help="Proposed run metadata used only as the common physical/training configuration source",
    )
    parser.add_argument("--updates", type=int, default=320)
    parser.add_argument("--episode-minutes", type=int, default=60)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default=None)
    return parser


def _agent_from_config(env: TraceReplayEnv, args: dict[str, object], device: str) -> HierarchicalPPOAgent:
    from inspect import signature

    kwargs: dict[str, object] = {}
    for name in signature(HierarchicalPPOAgent.from_env).parameters:
        if name in {"cls", "env"}:
            continue
        if name in args and args[name] is not None:
            kwargs[name] = args[name]
    kwargs["device"] = device
    kwargs["replicas_per_stage"] = env.config.num_edge_nodes
    # The Proposed metadata calls this field reward_scale; the agent API uses
    # slow_reward_scale. Keep the mapping local to this comparison trainer.
    if "reward_scale" in args:
        kwargs["slow_reward_scale"] = args["reward_scale"]
    return HierarchicalPPOAgent.from_env(env, **kwargs)


def main() -> None:
    cli = build_parser().parse_args()
    if cli.updates <= 0 or cli.episode_minutes <= 0 or cli.episodes_per_update <= 0:
        raise ValueError("updates, episode-minutes, and episodes-per-update must be positive")
    base_path, base_args, _ = load_checkpoint_configuration(cli.base_checkpoint)
    config = edge_config_from_checkpoint(
        base_args,
        episode_minutes=cli.episode_minutes,
        environment_seed=cli.seed + 100_000,
        demand_seed=cli.seed,
        demand_load_multiplier=1.0,
    )
    base_env = TraceReplayEnv(
        config,
        collapse_scenario(_scenario_from_config(config)),
        _empty_trace(config.episode_minutes * 60),
    )
    base_env.reset()
    scenario = deepcopy(base_env.scenario)
    assert scenario is not None
    agent = _agent_from_config(base_env, base_args, cli.device)
    run_name = cli.run_name or f"monolithic_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cli.run_root) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    best_score = float("inf")
    history: list[dict[str, float]] = []

    # Training uses fresh independent request traces. They are not the traces
    # used later by the comparison runner and are never read by other schemes.
    for update in range(1, cli.updates + 1):
        stats = None
        for episode_offset in range(cli.episodes_per_update):
            staged_base_env = TraceReplayEnv(
                config,
                deepcopy(scenario),
                _empty_trace(config.episode_minutes * 60),
            )
            staged_base_env.reset()
            request_trace = generate_comparison_trace(
                scenario=scenario,
                logical_steps=config.episode_minutes * 60,
                requests_per_minute=staged_base_env._arrival_rate_per_minute(),
                physical_seed=int(config.physical_seed or config.seed),
                demand_seed=cli.seed + update * cli.episodes_per_update + episode_offset,
                request_seed=cli.seed + 1_000_000 + update * cli.episodes_per_update + episode_offset,
                task_compute_scale=config.task_compute_scale,
                task_data_scale=config.task_data_scale,
            )
            env = TraceReplayEnv(config, collapse_scenario(scenario), collapse_trace(request_trace))
            env.reset()
            episode_stats = rollout(
                env,
                agent,
                max_requests=0,
                args=argparse.Namespace(
                    reward_mode="latency",
                    compute_hotspot_threshold=0.60,
                    link_hotspot_threshold=0.60,
                    resource_active_load_threshold=0.01,
                    compute_hotspot_coef=0.0,
                    link_hotspot_coef=0.0,
                    compute_imbalance_coef=0.0,
                    link_imbalance_coef=0.0,
                    idle_deployed_node_coef=0.0,
                    fast_congestion_credit_coef=1.0,
                    sampled_seconds_per_window=0,
                ),
                reward_scale=float(base_args.get("reward_scale", 1.0)),
                train_mode="joint",
                rollout_unit="episode",
                progress_label=f"monolithic update {update}/{cli.updates} episode {episode_offset + 1}/{cli.episodes_per_update}",
                progress_interval_seconds=30.0,
            )
            if stats is None:
                stats = episode_stats
            else:
                for key in ("avg_latency_s", "p95_latency_s", "avg_reward"):
                    stats[key] = float(stats.get(key, 0.0) + episode_stats.get(key, 0.0))
        assert stats is not None
        for key in ("avg_latency_s", "p95_latency_s", "avg_reward"):
            stats[key] = float(stats.get(key, 0.0)) / cli.episodes_per_update
        update_metrics = agent.update(progress_label=f"monolithic update {update}")
        row = {
            "update": float(update),
            "avg_latency_s": float(stats.get("avg_latency_s", 0.0)),
            "p95_latency_s": float(stats.get("p95_latency_s", 0.0)),
            "avg_reward": float(stats.get("avg_reward", 0.0)),
            **{f"fast_{key}": float(value) for key, value in update_metrics.get("fast", {}).items() if isinstance(value, (int, float))},
            **{f"slow_{key}": float(value) for key, value in update_metrics.get("slow", {}).items() if isinstance(value, (int, float))},
        }
        history.append(row)
        metadata = {"args": {**base_args, "seed": cli.seed, "episode_minutes": cli.episode_minutes, "run_name": run_name, "training_scheme": "Monolithic"}, "update": update, "history": row}
        save_checkpoint(agent, checkpoint_dir / "latest.pt", metadata)
        score = float(stats.get("avg_latency_s", float("inf")))
        if score < best_score:
            best_score = score
            save_checkpoint(agent, checkpoint_dir / "best.pt", metadata)
        (run_dir / "training_log.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"monolithic update={update} avg_latency_s={score:.6f}")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {"args": {**base_args, "seed": cli.seed, "episode_minutes": cli.episode_minutes, "run_name": run_name, "training_scheme": "Monolithic"}, "best_score_avg_latency_s": best_score},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"monolithic checkpoints: {checkpoint_dir.resolve()}")


def _scenario_from_config(config):
    from edge_drl.env.environment import EdgeComputingEnv

    env = EdgeComputingEnv(config)
    env.reset()
    return env.scenario


def _empty_trace(logical_steps: int):
    from edge_drl.comparison.trace import ComparisonTrace

    return ComparisonTrace(tuple(() for _ in range(logical_steps)), 0, 0, 0, "empty")


if __name__ == "__main__":
    main()
