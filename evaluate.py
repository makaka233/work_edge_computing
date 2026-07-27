from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from edge_drl.agents.baselines import HeuristicScheduler, RandomScheduler
from edge_drl.agents.torch_schedulers import TorchAgentSScheduler
from edge_drl.config import load_config
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.training import IntegratedTrainer
from edge_drl.utils import append_csv_row, save_json, timestamped_run_dir


def evaluate_baseline(config: dict, mode: str, episodes: int, seconds: int | None) -> list[dict[str, float]]:
    rows = []
    base_seed = int(config["simulation"]["seed"])
    for ep in range(episodes):
        cfg = dict(config)
        cfg["simulation"] = dict(config["simulation"])
        cfg["simulation"]["seed"] = base_seed + ep
        if seconds is not None:
            cfg["simulation"]["seconds_per_episode"] = int(seconds)
        env = EdgeComputingEnv(cfg)
        if mode == "heuristic":
            scheduler = HeuristicScheduler(env.path_manager, env.compute_capacity, env.bandwidth)
        elif mode == "random":
            scheduler = RandomScheduler(env.rng)
        else:
            raise ValueError(f"Unsupported baseline mode: {mode}")
        rows.append(run_eval_episode(env, scheduler, seconds, use_deployer=False))
        rows[-1]["episode"] = float(ep)
    return rows


def evaluate_neural(
    checkpoint: Path,
    episodes: int,
    seconds: int | None,
    device: str,
    freeze_agent_d: bool,
) -> list[dict[str, float]]:
    payload_config = None
    trainer = None
    rows = []
    base_seed = None
    for ep in range(episodes):
        if trainer is None:
            payload = torch.load(checkpoint, map_location=device)
            payload_config = copy.deepcopy(payload["config"])
            base_seed = int(payload_config["simulation"]["seed"])
            if seconds is not None:
                payload_config["simulation"]["seconds_per_episode"] = int(seconds)
            trainer = IntegratedTrainer(payload_config, device=device)
            trainer.load_checkpoint(checkpoint)
            trainer.scheduler = TorchAgentSScheduler(trainer.agent_s, device=device, deterministic=True)
            trainer.deployer.deterministic = True
        assert payload_config is not None
        assert base_seed is not None
        payload_config["simulation"]["seed"] = base_seed + ep
        trainer.env = EdgeComputingEnv(payload_config)
        trainer.scheduler = TorchAgentSScheduler(trainer.agent_s, device=device, deterministic=True)
        trainer.deployer.deterministic = True
        rows.append(
            run_eval_episode(
                trainer.env,
                trainer.scheduler,
                seconds,
                use_deployer=not freeze_agent_d,
                deployer=trainer.deployer,
            )
        )
        rows[-1]["episode"] = float(ep)
    return rows


def run_eval_episode(env: EdgeComputingEnv, scheduler, seconds: int | None, use_deployer: bool, deployer=None) -> dict[str, float]:
    state = env.reset()
    max_seconds = seconds or int(env.config["simulation"]["seconds_per_episode"])
    deployment_interval = int(env.config["simulation"]["deployment_interval_seconds"])
    rewards = []
    task_counts = []
    delays = []
    invalid = []
    compute_delays = []
    transmission_delays = []
    deployments = 0

    for _ in range(max_seconds):
        deployment_action = None
        if use_deployer and env.time_s % deployment_interval == 0:
            deployment_action = deployer.decide(state)
            deployments += 1
        if hasattr(scheduler, "reset_records"):
            scheduler.reset_records()
        state, reward, done, info = env.step(scheduler, deployment_action=deployment_action)
        rewards.append(float(reward))
        task_counts.append(float(info["num_tasks"]))
        delays.append(float(info["average_delay"]))
        invalid.append(float(info["invalid_schedule"]))
        compute_delays.append(float(info["compute_delay"]))
        transmission_delays.append(float(info["transmission_delay"]))
        if done:
            break

    return {
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "tasks_total": float(np.sum(task_counts)),
        "tasks_per_second": float(np.mean(task_counts)) if task_counts else 0.0,
        "avg_delay": float(np.mean(delays)) if delays else 0.0,
        "compute_delay_mean": float(np.mean(compute_delays)) if compute_delays else 0.0,
        "transmission_delay_mean": float(np.mean(transmission_delays)) if transmission_delays else 0.0,
        "invalid_total": float(np.sum(invalid)),
        "deployments": float(deployments),
        "seconds": float(len(rewards)),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "episode"]
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["heuristic", "random", "neural"], default="heuristic")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze-agent-d", action="store_true")
    parser.add_argument("--run-root", default="runs_eval")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_root) / args.run_name if args.run_name else timestamped_run_dir(args.run_root, args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "neural":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --mode neural")
        rows = evaluate_neural(Path(args.checkpoint), args.episodes, args.seconds, args.device, args.freeze_agent_d)
    else:
        config = load_config(args.config)
        rows = evaluate_baseline(config, args.mode, args.episodes, args.seconds)

    for row in rows:
        append_csv_row(run_dir / "eval_log.csv", row)
        print(
            "episode={episode:.0f} reward_mean={reward_mean:.4f} tasks={tasks_total:.0f} "
            "avg_delay={avg_delay:.4f} invalid={invalid_total:.0f}".format(**row)
        )
    summary = summarize(rows)
    save_json(run_dir / "summary.json", {"rows": rows, "summary": summary})
    print(f"summary={summary}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
