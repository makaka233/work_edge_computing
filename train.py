from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from edge_drl.agents.baselines import HeuristicScheduler, RandomScheduler
from edge_drl.agents.torch_schedulers import TorchAgentSScheduler
from edge_drl.config import load_config
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.training import IntegratedTrainer
from edge_drl.utils import append_csv_row, save_json, timestamped_run_dir


def build_scheduler(mode: str, env: EdgeComputingEnv):
    if mode == "random":
        return RandomScheduler(env.rng)
    if mode == "heuristic":
        return HeuristicScheduler(env.path_manager, env.compute_capacity, env.bandwidth)
    raise ValueError(f"Unknown mode: {mode}")


def run_episode(env: EdgeComputingEnv, scheduler, seconds: int | None = None) -> dict[str, float]:
    obs = env.reset()
    del obs
    max_seconds = seconds or int(env.config["simulation"]["seconds_per_episode"])
    rewards = []
    task_counts = []
    delays = []
    invalid = []
    user_counts = []
    for _ in range(max_seconds):
        _, reward, done, info = env.step(scheduler)
        rewards.append(float(reward))
        task_counts.append(float(info["num_tasks"]))
        delays.append(float(info["average_delay"]))
        invalid.append(float(info["invalid_schedule"]))
        user_counts.append(float(info["user_count"]))
        if done:
            break
    return {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "tasks_total": float(np.sum(task_counts)),
        "tasks_per_second": float(np.mean(task_counts)) if task_counts else 0.0,
        "avg_delay": float(np.mean(delays)) if delays else 0.0,
        "invalid_total": float(np.sum(invalid)),
        "users_mean": float(np.mean(user_counts)) if user_counts else 0.0,
    }


def validate_trainer(
    trainer: IntegratedTrainer,
    config: dict,
    seconds: int,
    freeze_agent_d: bool = True,
    seed: int = 10007,
) -> dict[str, float]:
    eval_config = {
        **config,
        "simulation": {
            **config["simulation"],
            "seed": int(seed),
            "seconds_per_episode": int(seconds),
        },
    }
    env = EdgeComputingEnv(eval_config)
    scheduler = TorchAgentSScheduler(trainer.agent_s, device=str(trainer.device), deterministic=True)
    state = env.reset()
    rewards = []
    delays = []
    invalid = []
    tasks = []
    deployment_interval = int(eval_config["simulation"]["deployment_interval_seconds"])
    for _ in range(seconds):
        deployment_action = None
        if not freeze_agent_d and env.time_s % deployment_interval == 0:
            trainer.deployer.deterministic = True
            deployment_action = trainer.deployer.decide(state)
        scheduler.reset_records()
        state, reward, done, info = env.step(scheduler, deployment_action=deployment_action)
        rewards.append(float(reward))
        delays.append(float(info["average_delay"]))
        invalid.append(float(info["invalid_schedule"]))
        tasks.append(float(info["num_tasks"]))
        if done:
            break
    return {
        "val_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "val_avg_delay": float(np.mean(delays)) if delays else 0.0,
        "val_invalid_total": float(np.sum(invalid)),
        "val_tasks_total": float(np.sum(tasks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=None)
    parser.add_argument("--mode", choices=["heuristic", "random", "neural"], default="heuristic")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--bc-seconds", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-max-samples", type=int, default=None)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--agent-d-warmup-episodes", type=int, default=0)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--val-seconds", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-freeze-agent-d", action="store_true")
    parser.add_argument("--val-seed", type=int, default=10007)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.seed is not None:
        config["simulation"]["seed"] = int(args.seed)
    if args.seconds is not None:
        config["simulation"]["seconds_per_episode"] = int(args.seconds)
    seed = int(config["simulation"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    run_dir = Path(args.run_root) / args.run_name if args.run_name else timestamped_run_dir(args.run_root, args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    if args.mode == "neural":
        trainer = IntegratedTrainer(
            config,
            device=device,
            ppo_lr=args.ppo_lr,
        )
        bc_stats = None
        if args.bc_seconds > 0:
            bc_stats = trainer.pretrain_agent_s(
                seconds=args.bc_seconds,
                epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
                max_samples=args.bc_max_samples,
            )
            append_csv_row(run_dir / "bc_log.csv", bc_stats)
            trainer.save_checkpoint(run_dir / "checkpoints" / "bc_pretrained.pt", episode=None, extra=bc_stats)
            print(
                "bc_pretrain samples={samples:.0f} loss={loss:.4f} accuracy={acc:.4f}".format(
                    samples=bc_stats["bc_samples"],
                    loss=bc_stats["bc_loss"],
                    acc=bc_stats["bc_accuracy"],
                )
            )
        best_reward = -float("inf")
        history = []
        base_seed = int(config["simulation"]["seed"])
        for _ in range(args.episodes):
            trainer.env.config["simulation"]["seed"] = base_seed + len(history)
            warmup = 1 if len(history) < args.agent_d_warmup_episodes else 0
            episode_history = trainer.train(1, args.seconds, agent_d_warmup_episodes=warmup)
            item = episode_history[0]
            item["episode"] = float(len(history))
            did_validate = args.val_seconds > 0 and len(history) % max(args.val_every, 1) == 0
            if did_validate:
                item.update(
                    validate_trainer(
                        trainer,
                        config,
                        args.val_seconds,
                        freeze_agent_d=args.val_freeze_agent_d,
                        seed=args.val_seed,
                    )
                )
            elif args.val_seconds > 0:
                item.update(
                    {
                        "val_reward_mean": float("nan"),
                        "val_avg_delay": float("nan"),
                        "val_invalid_total": float("nan"),
                        "val_tasks_total": float("nan"),
                    }
                )
            history.append(item)
            append_csv_row(run_dir / "train_log.csv", item)
            if int(item["episode"]) % max(args.save_every, 1) == 0:
                trainer.save_checkpoint(
                    run_dir / "checkpoints" / "latest.pt",
                    episode=int(item["episode"]),
                    extra={"train": item, "bc": bc_stats},
                )
            score = item["val_reward_mean"] if did_validate else (item["reward"] if args.val_seconds <= 0 else None)
            if score is not None and score > best_reward:
                best_reward = score
                trainer.save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    episode=int(item["episode"]),
                    extra={"train": item, "bc": bc_stats},
                )
        save_json(run_dir / "summary.json", {"history": history, "best_reward": best_reward, "bc": bc_stats})
        for item in history:
            print(
                "episode={ep:.0f} reward={reward:.4f} tasks={tasks:.0f} invalid={invalid:.0f} "
                "avg_delay={avg_delay:.4f} deployments={deployments:.0f} agent_d_enabled={agent_d_enabled:.0f} "
                "agent_d_loss={agent_d_loss:.4f} "
                "world_replay={replay:.0f} entropy={entropy:.4f} kl={kl:.6f} "
                "explained_var={explained_var:.4f}".format(
                    ep=item["episode"],
                    reward=item["reward"],
                    tasks=item["tasks"],
                    invalid=item["invalid"],
                    avg_delay=item["avg_delay"],
                    deployments=item["deployments"],
                    agent_d_enabled=item["agent_d_enabled"],
                    agent_d_loss=item["agent_d_loss"],
                    replay=item["world_replay"],
                    entropy=item["ppo_entropy"],
                    kl=item["ppo_approx_kl"],
                    explained_var=item["ppo_explained_variance"],
                )
            )
        print(f"run_dir={run_dir}")
        return

    all_stats = []
    for ep in range(args.episodes):
        config["simulation"]["seed"] = int(config["simulation"]["seed"]) + ep
        env = EdgeComputingEnv(config)
        scheduler = build_scheduler(args.mode, env)
        stats = run_episode(env, scheduler, args.seconds)
        stats["episode"] = float(ep)
        all_stats.append(stats)
        append_csv_row(run_dir / "train_log.csv", stats)
        print(
            "episode={ep} reward_mean={reward:.4f} tasks={tasks:.0f} "
            "tasks_per_second={tps:.2f} avg_delay={delay:.4f} invalid={invalid:.0f} users_mean={users:.1f}".format(
                ep=ep,
                reward=stats["reward_mean"],
                tasks=stats["tasks_total"],
                tps=stats["tasks_per_second"],
                delay=stats["avg_delay"],
                invalid=stats["invalid_total"],
                users=stats["users_mean"],
            )
        )
    save_json(run_dir / "summary.json", {"history": all_stats})
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
