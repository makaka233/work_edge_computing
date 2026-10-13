"""Profile one isolated window; no writes to the source run or checkpoints."""
from argparse import ArgumentParser, Namespace
import cProfile
import json
from pathlib import Path
import pstats
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_dual_ppo import build_env, rollout
from edge_drl.comparison.checkpoint import build_proposed_agent


def main():
    parser = ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    cli = parser.parse_args()
    torch.set_num_threads(cli.threads)
    config = json.loads((cli.run / "metadata.json").read_text(encoding="utf-8"))["args"]
    config["device"] = cli.device
    config["sampled_seconds_per_window"] = cli.samples
    config["sampling_max_interval_seconds"] = 600 / cli.samples
    args = Namespace(**config)
    env = build_env(args)
    env.reset()
    agent, _ = build_proposed_agent(env, cli.run / "checkpoints/latest.pt", config, device=cli.device)
    profile = cProfile.Profile()
    profile.enable()
    stats = rollout(env, agent, max_requests=0, args=args, rollout_unit="window", reward_scale=10)
    profile.disable()
    pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(28)
    print({k:stats[k] for k in ("request_events", "settlement_steps", "avg_latency_s")})


if __name__ == "__main__":
    main()
