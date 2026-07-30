from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from edge_drl.agents.hierarchical import build_baseline_agent
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate hierarchical edge DRL scaffold.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-users", type=int, default=12_000)
    parser.add_argument("--num-edge-nodes", type=int, default=48)
    parser.add_argument("--num-service-types", type=int, default=5)
    parser.add_argument("--episode-hours", type=int, default=24)
    parser.add_argument("--max-requests", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EdgeEnvConfig(
        seed=args.seed,
        num_users=args.num_users,
        num_edge_nodes=args.num_edge_nodes,
        num_service_types=args.num_service_types,
        episode_hours=args.episode_hours,
    )
    env = EdgeComputingEnv(config)
    agent = build_baseline_agent()
    obs = env.reset()

    rewards = []
    latencies = []
    start = datetime.now().replace(microsecond=0)
    print("Hierarchical edge scaffold")
    print(f"  users={config.num_users}, nodes={config.num_edge_nodes}, services={config.num_service_types}")
    print("  slow deployment interval=240 minutes, fast scheduling=per request")

    while not env.done and env.metrics["requests"] < args.max_requests:
        action = agent.act(env)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        latencies.append(info["latency_s"])

    requests = max(env.metrics["requests"], 1.0)
    print("Rollout complete")
    print(f"  requests={int(env.metrics['requests'])}")
    print(f"  deployment_updates={int(env.metrics['deployment_updates'])}")
    print(f"  invalid_actions={int(env.metrics['invalid_actions'])}")
    print(f"  avg_latency_s={float(np.mean(latencies)):.4f}")
    print(f"  p95_latency_s={float(np.percentile(latencies, 95)):.4f}")
    print(f"  deadline_violation_rate={env.metrics['deadline_violations'] / requests:.4f}")
    print(f"  avg_reward={float(np.mean(rewards)):.4f}")
    print(f"  elapsed={datetime.now().replace(microsecond=0) - start}")


if __name__ == "__main__":
    main()

