"""Evaluation adapter for the independently trained UnifiedPPO baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from edge_drl.agents.unified import UnifiedPPOAgent
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


class UnifiedPPOScheme(BaseComparisonScheme):
    """One frozen actor-critic for deployment, routing, and resource priority."""

    name = "UnifiedPPO"

    def __init__(
        self,
        env: EdgeComputingEnv,
        checkpoint_path: str | Path,
        *,
        device: str,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        args: dict[str, Any] = dict(checkpoint.get("metadata", {}).get("args", {}))
        self.agent = UnifiedPPOAgent.from_env(
            env,
            hidden_dim=int(args.get("hidden_dim", 256)),
            lr=float(args.get("lr", 1e-4)),
            environment_gamma=float(args.get("environment_gamma", 0.95)),
            k_epochs=int(args.get("k_epochs", 4)),
            entropy_coef=float(args.get("entropy_coef", 0.001)),
            value_coef=float(args.get("value_coef", 0.5)),
            target_kl=(
                None
                if args.get("target_kl", 0.015) is None
                else float(args.get("target_kl", 0.015))
            ),
            minibatch_size=int(args.get("minibatch_size", 2048)),
            reward_scale=float(args.get("reward_scale", 1.0)),
            device=device,
        )
        self.agent.load_state_dict(checkpoint["agent"], load_optimizer=False)
        self.agent.ppo.policy.eval()
        self.deterministic = bool(deterministic)
        self._resource_weights: list[list[float]] = []
        env.config.compute_allocation_mode = "learned-share"
        env.config.link_allocation_mode = "learned-share"

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        deployment = self.agent.plan_deployment(
            env,
            deterministic=self.deterministic,
            record=False,
        )
        env.apply_deployment(deployment)
        self.diagnostics.planning.append(
            {
                "time_minute": float(env.current_time_minute),
                "planner": "unified_ppo_shared_actor",
                "replicas": int(deployment.sum()),
            }
        )

    def schedule_batch(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
    ) -> list[list[int]]:
        schedules, weights = self.agent.schedule_batch(
            env,
            requests,
            deterministic=self.deterministic,
            record=False,
        )
        self._resource_weights = weights
        return schedules

    def resource_weights(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        schedules: list[list[int]],
    ) -> list[list[float]]:
        del env, requests, schedules
        return self._resource_weights
