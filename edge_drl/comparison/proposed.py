from __future__ import annotations

from pathlib import Path

from edge_drl.comparison.checkpoint import build_proposed_agent
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


class ProposedScheme(BaseComparisonScheme):
    name = "Proposed"

    def __init__(
        self,
        env: EdgeComputingEnv,
        checkpoint_path: str | Path,
        checkpoint_args: dict[str, object],
        *,
        device: str,
    ) -> None:
        super().__init__()
        self.agent, self.checkpoint_metadata = build_proposed_agent(
            env, checkpoint_path, checkpoint_args, device=device
        )

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        self.agent.maybe_update_deployment(env, deterministic=True, record=False)

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        if requests is not env.current_requests:
            env.current_requests = requests
            env.current_request = requests[0] if requests else None
        return self.agent.fast_agent.schedule_batch(
            env, requests, deterministic=True, record=False
        )
