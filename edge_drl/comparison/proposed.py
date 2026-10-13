from __future__ import annotations

from pathlib import Path

from edge_drl.comparison.checkpoint import build_proposed_agent
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.window_feedback import SlowWindowFeedbackMixin
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


class ProposedScheme(SlowWindowFeedbackMixin, BaseComparisonScheme):
    name = "Proposed"

    def __init__(
        self,
        env: EdgeComputingEnv,
        checkpoint_path: str | Path,
        checkpoint_args: dict[str, object],
        *,
        device: str,
        fast_deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.fast_deterministic = bool(fast_deterministic)
        self.agent, self.checkpoint_metadata = build_proposed_agent(
            env, checkpoint_path, checkpoint_args, device=device
        )
        self._initialize_window_feedback()

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        self._before_slow_plan(env)
        self.agent.maybe_update_deployment(env, deterministic=True, record=False)
        self._capture_deployment_usage(env)

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        if requests is not env.current_requests:
            env.current_requests = requests
            env.current_request = requests[0] if requests else None
        return self.agent.fast_agent.schedule_batch(
            env, requests, deterministic=self.fast_deterministic, record=False
        )
