from __future__ import annotations

from edge_drl.comparison.types import SchemeDiagnostics
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


class BaseComparisonScheme:
    name = "base"

    def __init__(self) -> None:
        self.diagnostics = SchemeDiagnostics()

    def adapt_requests(self, requests: list[TaskRequest]) -> list[TaskRequest]:
        return requests

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        raise NotImplementedError

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]:
        raise NotImplementedError
