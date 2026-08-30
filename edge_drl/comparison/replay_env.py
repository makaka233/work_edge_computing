from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from edge_drl.comparison.trace import ComparisonTrace
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from edge_drl.env.scenario import EdgeScenario, TaskRequest


class TraceReplayEnv(EdgeComputingEnv):
    """Existing physical simulator driven only by an immutable external trace."""

    def __init__(self, config: EdgeEnvConfig, scenario: EdgeScenario, trace: ComparisonTrace):
        if config.episode_minutes * 60 != trace.logical_steps:
            raise ValueError("trace length must equal episode_minutes * 60")
        self._comparison_scenario = deepcopy(scenario)
        self._comparison_trace = trace
        super().__init__(config)

    def reset(self) -> dict[str, Any]:
        self.rng = np.random.default_rng(self.config.seed)
        self.scenario = deepcopy(self._comparison_scenario)
        self.deployment = np.zeros(
            (self.config.num_service_types, self.config.max_service_stages, self.config.num_edge_nodes),
            dtype=bool,
        )
        self.current_time_minute = 0.0
        self.next_deployment_update_minute = 0.0
        self.request_counter = 0
        self.node_compute_load = np.zeros(self.config.num_edge_nodes, dtype=np.float64)
        self.link_load = np.zeros(
            (self.config.num_edge_nodes, self.config.num_edge_nodes), dtype=np.float64
        )
        self.last_load_update_minute = 0.0
        self.last_migration_cost = 0.0
        self._route_cache.clear()
        self.metrics = {
            "requests": 0.0,
            "request_events": 0.0,
            "time_steps": 0.0,
            "settlement_steps": 0.0,
            "invalid_actions": 0.0,
            "total_latency_s": 0.0,
            "valid_requests": 0.0,
            "total_valid_latency_s": 0.0,
            "total_penalty_latency_s": 0.0,
            "deadline_violations": 0.0,
            "deployment_updates": 0.0,
        }
        self.current_requests = self._requests_for_step(0)
        self.current_request = self.current_requests[0] if self.current_requests else None
        return self.observe()

    def _requests_for_step(self, step: int) -> list[TaskRequest]:
        if step < 0 or step >= self._comparison_trace.logical_steps:
            return []
        return list(self._comparison_trace.slots[step])

    def _generate_current_second_requests(self) -> list[TaskRequest]:
        step = int(round(self.current_time_minute * 60.0))
        return self._requests_for_step(step)
