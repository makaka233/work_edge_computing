from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from edge_drl.comparison.monolithic import collapse_scenario
from edge_drl.comparison.scenario_transforms import transform_scenario
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from edge_drl.env.scenario import EdgeScenario, TaskRequest


MONOLITHIC_REQUEST_GENERATOR = "proposed-on-demand-v1"


class MonolithicTrainingEnv(EdgeComputingEnv):
    """Proposed's on-demand request stream, with a one-stage execution view.

    Requests are drawn from the original staged catalogue before collapse. This
    preserves the number/order of RNG draws, payloads, deadlines and total work.
    The inherited reset, step and temporal-jitter sampler advance both random
    streams exactly as in Proposed; unobserved seconds are not pre-generated.
    Immutable TraceReplayEnv remains the separate comparison-evaluation path.
    """

    def __init__(self, config: EdgeEnvConfig, point: ExperimentPoint):
        self.point = point
        self._request_scenario: EdgeScenario | None = None
        self.generated_request_batches = 0
        self.generated_request_events = 0
        super().__init__(config)

    def reset(self) -> dict[str, Any]:
        self._request_scenario = None
        self.generated_request_batches = 0
        self.generated_request_events = 0
        super().reset()
        assert self.scenario is not None
        self._request_scenario = self.scenario
        self.scenario = collapse_scenario(transform_scenario(self.scenario, self.point))
        return self.observe()

    def _generate_current_second_requests(self) -> list[TaskRequest]:
        execution_scenario = self.scenario
        if self._request_scenario is not None:
            self.scenario = self._request_scenario
        try:
            requests = super()._generate_current_second_requests()
        finally:
            self.scenario = execution_scenario
        self.generated_request_batches += 1
        self.generated_request_events += len(requests)
        # Intermediate-data and heterogeneity transforms preserve total compute
        # and input payload. Inter-stage data is absent in the complete service.
        return [
            replace(request,
                    stage_compute_gcycles=(sum(request.stage_compute_gcycles),),
                    stage_output_mb=(0.0,))
            for request in requests
        ]

    def set_window_load(self, multiplier: float, group: int) -> None:
        """Match Proposed's window-boundary observation and RNG transition."""
        previous_multiplier = float(self.config.demand_load_multiplier)
        self.config.demand_load_multiplier = float(multiplier)
        self.config.demand_load_group = int(group)
        if not np.isclose(previous_multiplier, multiplier):
            self.current_requests = self._generate_current_second_requests()
            self.current_request = self.current_requests[0] if self.current_requests else None
