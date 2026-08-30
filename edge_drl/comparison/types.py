from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


@dataclass(frozen=True)
class ExperimentPoint:
    family: str
    value: float
    label: str


@dataclass
class SchemeDiagnostics:
    planning: list[dict[str, Any]] = field(default_factory=list)
    routing: list[dict[str, Any]] = field(default_factory=list)


class ComparisonScheme(Protocol):
    name: str
    diagnostics: SchemeDiagnostics

    def maybe_plan(self, env: EdgeComputingEnv) -> None: ...

    def adapt_requests(self, requests: list[TaskRequest]) -> list[TaskRequest]: ...

    def schedule_batch(
        self, env: EdgeComputingEnv, requests: list[TaskRequest]
    ) -> list[list[int]]: ...


@dataclass
class EpisodeResult:
    scheme: str
    scenario_family: str
    scenario_value: float
    scenario_label: str
    eval_seed: int
    routing_repeat: int
    trace_hash: str
    logical_steps: int
    settlement_steps: int
    request_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    episode_total_latency_s: float
    mean_slot_total_latency_s: float
    deadline_violation_rate: float
    invalid_action_rate: float
    mean_compute_delay_ms: float
    mean_link_delay_ms: float
    mean_access_delay_ms: float
    mean_propagation_delay_ms: float
    mean_replicas: float
    avg_replicas_per_stage: float
    mean_used_nodes: float
    used_replica_rate: float
    cross_node_transition_rate: float
    mean_deployment_memory_utilization: float
    mean_deployment_storage_utilization: float
    mean_node_load: float
    max_node_load: float
    mean_link_load: float
    max_link_load: float
    planning_time_s: float
    scheduling_time_s: float
    kkt_settlement_time_s: float
    total_runtime_s: float
    failed: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deployment_replica_count(env: EdgeComputingEnv) -> int:
    return int(np.asarray(env.deployment, dtype=bool).sum())
