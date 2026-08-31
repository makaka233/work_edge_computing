from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.comparison.checkpoint import load_checkpoint_configuration
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.trace import ComparisonTrace, rehash_trace
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import EdgeScenario, Service, TaskRequest
from train_dual_ppo import load_checkpoint


def collapse_scenario(base: EdgeScenario) -> EdgeScenario:
    """Create the Monolithic view: one aggregated stage per service."""

    scenario = deepcopy(base)
    services: list[Service] = []
    for service in scenario.services:
        if not service.stages:
            raise ValueError(f"service {service.service_id} has no stages")
        aggregated = replace(
            service.stages[0],
            stage_id=0,
            memory_gb=sum(stage.memory_gb for stage in service.stages),
            storage_gb=sum(stage.storage_gb for stage in service.stages),
            compute_gcycles_mean=sum(stage.compute_gcycles_mean for stage in service.stages),
            output_mb_mean=0.0,
        )
        services.append(replace(service, stages=(aggregated,)))
    scenario.services = services
    return scenario


def collapse_trace(trace: ComparisonTrace) -> ComparisonTrace:
    """Collapse each request's staged workload while retaining its metadata."""

    slots: list[tuple[TaskRequest, ...]] = []
    for slot in trace.slots:
        slots.append(
            tuple(
                replace(
                    request,
                    stage_compute_gcycles=(sum(request.stage_compute_gcycles),),
                    stage_output_mb=(0.0,),
                )
                for request in slot
            )
        )
    return rehash_trace(trace, tuple(slots))


def adapt_request_for_monolithic_evaluation(request: TaskRequest) -> TaskRequest:
    """Keep the physical stage path while concentrating work in one stage.

    Training uses a genuinely one-stage scenario and trace.  Evaluation must
    still call the shared physical simulator with the original number of
    stages, otherwise stage-wise deployment/resource metrics would no longer
    be comparable.  The first stage carries the aggregate compute demand and
    all remaining stages are zero-work; every inter-stage output is zero.
    """

    stage_count = len(request.stage_compute_gcycles)
    if stage_count < 1:
        raise ValueError("a TaskRequest must contain at least one stage")
    return replace(
        request,
        stage_compute_gcycles=(sum(request.stage_compute_gcycles),)
        + (0.0,) * (stage_count - 1),
        stage_output_mb=(0.0,) * stage_count,
    )


def compact_request_for_monolithic_policy(request: TaskRequest) -> TaskRequest:
    """Build the one-stage copy consumed by the Monolithic Fast policy."""

    if not request.stage_compute_gcycles:
        raise ValueError("a TaskRequest must contain at least one stage")
    return replace(
        request,
        stage_compute_gcycles=(float(sum(request.stage_compute_gcycles)),),
        stage_output_mb=(0.0,),
    )


def expand_monolithic_schedule(schedule: list[int] | tuple[int, ...], stage_count: int) -> list[int]:
    """Project one aggregate node choice onto every original service stage."""

    if stage_count < 1:
        raise ValueError("stage_count must be positive")
    if not schedule:
        raise ValueError("Monolithic Fast policy returned an empty schedule")
    return [int(schedule[0])] * stage_count


def _copy_runtime_state(source: EdgeComputingEnv, target: EdgeComputingEnv) -> None:
    target.node_compute_load = source.node_compute_load.copy()
    target.link_load = source.link_load.copy()
    target.current_time_minute = source.current_time_minute
    target.next_deployment_update_minute = source.next_deployment_update_minute


class MonolithicScheme(BaseComparisonScheme):
    """DRL counterpart whose only structural change is stage aggregation.

    PPO architecture, optimizer, reward path, two time scales, and KKT
    settlement are shared with the Proposed implementation. A separately
    trained checkpoint is mandatory; no MILP or heuristic is used here.
    """

    name = "Monolithic"

    def __init__(self, env: EdgeComputingEnv, checkpoint_path: str | Path, *, device: str) -> None:
        super().__init__()
        if env.scenario is None or env.deployment is None:
            raise RuntimeError("environment must be reset before building Monolithic")
        self.checkpoint_path, checkpoint_args, self.checkpoint_metadata = load_checkpoint_configuration(
            checkpoint_path
        )
        self._planning_env = deepcopy(env)
        self._planning_env.scenario = collapse_scenario(env.scenario)
        self._planning_env.deployment = np.zeros_like(env.deployment)
        self._planning_env._route_cache.clear()
        self.agent = self._load_agent(self._planning_env, checkpoint_args, device)

    def _load_agent(
        self, planning_env: EdgeComputingEnv, checkpoint_args: dict[str, Any], device: str
    ) -> HierarchicalPPOAgent:
        from inspect import signature

        kwargs: dict[str, Any] = {}
        for name in signature(HierarchicalPPOAgent.from_env).parameters:
            if name in {"cls", "env"}:
                continue
            if name in checkpoint_args and checkpoint_args[name] is not None:
                kwargs[name] = checkpoint_args[name]
        kwargs["device"] = device
        replicas = int(checkpoint_args.get("replicas_per_stage", 0))
        kwargs["replicas_per_stage"] = planning_env.config.num_edge_nodes if replicas <= 0 else replicas
        agent = HierarchicalPPOAgent.from_env(planning_env, **kwargs)
        load_checkpoint(agent, self.checkpoint_path)
        agent.slow_agent.count_ppo.policy.eval()
        agent.slow_agent.placement_ppo.policy.eval()
        agent.slow_agent.window_critic.eval()
        agent.fast_agent.ppo.policy.eval()
        return agent

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        _copy_runtime_state(env, self._planning_env)
        self._planning_env.deployment = np.zeros_like(env.deployment)
        self._planning_env.current_requests = []
        self._planning_env.current_request = None
        monolithic_deployment = self.agent.slow_agent.plan_deployment(
            self._planning_env, deterministic=True, record=False
        )
        projected = np.zeros_like(env.deployment, dtype=bool)
        assert env.scenario is not None
        for service in env.scenario.services:
            source = monolithic_deployment[service.service_id, 0]
            for stage in service.stages:
                projected[service.service_id, stage.stage_id] = source
        env.apply_deployment(projected)

    def adapt_requests(self, requests: list[TaskRequest]) -> list[TaskRequest]:
        return [adapt_request_for_monolithic_evaluation(request) for request in requests]

    def schedule_batch(self, env: EdgeComputingEnv, requests: list[TaskRequest]) -> list[list[int]]:
        if not requests:
            return []

        # The policy was trained against a one-stage scenario.  Run inference
        # on one-stage copies, while preserving the original stage-shaped
        # requests on the physical evaluation environment for env.step().
        compact_requests = [compact_request_for_monolithic_policy(request) for request in requests]
        previous_requests = env.current_requests
        previous_request = env.current_request
        env.current_requests = compact_requests
        env.current_request = compact_requests[0]
        try:
            compact_schedules = self.agent.fast_agent.schedule_batch(
                env, compact_requests, deterministic=True, record=False
            )
        finally:
            env.current_requests = previous_requests
            env.current_request = previous_request
            # Avoid retaining a workload estimate computed for the temporary
            # compact request list after restoring the physical view.
            self.agent.fast_agent._workload_cache_key = None
            self.agent.fast_agent._workload_cache = None

        if len(compact_schedules) != len(requests):
            raise RuntimeError(
                "Monolithic Fast policy returned a schedule count different from the request count"
            )
        return [
            expand_monolithic_schedule(schedule, len(request.stage_compute_gcycles))
            for request, schedule in zip(requests, compact_schedules)
        ]
