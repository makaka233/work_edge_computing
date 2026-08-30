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
        return [
            replace(
                request,
                stage_compute_gcycles=(sum(request.stage_compute_gcycles),),
                stage_output_mb=(0.0,),
            )
            for request in requests
        ]

    def schedule_batch(self, env: EdgeComputingEnv, requests: list[TaskRequest]) -> list[list[int]]:
        return self.agent.fast_agent.schedule_batch(env, requests, deterministic=True, record=False)
