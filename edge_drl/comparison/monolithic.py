from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from edge_drl.agents.drl import (
    FastSchedulingPPOAgent,
    HierarchicalPPOAgent,
    fast_obs_dim,
)
from edge_drl.models.ppo import GraphAttentionNodeScoringActorCritic
from edge_drl.comparison.checkpoint import load_checkpoint_configuration
from edge_drl.comparison.scheme import BaseComparisonScheme
from edge_drl.comparison.trace import ComparisonTrace, rehash_trace
from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import EdgeScenario, Service, TaskRequest
from train_dual_ppo import load_checkpoint


MONOLITHIC_FAST_DESIGN_V2 = "aggregate-kkt-v2"
MONOLITHIC_FAST_DESIGN_V3 = "aggregate-kkt-v3"
MONOLITHIC_FAST_DESIGN = "aggregate-kkt-v4"


class MonolithicTemperatureGATPolicy(GraphAttentionNodeScoringActorCritic):
    """GAT node scorer with an externally controlled behavior temperature."""

    logit_temperature: float = 1.0

    def forward(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        global_features = states[:, : self.global_dim]
        node_features = states[:, self.node_offset : self.edge_offset].reshape(
            -1,
            self.num_nodes,
            self.node_feature_dim,
        )
        edge_features = states[:, self.edge_offset :].reshape(
            -1,
            self.num_nodes,
            self.num_nodes,
            self.edge_feature_dim,
        )

        global_emb = self.global_encoder(global_features)
        node_emb = self.node_encoder(node_features)
        graph_emb = self._graph_attention(node_emb, edge_features)
        expanded_global = global_emb.unsqueeze(1).expand(-1, self.num_nodes, -1)
        pair_emb = torch.cat([expanded_global, graph_emb], dim=-1)
        temperature = max(float(self.logit_temperature), 1.0)
        logits = self.scorer(pair_emb).squeeze(-1) / temperature

        safe_masks = masks.bool()
        fallback = torch.zeros_like(safe_masks)
        fallback[:, 0] = True
        safe_masks = torch.where(
            safe_masks.any(dim=1, keepdim=True),
            safe_masks,
            fallback,
        )
        logits = logits.masked_fill(~safe_masks, -1e9)
        dist = Categorical(logits=logits)

        mask_float = safe_masks.float().unsqueeze(-1)
        pooled_nodes = (graph_emb * mask_float).sum(dim=1) / mask_float.sum(
            dim=1
        ).clamp_min(1.0)
        values = self.critic(torch.cat([global_emb, pooled_nodes], dim=-1)).squeeze(-1)
        return dist, values


class MonolithicFastSchedulingPPOAgent(FastSchedulingPPOAgent):
    """Fast PPO observation specialized for one aggregated service stage.

    The actor, critic, PPO optimizer, causal KKT ledger, action mask, and reward
    path remain the shared implementation.  Only the interpretation of the
    fixed-size ``chain-kkt-v3`` observation changes: fields that describe a
    non-existent downstream stage are replaced with aggregate-service demand,
    residual path capacity, and the exact candidate KKT marginal cost.

    Keeping the tensor shape unchanged makes checkpoints use the normal shared
    serialization path while keeping this behavior opt-in for Monolithic runs.
    """

    def _batch_reward_queue(self) -> list[int]:
        queue = getattr(self, "_monolithic_batch_reward_counts", None)
        if queue is None:
            queue = []
            self._monolithic_batch_reward_counts = queue
        return queue

    def schedule_batch(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        deterministic: bool = False,
        record: bool = True,
    ) -> list[list[int]]:
        schedules = super().schedule_batch(
            env,
            requests,
            deterministic=deterministic,
            record=record,
        )
        if (
            record
            and requests
            and getattr(self, "monolithic_fast_design", None)
            == MONOLITHIC_FAST_DESIGN
        ):
            self._batch_reward_queue().append(len(requests))
        return schedules

    def assign_schedule_rewards(
        self,
        rewards: list[float],
        done: bool,
        weight: float = 1.0,
    ) -> None:
        """Treat every jointly settled request batch as one PPO trajectory."""

        if getattr(self, "monolithic_fast_design", None) != MONOLITHIC_FAST_DESIGN:
            return super().assign_schedule_rewards(rewards, done=done, weight=weight)
        if len(rewards) != 1:
            raise ValueError("Monolithic Fast expects exactly one reward per request")
        queue = self._batch_reward_queue()
        if not queue:
            raise RuntimeError("Monolithic Fast received a reward without a recorded batch")
        remaining = int(queue[0])
        if remaining <= 0:
            raise RuntimeError("invalid Monolithic Fast batch reward counter")
        is_batch_terminal = remaining == 1
        self.ppo.buffer.rewards.append(float(rewards[0]))
        self.ppo.buffer.dones.append(bool(done or is_batch_terminal))
        self.ppo.buffer.weights.append(float(weight))
        if is_batch_terminal:
            queue.pop(0)
        else:
            queue[0] = remaining - 1

    def clear_batch_reward_context(self) -> None:
        self._batch_reward_queue().clear()

    def _build_state(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        partial_nodes: list[int],
        *,
        reservation_node_delta: np.ndarray | None = None,
        reservation_stage_delta: dict[tuple[int, int], np.ndarray] | None = None,
        kkt_compute_sqrt: np.ndarray | None = None,
        kkt_link_sqrt: np.ndarray | None = None,
        processed_request_weight: float = 0.0,
        total_request_weight: float = 1.0,
        remaining_requests: list[TaskRequest] | tuple[TaskRequest, ...] | None = None,
    ) -> np.ndarray:
        if self.chain_context != "chain-kkt-v3":
            raise ValueError(
                f"{MONOLITHIC_FAST_DESIGN} requires fast_chain_context=chain-kkt-v3"
            )
        if stage_id != 0 or len(request.stage_compute_gcycles) != 1:
            raise ValueError(
                "Monolithic Fast expects one aggregated stage per request"
            )
        env._require_ready()
        assert env.scenario is not None

        nodes = env.scenario.nodes
        max_compute = max(float(node.compute_gcycles_per_s) for node in nodes)
        finite_bandwidth = np.where(
            np.isfinite(env.scenario.bandwidth_mb_s),
            env.scenario.bandwidth_mb_s,
            0.0,
        )
        max_bandwidth = max(float(np.max(finite_bandwidth)), 1e-9)
        use_remaining_context = (
            getattr(self, "monolithic_fast_design", None)
            == MONOLITHIC_FAST_DESIGN
        )
        active_requests = list(
            env.current_requests
            if remaining_requests is None or not use_remaining_context
            else remaining_requests
        )
        tick_weight = float(sum(item.request_count for item in active_requests))
        tick_events = float(len(active_requests))
        service_weight = float(
            sum(
                item.request_count
                for item in active_requests
                if item.service_id == request.service_id
            )
        )
        home_weight = np.zeros(self.num_nodes, dtype=np.float64)
        remaining_compute = 0.0
        remaining_input = 0.0
        for item in active_requests:
            home_weight[int(item.home_node)] += float(item.request_count)
            remaining_compute += float(item.request_count) * float(
                sum(item.stage_compute_gcycles)
            )
            remaining_input += float(item.request_count) * float(item.input_mb)
        full_compute = max(
            sum(
                float(item.request_count) * float(sum(item.stage_compute_gcycles))
                for item in env.current_requests
            ),
            1e-9,
        )
        full_input = max(
            sum(
                float(item.request_count) * float(item.input_mb)
                for item in env.current_requests
            ),
            1e-9,
        )

        expected_node_work, _ = self._expected_current_compute_workload(env)
        if reservation_node_delta is not None:
            expected_node_work = np.maximum(
                expected_node_work + reservation_node_delta,
                0.0,
            )
        candidate_post_work = self._candidate_post_compute_workload(
            env,
            request,
            stage_id,
            expected_node_work,
        )
        compute_capacity, _ = self._effective_kkt_capacities(env)
        deployed = (
            env.deployment[request.service_id, 0]
            if env.deployment is not None
            else np.zeros(self.num_nodes, dtype=bool)
        )
        mask = self._build_mask(env, request, stage_id, partial_nodes)

        stages = len(request.stage_compute_gcycles)
        if kkt_compute_sqrt is None:
            kkt_compute_sqrt = np.zeros((stages, self.num_nodes), dtype=np.float64)
        if kkt_link_sqrt is None:
            kkt_link_sqrt = np.zeros(
                (stages, self.num_nodes, self.num_nodes),
                dtype=np.float64,
            )

        multiplicity = max(float(request.request_count), 1.0)
        candidate_components: list[tuple[float, float, float, float]] = []
        for node_id in range(self.num_nodes):
            compute_cost = self._kkt_total_increment(
                kkt_compute_sqrt[stage_id, node_id],
                float(request.stage_compute_gcycles[stage_id]),
                multiplicity,
                compute_capacity[node_id],
            ) / multiplicity
            link_total, propagation_cost = self._routed_kkt_increment(
                env,
                phase=stage_id,
                source=int(request.home_node),
                target=node_id,
                data_mb=float(request.input_mb),
                multiplicity=multiplicity,
                link_sqrt=kkt_link_sqrt,
            )
            link_cost = link_total / multiplicity
            total_cost = compute_cost + link_cost + propagation_cost
            candidate_components.append(
                (
                    float(np.tanh(compute_cost)),
                    float(np.tanh(link_cost)) if np.isfinite(link_cost) else 1.0,
                    float(np.tanh(propagation_cost / 0.05))
                    if np.isfinite(propagation_cost)
                    else 1.0,
                    float(total_cost),
                )
            )
        finite_feasible_costs = [
            candidate_components[node_id][3]
            for node_id in np.flatnonzero(mask)
            if np.isfinite(candidate_components[node_id][3])
        ]
        best_cost = min(finite_feasible_costs) if finite_feasible_costs else 0.0

        node_features: list[float] = []
        for node in nodes:
            node_id = int(node.node_id)
            path = env.shortest_path(int(request.home_node), node_id)
            reachable = path is not None
            transfer_s, propagation_s = self._route_proxy_delay(
                env,
                int(request.home_node),
                node_id,
                float(request.input_mb),
            )
            if path is None or len(path) < 2:
                residual_bottleneck = max_bandwidth if node_id == request.home_node else 0.0
                hop_fraction = 0.0 if node_id == request.home_node else 1.0
            else:
                residual_bandwidths = [
                    float(env.scenario.bandwidth_mb_s[left, right])
                    * max(1.0 - 0.75 * float(env.link_load[left, right]), 0.10)
                    for left, right in zip(path, path[1:])
                ]
                residual_bottleneck = min(residual_bandwidths)
                hop_fraction = (len(path) - 1) / max(self.num_nodes - 1, 1)

            compute_feature, link_feature, propagation_feature, total_cost = (
                candidate_components[node_id]
            )
            relative_excess = (
                max(total_cost - best_cost, 0.0)
                if np.isfinite(total_cost)
                else 10.0
            )
            residual_compute = float(compute_capacity[node_id]) / max_compute
            node_features.extend(
                [
                    float(deployed[node_id]),
                    float(reachable),
                    float(node.compute_gcycles_per_s) / max_compute,
                    float(env.node_compute_load[node_id]),
                    residual_compute,
                    float(node_id == request.home_node),
                    residual_bottleneck / max_bandwidth,
                    float(hop_fraction),
                    float(np.tanh(transfer_s / 0.25)) if np.isfinite(transfer_s) else 1.0,
                    float(np.tanh(propagation_s / 0.05)) if np.isfinite(propagation_s) else 1.0,
                    home_weight[node_id] / max(tick_weight, 1.0),
                    self._normalize_compute_pressure(
                        expected_node_work[node_id],
                        compute_capacity[node_id],
                    ),
                    self._normalize_compute_pressure(
                        candidate_post_work[node_id],
                        compute_capacity[node_id],
                    ),
                    compute_feature,
                    link_feature,
                    propagation_feature,
                    float(np.tanh(total_cost)) if np.isfinite(total_cost) else 1.0,
                    float(np.tanh(relative_excess)),
                ]
            )

        finite_link_load = env.link_load[np.isfinite(env.link_load)]
        progress = float(processed_request_weight) / max(float(total_request_weight), 1.0)
        global_features = [
            request.service_id / max(env.config.num_service_types - 1, 1),
            request.home_node / max(self.num_nodes - 1, 1),
            float(np.tanh(float(request.input_mb) / 2.0)),
            float(np.tanh(float(request.stage_compute_gcycles[0]) / 8.0)),
            float(np.tanh(float(request.deadline_s) / 0.6)),
            np.log1p(float(request.request_count)) / np.log1p(5_000.0),
            env.current_time_minute / max(env.config.episode_minutes, 1),
            np.log1p(tick_weight) / np.log1p(5_000.0),
            tick_events / max(env.config.num_edge_nodes * env.config.num_service_types, 1),
            service_weight / max(tick_weight, 1.0),
            float(np.mean(env.node_compute_load)),
            float(np.max(env.node_compute_load)),
            float(np.mean(finite_link_load)) if finite_link_load.size else 0.0,
            float(np.max(finite_link_load)) if finite_link_load.size else 0.0,
            float(np.count_nonzero(mask)) / max(self.num_nodes, 1),
            progress,
            remaining_compute / full_compute,
            remaining_input / full_input,
        ]
        if self.policy_kind == "gat_node_scorer":
            node_features.extend(self._build_edge_features(env))
        state = np.asarray(global_features + node_features, dtype=np.float32)
        expected_dim = fast_obs_dim(
            self.num_nodes,
            self.policy_kind,
            self.chain_context,
        )
        if state.shape != (expected_dim,):
            raise RuntimeError(
                f"Monolithic Fast state shape changed from {(expected_dim,)} to {state.shape}"
            )
        return state


class StableMonolithicFastSchedulingPPOAgent(MonolithicFastSchedulingPPOAgent):
    """Monolithic Fast with PPO-consistent adaptive logit temperature.

    Temperature changes only between PPO updates.  A rollout and its following
    optimization therefore use the exact same policy distribution, preserving
    the behavior log-probability contract required by PPO.
    """

    logit_temperature_max: float = 3.0

    def _temperature_from_entropy_coefficient(self) -> float:
        lower = float(self.entropy_coef)
        upper = max(float(self.entropy_max_coef), lower + 1e-9)
        fraction = np.clip(
            (float(self.entropy_current_coef) - lower) / (upper - lower),
            0.0,
            1.0,
        )
        return float(
            np.exp(np.log(max(float(self.logit_temperature_max), 1.0)) * fraction)
        )

    def _sync_logit_temperature(self) -> float:
        if not isinstance(self.ppo.policy, MonolithicTemperatureGATPolicy):
            raise TypeError("stable Monolithic Fast requires the temperature GAT policy")
        temperature = self._temperature_from_entropy_coefficient()
        self.ppo.policy.logit_temperature = temperature
        return temperature

    def schedule_batch(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        deterministic: bool = False,
        record: bool = True,
    ) -> list[list[int]]:
        self._sync_logit_temperature()
        return super().schedule_batch(
            env,
            requests,
            deterministic=deterministic,
            record=record,
        )

    def update(
        self,
        *,
        progress_label: str = "",
        progress_interval_seconds: float = 0.0,
    ) -> dict[str, float]:
        temperature = self._sync_logit_temperature()
        metrics = super().update(
            progress_label=progress_label,
            progress_interval_seconds=progress_interval_seconds,
        )
        next_temperature = self._sync_logit_temperature()
        metrics["logit_temperature"] = temperature
        metrics["logit_temperature_next"] = next_temperature
        metrics["logit_temperature_max"] = float(self.logit_temperature_max)
        return metrics


def specialize_monolithic_fast_agent(
    agent: HierarchicalPPOAgent,
    design: str | None,
    *,
    logit_temperature_max: float = 3.0,
) -> HierarchicalPPOAgent:
    """Enable the opt-in Monolithic observation without rebuilding PPO state."""

    if design not in {
        MONOLITHIC_FAST_DESIGN_V2,
        MONOLITHIC_FAST_DESIGN_V3,
        MONOLITHIC_FAST_DESIGN,
    }:
        return agent
    if agent.fast_agent.chain_context != "chain-kkt-v3":
        raise ValueError(
            f"{MONOLITHIC_FAST_DESIGN} requires fast_chain_context=chain-kkt-v3"
        )
    # The specialization adds methods but no fields.  Retaining the same
    # instance preserves initialization, optimizer references, and checkpoint
    # serialization while changing behavior only for explicitly tagged runs.
    agent.fast_agent.monolithic_fast_design = design
    if design == MONOLITHIC_FAST_DESIGN_V3:
        if not isinstance(
            agent.fast_agent.ppo.policy,
            GraphAttentionNodeScoringActorCritic,
        ):
            raise ValueError(
                f"{MONOLITHIC_FAST_DESIGN} requires fast_policy_kind=gat_node_scorer"
            )
        agent.fast_agent.__class__ = StableMonolithicFastSchedulingPPOAgent
        agent.fast_agent.logit_temperature_max = max(
            float(logit_temperature_max),
            1.0,
        )
        agent.fast_agent.ppo.policy.__class__ = MonolithicTemperatureGATPolicy
        agent.fast_agent._sync_logit_temperature()
    else:
        agent.fast_agent.__class__ = MonolithicFastSchedulingPPOAgent
    return agent


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


def collapse_deployment_for_monolithic(
    scenario: EdgeScenario,
    deployment: np.ndarray,
) -> np.ndarray:
    """Project staged deployment state into complete-service replicas.

    A node is a valid Monolithic replica only if every original service stage
    is deployed there.  This preserves the previous deployment in Slow's
    evaluation observation without inventing a complete service replica.
    """

    collapsed = np.zeros_like(np.asarray(deployment, dtype=bool))
    for service in scenario.services:
        valid_stage_ids = [int(stage.stage_id) for stage in service.stages]
        if valid_stage_ids:
            collapsed[int(service.service_id), 0] = np.all(
                deployment[int(service.service_id), valid_stage_ids],
                axis=0,
            )
    return collapsed


def _copy_runtime_state(source: EdgeComputingEnv, target: EdgeComputingEnv) -> None:
    target.node_compute_load = source.node_compute_load.copy()
    target.link_load = source.link_load.copy()
    target.current_time_minute = source.current_time_minute
    target.next_deployment_update_minute = source.next_deployment_update_minute
    target.last_load_update_minute = source.last_load_update_minute


class MSOScheme(BaseComparisonScheme):
    """DRL counterpart whose only structural change is stage aggregation.

    PPO architecture, optimizer, reward path, two time scales, and KKT
    settlement are shared with the Proposed implementation. A separately
    trained checkpoint is mandatory; no MILP or heuristic is used here.
    """

    name = "Monolithic Service Orchestration (MSO)"

    def __init__(
        self,
        env: EdgeComputingEnv,
        checkpoint_path: str | Path,
        *,
        device: str,
        fast_deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.fast_deterministic = bool(fast_deterministic)
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
        self._window_latency_samples: list[tuple[float, float]] = []
        self._window_penalty_latency_sum = 0.0
        self._window_deadline_violations = 0.0
        self._window_invalid_actions = 0.0
        self._window_weight = 0.0
        self._window_max_node_load = 0.0
        self._window_max_link_load = 0.0
        self._window_memory_fraction = 0.0
        self._window_storage_fraction = 0.0

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
        specialize_monolithic_fast_agent(
            agent,
            str(checkpoint_args.get("monolithic_fast_design", "")) or None,
            logit_temperature_max=float(
                checkpoint_args.get("monolithic_fast_logit_temperature_max", 3.0)
            ),
        )
        load_checkpoint(agent, self.checkpoint_path)
        agent.slow_agent.count_ppo.policy.eval()
        if agent.slow_agent.stage_ppo is not None:
            agent.slow_agent.stage_ppo.policy.eval()
        agent.slow_agent.placement_ppo.policy.eval()
        agent.slow_agent.window_critic.eval()
        agent.fast_agent.ppo.policy.eval()
        return agent

    def maybe_plan(self, env: EdgeComputingEnv) -> None:
        if not env.needs_deployment_update:
            return
        if env.current_time_minute > 1e-9:
            self._finalize_window_feedback()
        else:
            self.agent.slow_agent.last_window_feedback = {}
            self._reset_window_feedback()
        _copy_runtime_state(env, self._planning_env)
        assert env.scenario is not None and env.deployment is not None
        self._planning_env.deployment = collapse_deployment_for_monolithic(
            env.scenario,
            env.deployment,
        )
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
        self._window_memory_fraction, self._window_storage_fraction = (
            self.agent._deployment_resource_fractions(env)
        )

    def _finalize_window_feedback(self) -> None:
        if self._window_weight <= 0.0:
            self.agent.slow_agent.last_window_feedback = {}
            self._reset_window_feedback()
            return
        ordered = sorted(self._window_latency_samples, key=lambda item: item[0])
        cutoff = 0.95 * self._window_weight
        cumulative = 0.0
        p95 = ordered[-1][0]
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= cutoff:
                p95 = value
                break
        mean_latency = sum(
            value * weight for value, weight in self._window_latency_samples
        ) / self._window_weight
        self.agent.slow_agent.last_window_feedback = {
            "avg_latency_s": float(mean_latency),
            "p95_latency_s": float(p95),
            "avg_penalty_latency_s": float(
                self._window_penalty_latency_sum / self._window_weight
            ),
            "deadline_violation_rate": float(
                self._window_deadline_violations / self._window_weight
            ),
            "invalid_action_rate": float(
                self._window_invalid_actions / self._window_weight
            ),
            "max_node_compute_load": float(self._window_max_node_load),
            "max_link_load": float(self._window_max_link_load),
            "deployment_memory_fraction": float(self._window_memory_fraction),
            "deployment_storage_fraction": float(self._window_storage_fraction),
        }
        self._reset_window_feedback()

    def _reset_window_feedback(self) -> None:
        self._window_latency_samples.clear()
        self._window_penalty_latency_sum = 0.0
        self._window_deadline_violations = 0.0
        self._window_invalid_actions = 0.0
        self._window_weight = 0.0
        self._window_max_node_load = 0.0
        self._window_max_link_load = 0.0

    def observe_step(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        schedules: list[list[int]],
        info: dict[str, Any],
    ) -> None:
        del schedules
        for request, group_info in zip(requests, info.get("group_infos", [])):
            weight = float(group_info.get("request_count", request.request_count))
            latency = float(group_info["latency_s"])
            self._window_latency_samples.append((latency, weight))
            self._window_penalty_latency_sum += float(
                group_info.get("penalty_latency_s", 0.0)
            ) * weight
            self._window_deadline_violations += float(
                latency > float(request.deadline_s)
            ) * weight
            self._window_invalid_actions += float(
                not bool(group_info.get("valid", False))
            ) * weight
            self._window_weight += weight
        self._window_max_node_load = max(
            self._window_max_node_load,
            float(np.max(env.node_compute_load)),
        )
        finite_links = env.link_load[np.isfinite(env.link_load)]
        if finite_links.size:
            self._window_max_link_load = max(
                self._window_max_link_load,
                float(np.max(finite_links)),
            )

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
                env,
                compact_requests,
                deterministic=self.fast_deterministic,
                record=False,
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


MonolithicScheme = MSOScheme
