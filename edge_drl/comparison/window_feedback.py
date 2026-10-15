"""Evaluation-only previous-window observations for frozen Slow PPO policies."""
from __future__ import annotations

from typing import Any

import numpy as np

from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest
from edge_drl.agents.slow_history import SlowRouteHistory


SLOW_FEEDBACK_PROTOCOL = "own-window-weighted-feedback-v1"


class SlowWindowFeedbackMixin:
    """Observe each scheme's own settlement without recording PPO samples.

    The next deployment sees only the preceding window. Policy weights,
    optimizers, action selection rules and reward definitions remain unchanged.
    """

    def _initialize_window_feedback(self) -> None:
        self._window_latency_samples: list[tuple[float, float]] = []
        self._window_memory_fraction = 0.0
        self._window_storage_fraction = 0.0
        self._reset_window_feedback()

    def _reset_window_feedback(self) -> None:
        self._window_latency_samples.clear()
        self._window_penalty_latency_sum = 0.0
        self._window_deadline_violations = 0.0
        self._window_invalid_actions = 0.0
        self._window_weight = 0.0
        self._window_max_node_load = 0.0
        self._window_max_link_load = 0.0
        self._window_route_history = None

    def _before_slow_plan(self, env: EdgeComputingEnv) -> None:
        if env.current_time_minute <= 1e-9:
            self.agent.slow_agent.last_window_feedback = {}
            self.agent.slow_agent.last_route_history = {}
            self._reset_window_feedback()
            self._window_memory_fraction = 0.0
            self._window_storage_fraction = 0.0
        else:
            self._finalize_window_feedback()
        self.diagnostics.planning.append({
            "time_minute": float(env.current_time_minute),
            "planner": "slow_window_feedback",
            "feedback_protocol": SLOW_FEEDBACK_PROTOCOL,
            "has_previous_window_feedback": bool(self.agent.slow_agent.last_window_feedback),
            **self.agent.slow_agent.last_window_feedback,
        })

    def _capture_deployment_usage(self, env: EdgeComputingEnv) -> None:
        # Match training's global-capacity fractions, not the unweighted mean
        # of per-node utilization; physical MSO projection uses the same sums.
        assert env.scenario is not None and env.deployment is not None
        memory = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        storage = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        for service in env.scenario.services:
            for stage in service.stages:
                placed = env.deployment[service.service_id, stage.stage_id]
                memory += placed * stage.memory_gb
                storage += placed * stage.storage_gb
        self._window_memory_fraction = float(memory.sum() / max(float(env.service_memory_capacities().sum()), 1e-9))
        self._window_storage_fraction = float(storage.sum() / max(float(env.service_storage_capacities().sum()), 1e-9))

    def _finalize_window_feedback(self) -> None:
        if self._window_weight <= 0.0:
            self.agent.slow_agent.last_window_feedback = {}
            self.agent.slow_agent.last_route_history = {}
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
        self.agent.slow_agent.last_window_feedback = {
            "avg_latency_s": float(sum(value * weight for value, weight in self._window_latency_samples) / self._window_weight),
            "p95_latency_s": float(p95),
            "avg_penalty_latency_s": float(self._window_penalty_latency_sum / self._window_weight),
            "deadline_violation_rate": float(self._window_deadline_violations / self._window_weight),
            "invalid_action_rate": float(self._window_invalid_actions / self._window_weight),
            "max_node_compute_load": float(self._window_max_node_load),
            "max_link_load": float(self._window_max_link_load),
            "deployment_memory_fraction": float(self._window_memory_fraction),
            "deployment_storage_fraction": float(self._window_storage_fraction),
        }
        if getattr(self.agent.slow_agent,"tail_reallocation",False):
            self.agent.slow_agent.last_route_history = self._window_route_history.snapshot() if self._window_route_history is not None else {}
        self._reset_window_feedback()

    def observe_step(self, env: EdgeComputingEnv, requests: list[TaskRequest],
                     schedules: list[list[int]], info: dict[str, Any]) -> None:
        del schedules
        groups = info.get("group_infos", [])
        if len(groups) != len(requests):
            raise ValueError("Slow evaluation feedback requires one settlement result per request")
        for request, group in zip(requests, groups):
            # env.step has already weighted request_count by the temporal hold.
            # Do not multiply it again or replace it with an unweighted count.
            weight = float(group.get("request_count", request.request_count))
            if weight <= 0.0:
                continue
            latency = float(group["latency_s"])
            self._window_latency_samples.append((latency, weight))
            self._window_penalty_latency_sum += float(group.get("penalty_latency_s", 0.0)) * weight
            self._window_deadline_violations += float(latency > float(request.deadline_s)) * weight
            self._window_invalid_actions += float(not bool(group.get("valid", False))) * weight
            self._window_weight += weight
            if getattr(self.agent.slow_agent,"tail_reallocation",False):
                if self._window_route_history is None:
                    self._window_route_history = SlowRouteHistory(env.config.num_edge_nodes)
                self._window_route_history.add(request.service_id,group['stage_nodes'],group,weight)
        self._window_max_node_load = max(self._window_max_node_load, float(np.max(env.node_compute_load)))
        finite_links = env.link_load[np.isfinite(env.link_load)]
        if finite_links.size:
            self._window_max_link_load = max(self._window_max_link_load, float(np.max(finite_links)))

    def finish_evaluation_episode(self, env: EdgeComputingEnv) -> None:
        del env
        self._finalize_window_feedback()
