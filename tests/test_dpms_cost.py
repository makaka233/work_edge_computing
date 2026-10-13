"""Differential tests against the pre-acceleration DPMS scalar recurrence."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from edge_drl.comparison.algorithmic import DPMSScheme, DemandAwareMILPPlacement
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.placement_cost import PreparedServiceCost
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.env.scenario import Service, ServiceStage
from tests.comparison_helpers import tiny_replay_env


def scalar_service_cost(service, selected, demand, node_capacities, propagation,
                        inverse_bandwidth, task_compute_scale, task_data_scale,
                        arrival_rate_per_s):
    """Frozen original implementation; keep arithmetic and iteration order."""
    allowed = [np.flatnonzero(selected[stage.stage_id]) for stage in service.stages]
    if any(len(nodes) == 0 for nodes in allowed):
        return float("inf")
    totals = 0.0
    for home in np.flatnonzero(demand):
        first = allowed[0]
        cost = (
            propagation[home, first]
            + service.input_mb_mean * task_data_scale * inverse_bandwidth[home, first]
            + service.stages[0].compute_gcycles_mean * task_compute_scale / node_capacities[first]
        )
        for stage_id in range(1, len(allowed)):
            previous, current = allowed[stage_id - 1], allowed[stage_id]
            transfer = (
                propagation[np.ix_(previous, current)]
                + service.stages[stage_id - 1].output_mb_mean * task_data_scale
                * inverse_bandwidth[np.ix_(previous, current)]
            )
            cost = np.min(cost[:, None] + transfer, axis=0)
            cost += service.stages[stage_id].compute_gcycles_mean * task_compute_scale / node_capacities[current]
        totals += float(demand[home]) * float(np.min(cost))
    mass = float(demand.sum())
    for stage in service.stages:
        pooled = float(node_capacities[allowed[stage.stage_id]].sum())
        totals += mass**2 * arrival_rate_per_s * stage.compute_gcycles_mean * task_compute_scale / max(pooled, 1e-9)
    return totals


class ScalarReferencePlacement(DemandAwareMILPPlacement):
    @staticmethod
    def _prepare_service_cost(*args):
        return SimpleNamespace(evaluate=lambda selected: scalar_service_cost(args[0], selected, *args[1:]))


@pytest.mark.parametrize("stage_count", [1, 2, 3, 4, 5])
def test_vectorized_cost_is_bit_exact_for_random_deployments(stage_count):
    rng = np.random.default_rng(2026 + stage_count)
    for case in range(40):
        nodes = 32
        service = Service(0, "differential", tuple(
            ServiceStage(0, stage, 0.3, 0.1, float(rng.uniform(0.01, 20)), float(rng.uniform(0.001, 2)))
            for stage in range(stage_count)
        ), float(rng.uniform(0.01, 2)), 1.0)
        demand = rng.uniform(0, 0.03, nodes)
        demand[rng.random(nodes) < 0.4] = 0
        if case == 0:
            demand[:] = 0
        capacities = rng.uniform(5, 200, nodes)
        propagation = rng.uniform(0, 0.05, (nodes, nodes))
        inverse_bandwidth = rng.uniform(0, 0.05, (nodes, nodes))
        if case == 1:
            propagation[0, :] = np.inf  # No reachable chain for this home.
            demand[0] = 0.01
        if case == 2:
            capacities[:] = 100.0  # Exact ties must retain the original choice.
            propagation[:] = 0.01
            inverse_bandwidth[:] = 0.01
        if case == 3:
            service = replace(service, input_mb_mean=0.0, stages=tuple(
                replace(stage, compute_gcycles_mean=0.0, output_mb_mean=0.0)
                for stage in service.stages
            ))
        args = (service, demand, capacities, propagation, inverse_bandwidth,
                float(rng.uniform(0.5, 2)), float(rng.uniform(0.5, 2)), float(rng.uniform(1, 100)))
        prepared = PreparedServiceCost(*args)
        # Reuse one cache with several different deployments, including no coverage.
        for replicas in (1, 4, 16, 32, 0):
            selected = np.zeros((stage_count, nodes), dtype=bool)
            for stage in range(stage_count):
                selected[stage, rng.choice(nodes, replicas, replace=False)] = True
            expected = scalar_service_cost(service, selected, *args[1:])
            assert prepared.evaluate(selected) == expected


@pytest.mark.parametrize("seed", [2026, 2027, 2028])
def test_full_replica_search_matches_scalar_and_refreshes_next_plan(seed):
    env = tiny_replay_env()
    env.reset()
    env.config.service_resource_fraction = 0.04  # Bind memory, not just all-node deployment.
    rng = np.random.default_rng(seed)
    optimized = DemandAwareMILPPlacement()
    original = ScalarReferencePlacement()
    for window in range(3):
        env.node_compute_load[:] = rng.uniform(0, 0.9, env.node_compute_load.shape)
        env.link_load[:] = rng.uniform(0, 0.9, env.link_load.shape)
        env.config.task_compute_scale = 0.7 + 0.3 * window
        env.config.task_data_scale = 1.2 - 0.2 * window
        env.config.demand_load_multiplier = 0.8 + 0.2 * window
        fast, fast_details = optimized.plan(env)
        slow, slow_details = original.plan(env)
        assert np.array_equal(fast, slow)
        for key in ("mandatory_objective", "forecast_latency_cost", "added_replicas", "replicas"):
            assert fast_details[key] == slow_details[key]
        feasible, reason = env.check_deployment_feasible(fast)
        assert feasible, reason
        env.apply_deployment(fast)


def test_equivalent_deployment_preserves_routing_and_settled_episode_metrics():
    reports = []
    for planner in (ScalarReferencePlacement(), DemandAwareMILPPlacement()):
        scheme = DPMSScheme()
        scheme.planner = planner
        result = evaluate_scheme_episode(
            scheme=scheme, env=tiny_replay_env(),
            point=ExperimentPoint("request_load", 1.0, "nominal"),
            eval_seed=2026, routing_repeat=0, sampled_seconds_per_window=6,
        )
        report = result.to_dict()
        for timing in ("planning_time_s", "scheduling_time_s", "kkt_settlement_time_s", "total_runtime_s"):
            report.pop(timing)
        reports.append(report)
    assert reports[0] == reports[1]
    assert reports[1]["invalid_action_rate"] == 0.0
