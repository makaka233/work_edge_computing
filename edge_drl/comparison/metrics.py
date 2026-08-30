from __future__ import annotations

import time

import numpy as np

from edge_drl.comparison.replay_env import TraceReplayEnv
from edge_drl.comparison.types import EpisodeResult, ExperimentPoint, deployment_replica_count


def evaluate_scheme_episode(
    *,
    scheme,
    env: TraceReplayEnv,
    point: ExperimentPoint,
    eval_seed: int,
    routing_repeat: int,
) -> EpisodeResult:
    env.reset()
    latencies: list[float] = []
    request_weights: list[float] = []
    components = {key: 0.0 for key in ("compute", "link", "access", "propagation")}
    deadline_violations = 0.0
    invalid = 0.0
    total_requests = 0.0
    total_latency = 0.0
    replica_sum = 0.0
    used_node_sum = 0.0
    used_replica_rate_sum = 0.0
    transition_total = 0.0
    cross_node_total = 0.0
    node_load_sum = 0.0
    node_load_max = 0.0
    link_load_sum = 0.0
    link_load_max = 0.0
    memory_utilization_sum = 0.0
    storage_utilization_sum = 0.0
    planning_time = 0.0
    scheduling_time = 0.0
    settlement_time = 0.0
    wall_start = time.perf_counter()

    for _ in range(env._comparison_trace.logical_steps):
        planning_start = time.perf_counter()
        scheme.maybe_plan(env)
        planning_time += time.perf_counter() - planning_start
        original = list(env.current_requests)
        requests = scheme.adapt_requests(original)
        env.current_requests = requests
        env.current_request = requests[0] if requests else None
        scheduling_start = time.perf_counter()
        schedules = scheme.schedule_batch(env, requests)
        scheduling_time += time.perf_counter() - scheduling_start
        settlement_start = time.perf_counter()
        _, _, _, info = env.step(schedules, represented_seconds=1.0)
        settlement_time += time.perf_counter() - settlement_start

        slot_used_nodes: set[int] = set()
        slot_used_replicas: set[tuple[int, int, int]] = set()
        for request, schedule, group_info in zip(requests, schedules, info["group_infos"]):
            weight = float(request.request_count)
            latency = float(group_info["latency_s"])
            latencies.append(latency)
            request_weights.append(weight)
            total_requests += weight
            total_latency += latency * weight
            components["compute"] += float(group_info["compute_delay_s"]) * weight
            components["link"] += float(group_info["link_delay_s"]) * weight
            components["access"] += float(group_info["access_delay_s"]) * weight
            components["propagation"] += float(group_info["propagation_delay_s"]) * weight
            deadline_violations += float(latency > request.deadline_s) * weight
            invalid += float(not group_info["valid"]) * weight
            slot_used_nodes.update(int(node) for node in schedule)
            slot_used_replicas.update(
                (request.service_id, stage_id, int(node))
                for stage_id, node in enumerate(schedule)
            )
            for left, right in zip(schedule, schedule[1:]):
                transition_total += weight
                cross_node_total += float(left != right) * weight
        current_replicas = deployment_replica_count(env)
        replica_sum += current_replicas
        used_node_sum += len(slot_used_nodes)
        used_replica_rate_sum += len(slot_used_replicas) / max(current_replicas, 1)
        memory_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        storage_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        assert env.scenario is not None and env.deployment is not None
        for service in env.scenario.services:
            for stage in service.stages:
                placed = env.deployment[service.service_id, stage.stage_id]
                memory_used += placed * stage.memory_gb
                storage_used += placed * stage.storage_gb
        memory_utilization_sum += float(
            np.mean(memory_used / np.maximum(env.service_memory_capacities(), 1e-12))
        )
        storage_utilization_sum += float(
            np.mean(storage_used / np.maximum(env.service_storage_capacities(), 1e-12))
        )
        node_load_sum += float(np.mean(env.node_compute_load))
        node_load_max = max(node_load_max, float(np.max(env.node_compute_load)))
        finite_links = env.link_load[np.isfinite(env.link_load)]
        link_load_sum += float(np.mean(finite_links)) if finite_links.size else 0.0
        link_load_max = max(link_load_max, float(np.max(finite_links))) if finite_links.size else link_load_max

    logical_steps = env._comparison_trace.logical_steps
    mean_latency = total_latency / max(total_requests, 1.0)
    p95 = _weighted_percentile(latencies, request_weights, 95.0)
    assert env.scenario is not None
    stage_count = sum(len(service.stages) for service in env.scenario.services)
    return EpisodeResult(
        scheme=scheme.name,
        scenario_family=point.family,
        scenario_value=point.value,
        scenario_label=point.label,
        eval_seed=int(eval_seed),
        routing_repeat=int(routing_repeat),
        trace_hash=env._comparison_trace.trace_hash,
        logical_steps=logical_steps,
        settlement_steps=int(env.metrics["settlement_steps"]),
        request_count=int(total_requests),
        mean_latency_ms=mean_latency * 1000.0,
        p95_latency_ms=p95 * 1000.0,
        episode_total_latency_s=total_latency,
        mean_slot_total_latency_s=total_latency / logical_steps,
        deadline_violation_rate=deadline_violations / max(total_requests, 1.0),
        invalid_action_rate=invalid / max(total_requests, 1.0),
        mean_compute_delay_ms=components["compute"] / max(total_requests, 1.0) * 1000.0,
        mean_link_delay_ms=components["link"] / max(total_requests, 1.0) * 1000.0,
        mean_access_delay_ms=components["access"] / max(total_requests, 1.0) * 1000.0,
        mean_propagation_delay_ms=components["propagation"] / max(total_requests, 1.0) * 1000.0,
        mean_replicas=replica_sum / logical_steps,
        avg_replicas_per_stage=replica_sum / logical_steps / max(stage_count, 1),
        mean_used_nodes=used_node_sum / logical_steps,
        used_replica_rate=used_replica_rate_sum / logical_steps,
        cross_node_transition_rate=cross_node_total / max(transition_total, 1.0),
        mean_deployment_memory_utilization=memory_utilization_sum / logical_steps,
        mean_deployment_storage_utilization=storage_utilization_sum / logical_steps,
        mean_node_load=node_load_sum / logical_steps,
        max_node_load=node_load_max,
        mean_link_load=link_load_sum / logical_steps,
        max_link_load=link_load_max,
        planning_time_s=planning_time,
        scheduling_time_s=scheduling_time,
        kkt_settlement_time_s=settlement_time,
        total_runtime_s=time.perf_counter() - wall_start,
        failed=invalid > 0.0,
        failure_reason="invalid_action_rate > 0" if invalid > 0.0 else "",
    )


def _weighted_percentile(values: list[float], weights: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    order = np.argsort(array)
    array = array[order]
    weight = weight[order]
    cutoff = percentile / 100.0 * float(weight.sum())
    index = int(np.searchsorted(np.cumsum(weight), cutoff, side="left"))
    return float(array[min(index, len(array) - 1)])
