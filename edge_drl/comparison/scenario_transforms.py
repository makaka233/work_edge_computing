from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from edge_drl.comparison.trace import ComparisonTrace, rehash_trace
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.env.scenario import EdgeScenario, Service, TaskRequest


REQUEST_LOAD_VALUES = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
COMPUTE_VALUES = (0.6, 0.8, 1.0, 1.2, 1.4)
WIRED_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
INTERMEDIATE_DATA_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
HETEROGENEITY_VALUES = (1.0, 2.0, 4.0, 6.0, 8.0)


def formal_experiment_points() -> list[ExperimentPoint]:
    families = (
        ("request_load", REQUEST_LOAD_VALUES),
        ("compute_capacity", COMPUTE_VALUES),
        ("wired_bandwidth", WIRED_VALUES),
        ("intermediate_data", INTERMEDIATE_DATA_VALUES),
        ("stage_heterogeneity", HETEROGENEITY_VALUES),
    )
    return [ExperimentPoint(family, float(value), f"{family}:{value:g}") for family, values in families for value in values]


def transform_scenario(base: EdgeScenario, point: ExperimentPoint) -> EdgeScenario:
    scenario = deepcopy(base)
    eta = float(point.value)
    if point.family == "request_load":
        return scenario
    if point.family == "compute_capacity":
        scenario.nodes = [replace(node, compute_gcycles_per_s=node.compute_gcycles_per_s * eta) for node in scenario.nodes]
    elif point.family == "wired_bandwidth":
        finite = np.isfinite(scenario.bandwidth_mb_s)
        scenario.bandwidth_mb_s[finite] *= eta
    elif point.family == "intermediate_data":
        services: list[Service] = []
        for service in scenario.services:
            stages = tuple(
                replace(stage, output_mb_mean=stage.output_mb_mean * eta)
                if stage.stage_id < len(service.stages) - 1
                else stage
                for stage in service.stages
            )
            services.append(replace(service, stages=stages))
        scenario.services = services
    elif point.family == "stage_heterogeneity":
        scenario.services = [
            replace(service, stages=_redistribute_service_stages(service, eta))
            for service in scenario.services
        ]
    else:
        raise ValueError(f"unknown scenario family: {point.family}")
    return scenario


def transform_trace(trace: ComparisonTrace, point: ExperimentPoint) -> ComparisonTrace:
    if point.family not in {"intermediate_data", "stage_heterogeneity"}:
        return trace
    slots: list[tuple[TaskRequest, ...]] = []
    for slot in trace.slots:
        transformed: list[TaskRequest] = []
        for request in slot:
            if point.family == "intermediate_data":
                outputs = tuple(
                    value * point.value if index < len(request.stage_output_mb) - 1 else value
                    for index, value in enumerate(request.stage_output_mb)
                )
                transformed.append(replace(request, stage_output_mb=outputs))
            else:
                transformed.append(
                    replace(
                        request,
                        stage_compute_gcycles=_redistribute_values_exact(
                            request.stage_compute_gcycles, point.value
                        ),
                    )
                )
        slots.append(tuple(transformed))
    return rehash_trace(trace, tuple(slots))


def _rank_weights(count: int, heterogeneity: float) -> np.ndarray:
    if count == 2:
        return np.asarray([1.0, heterogeneity], dtype=np.float64)
    if count == 3:
        return np.asarray([1.0, np.sqrt(heterogeneity), heterogeneity], dtype=np.float64)
    return np.geomspace(1.0, heterogeneity, count, dtype=np.float64)


def _redistribute_values_exact(values: tuple[float, ...], heterogeneity: float) -> tuple[float, ...]:
    source = np.asarray(values, dtype=np.float64)
    weights = _rank_weights(len(source), heterogeneity)
    rank = np.argsort(source, kind="stable")
    assigned = np.empty_like(weights)
    assigned[rank] = weights
    total = float(source.sum())
    result = total * assigned / float(assigned.sum())
    if result.size:
        result[-1] += total - float(result.sum())
    return tuple(float(value) for value in result)


def _redistribute_service_stages(service: Service, heterogeneity: float):
    values = tuple(stage.compute_gcycles_mean for stage in service.stages)
    transformed = _redistribute_values_exact(values, heterogeneity)
    return tuple(
        replace(stage, compute_gcycles_mean=transformed[index])
        for index, stage in enumerate(service.stages)
    )
