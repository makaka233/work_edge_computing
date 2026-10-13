"""Plan-local, vectorized DPMS cost evaluation with unchanged arithmetic order."""
from __future__ import annotations

import numpy as np

from edge_drl.env.scenario import Service


DPMS_COST_IMPLEMENTATION = "plan-local-home-vectorized-v1"


class PreparedServiceCost:
    """Cache fixed data, not deployments, candidate sets or marginal gains.

    Node/link load, traffic and service transforms may change at each planning
    boundary. A fresh instance is therefore constructed for every service in
    every plan. All selected nodes remain in the exact serial-chain recurrence.
    """

    def __init__(self, service: Service, demand: np.ndarray,
                 node_capacities: np.ndarray, propagation: np.ndarray,
                 inverse_bandwidth: np.ndarray, task_compute_scale: float,
                 task_data_scale: float, arrival_rate_per_s: float):
        self.stage_ids = tuple(stage.stage_id for stage in service.stages)
        self.homes = np.flatnonzero(demand)
        self.demand = np.asarray(demand, dtype=np.float64).copy()
        self.capacities = np.asarray(node_capacities, dtype=np.float64).copy()
        self.compute = tuple(stage.compute_gcycles_mean * task_compute_scale / self.capacities
                             for stage in service.stages)
        self.ingress = (
            propagation[self.homes, :]
            + service.input_mb_mean * task_data_scale * inverse_bandwidth[self.homes, :]
            + self.compute[0][None, :]
        )
        self.transfers = tuple(
            propagation + service.stages[index - 1].output_mb_mean * task_data_scale * inverse_bandwidth
            for index in range(1, len(service.stages))
        )
        mass = float(self.demand.sum())
        self.pooled_numerators = tuple(
            mass**2 * arrival_rate_per_s * stage.compute_gcycles_mean * task_compute_scale
            for stage in service.stages
        )

    def evaluate(self, selected: np.ndarray) -> float:
        allowed = [np.flatnonzero(selected[index]) for index in self.stage_ids]
        if any(len(nodes) == 0 for nodes in allowed):
            return float("inf")
        cost = self.ingress[:, allowed[0]]
        for index in range(1, len(allowed)):
            previous, current = allowed[index - 1], allowed[index]
            transfer = self.transfers[index - 1][np.ix_(previous, current)]
            # Same per-home parent reduction and left-to-right stage additions
            # as the old scalar loop, just evaluated for all homes together.
            cost = np.min(cost[:, :, None] + transfer[None, :, :], axis=1)
            cost += self.compute[index][current]
        totals = 0.0
        # Keep the original sorted-home accumulation order. A dot product or
        # new summation order can perturb near-tied heap priorities by an ULP.
        for home, value in zip(self.homes, np.min(cost, axis=1)):
            totals += float(self.demand[home]) * float(value)
        for index, nodes in enumerate(allowed):
            pooled = float(self.capacities[nodes].sum())
            totals += self.pooled_numerators[index] / max(pooled, 1e-9)
        return totals
