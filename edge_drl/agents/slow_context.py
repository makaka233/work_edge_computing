"""Deployment observations, never action selection or replacement rewards.

The pooled compute term is a demand-based surrogate, not a certified joint
network/KKT solution. Every feature is available BEFORE the recorded action.
"""
from __future__ import annotations

import numpy as np

from edge_drl.agents.rollout_features import route_plan

SLOW_DEPLOYMENT_CONTEXT = "deployment-aware-v2"
INVENTORY_DIM = 7
BUDGET_DIM = 6
CURRENT_STAGE_DIM = 8
CANDIDATE_DIM = 5
STAGE_COST_DIM = 6


class SlowDeploymentContext:
    """Fixed demand/network data for ONE planning call; deployment stays live."""

    def __init__(self, env, demand):
        self.nodes = env.config.num_edge_nodes
        self.services = env.config.num_service_types
        self.stages = env.config.max_service_stages
        self.size = self.services * self.stages
        self.memory_capacity = np.maximum(env.service_memory_capacities(), 1e-9)
        self.storage_capacity = np.maximum(env.service_storage_capacities(), 1e-9)
        self.capacities = np.asarray([n.compute_gcycles_per_s for n in env.scenario.nodes])
        self.capacities *= np.clip(1.0 - .75 * env.node_compute_load, .10, 1.0)
        self.total_compute = max(float(self.capacities.sum()), 1e-9)
        self.valid = np.zeros(self.size, dtype=bool)
        self.compute = np.zeros(self.size)
        self.memory = np.zeros(self.size)
        self.storage = np.zeros(self.size)
        self.incoming = np.zeros(self.size)
        self.outgoing = np.zeros(self.size)
        # Slow's supplied demand is log1p(requests/min)/log1p(500), NOT a
        # probability matrix. Physical proxies must use original user weights.
        physical_demand = np.zeros((self.nodes, self.services), dtype=np.float64)
        for user in env.scenario.users:
            physical_demand[user.home_node] += user.service_weights
        physical_demand /= max(len(env.scenario.users), 1)
        self.mass = np.repeat(physical_demand.sum(axis=0), self.stages)
        self.arrival = env._arrival_rate_per_minute() / 60.0
        self.home_weights = physical_demand / np.maximum(physical_demand.sum(axis=0), 1e-9)
        for service in env.scenario.services:
            for stage in service.stages:
                i = service.service_id * self.stages + stage.stage_id
                self.valid[i] = True
                self.compute[i] = stage.compute_gcycles_mean * env.config.task_compute_scale
                self.memory[i], self.storage[i] = stage.memory_gb, stage.storage_gb
                self.incoming[i] = (service.input_mb_mean if stage.stage_id == 0
                                    else service.stages[stage.stage_id - 1].output_mb_mean) * env.config.task_data_scale
                self.outgoing[i] = stage.output_mb_mean * env.config.task_data_scale
        # Existing router over all physical node pairs, not execution path candidates.
        routes = route_plan(env)
        bandwidth = env.scenario.bandwidth_mb_s * np.clip(1 - .75 * env.link_load, .1, 1)
        inverse = np.zeros_like(bandwidth)
        np.divide(1, bandwidth, out=inverse, where=np.isfinite(bandwidth) & (bandwidth > 0))
        self.propagation = routes.propagation
        self.inverse_bandwidth = routes.sum_edges(inverse)
        self.numerators = self.mass**2 * self.arrival * self.compute

    def inventory(self, deployment):
        selected = deployment.reshape(self.size, self.nodes)
        pooled = selected @ self.capacities
        proxy = np.divide(self.numerators, pooled, out=np.zeros(self.size), where=pooled > 0)
        values = np.column_stack([
            self.valid, selected.sum(axis=1) / self.nodes, pooled / self.total_compute,
            np.log1p(self.arrival * self.mass * self.compute), np.log1p(1000 * proxy),
            selected.sum(axis=1) * self.memory / self.memory_capacity.sum(),
            selected.sum(axis=1) * self.storage / self.storage_capacity.sum(),
        ])
        return values, pooled, proxy

    def stage_costs(self, deployment, remaining_memory, remaining_storage):
        selected = deployment.reshape(self.size, self.nodes)
        _, pooled, proxy = self.inventory(deployment)
        feasible = (~selected & (remaining_memory[None, :] + 1e-9 >= self.memory[:, None])
                    & (remaining_storage[None, :] + 1e-9 >= self.storage[:, None]) & self.valid[:, None])
        after = self.numerators[:, None] / np.maximum(pooled[:, None] + self.capacities, 1e-9)
        gain = np.maximum(proxy[:, None] - after, 0)
        gain = np.where(feasible, gain, 0)
        footprint = self.memory[:, None] / self.memory_capacity + self.storage[:, None] / self.storage_capacity
        values = np.column_stack([
            pooled / self.total_compute, np.log1p(self.arrival * self.mass * self.compute),
            np.log1p(1000 * proxy), np.log1p(1000 * gain.max(axis=1)),
            np.log1p(1000 * np.max(gain / np.maximum(footprint, 1e-9), axis=1)),
            (feasible @ self.capacities) / self.total_compute,
        ])
        return values

    def features(self, deployment, *, budget, service_id=None, stage_id=None):
        inventory, pooled, proxy = self.inventory(deployment)
        current = np.zeros(CURRENT_STAGE_DIM)
        candidates = np.zeros((self.nodes, CANDIDATE_DIM))
        if service_id is not None and stage_id is not None:
            i = service_id * self.stages + stage_id
            current[:] = [self.mass[i], np.log1p(self.compute[i]),
                          self.memory[i] / self.memory_capacity.sum(),
                          self.storage[i] / self.storage_capacity.sum(),
                          np.log1p(self.incoming[i]), np.log1p(self.outgoing[i]),
                          pooled[i] / self.total_compute, np.log1p(1000 * proxy[i])]
            selected = deployment[service_id, stage_id]
            # Ineligible nodes are still observable; action feasibility comes ONLY from masks.
            gain = np.maximum(proxy[i] - self.numerators[i] / np.maximum(pooled[i] + self.capacities, 1e-9), 0)
            gain[selected] = 0
            sources = self.home_weights[:, service_id]
            if stage_id > 0:
                previous = deployment[service_id, stage_id - 1] * self.capacities
                if previous.sum() > 0:
                    sources = previous / previous.sum()
            active = sources > 0
            transport = self.propagation + self.incoming[i] * self.inverse_bandwidth
            estimate = sources[active] @ transport[active] if active.any() else np.zeros(self.nodes)
            # Keep unreachable costs finite in an observation without making them legal.
            estimate = np.where(np.isfinite(estimate), estimate, 1e3)
            next_nodes = (deployment[service_id, stage_id + 1] if stage_id + 1 < self.stages
                          else np.zeros(self.nodes))
            candidates[:] = np.column_stack([selected, next_nodes, np.log1p(1000 * gain),
                                              np.log1p(1000 * estimate),
                                              self.capacities / self.total_compute])
        global_features = np.r_[inventory.reshape(-1), np.asarray(budget), current]
        node_features = np.c_[deployment.reshape(self.size, self.nodes).T, candidates]
        return global_features, node_features
