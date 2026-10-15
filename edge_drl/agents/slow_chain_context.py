"""Opt-in causal chain/port/resource observations for learned Slow actions.

These are capacity-weighted approximations, not Fast rollouts, KKT solutions,
action labels or replacement rewards. Every physical candidate remains open.
"""
from __future__ import annotations

import numpy as np

from edge_drl.agents.slow_context import SlowDeploymentContext
from edge_drl.agents.slow_history import history_features


SLOW_CHAIN_CONTEXT = "deployment-chain-v3"
CHAIN_NODE_DIM = 8
CHAIN_STAGE_DIM = 4


def signed_log_ms(values):
    return np.sign(values) * np.log1p(1000 * np.abs(values))


class SlowChainContext(SlowDeploymentContext):
    def __init__(self, env, demand, competition_aware=True, history=None):
        super().__init__(env, demand, competition_aware=competition_aware)
        self.history = history
        self.stage_lengths = {s.service_id: len(s.stages) for s in env.scenario.services}
        bandwidth = np.where(np.isfinite(env.scenario.bandwidth_mb_s), env.scenario.bandwidth_mb_s, 0.)
        offered = np.maximum(np.where(np.isfinite(env.link_load), env.link_load, 0.), 0.) * bandwidth
        nic = env.node_nic_capacities()
        finite = np.isfinite(nic)
        self.port_headroom = np.ones((self.nodes, 2))
        for column, offered_rate in enumerate((offered.sum(axis=1), offered.sum(axis=0))):
            self.port_headroom[finite, column] = np.clip(
                1 - offered_rate[finite] / np.maximum(nic[finite], 1e-9), 0., 1.)

    def _transport(self, volume):
        # An unreachable path is expensive but finite in an observation.
        cost = self.propagation + volume * self.inverse_bandwidth
        return np.where(np.isfinite(cost), cost, 1e3)

    def _chain_costs(self, deployment, service_id, stage_id):
        i = service_id * self.stages + stage_id
        home = self.home_weights[:, service_id]
        sources = home
        if stage_id > 0:
            previous = deployment[service_id, stage_id - 1] * self.capacities
            if previous.sum() > 0:
                sources = previous / previous.sum()
        incoming = self._transport(self.incoming[i])
        upstream = sources @ incoming
        home_cost = home @ incoming
        downstream = np.zeros(self.nodes)
        if stage_id + 1 < self.stage_lengths[service_id]:
            following = deployment[service_id, stage_id + 1] * self.capacities
            if following.sum() > 0:
                downstream = self._transport(self.outgoing[i]) @ (following / following.sum())
        return upstream, downstream, home_cost

    def _opportunity_loss(self, deployment, i):
        selected = deployment.reshape(self.size, self.nodes)
        memory = self.memory_capacity - selected.T @ self.memory
        storage = self.storage_capacity - selected.T @ self.storage
        feasible = (~selected & self.valid[:, None]
                    & (memory[None, :] + 1e-9 >= self.memory[:, None])
                    & (storage[None, :] + 1e-9 >= self.storage[:, None]))
        after = ((memory[None, :] - self.memory[i] + 1e-9 >= self.memory[:, None])
                 & (storage[None, :] - self.storage[i] + 1e-9 >= self.storage[:, None]))
        lost = feasible & ~after
        lost[i] = False
        return lost.sum(axis=0) / max(int(self.valid.sum()) - 1, 1)

    def features(self, deployment, *, budget, service_id=None, stage_id=None):
        global_features, node_features = super().features(
            deployment, budget=budget, service_id=service_id, stage_id=stage_id)
        extra = np.zeros((self.nodes, CHAIN_NODE_DIM))
        # Port history is available even before coverage and Stage decisions.
        extra[:, 4:6] = self.port_headroom
        if service_id is not None and stage_id is not None:
            i = service_id * self.stages + stage_id
            upstream, downstream, home = self._chain_costs(deployment, service_id, stage_id)
            _, pooled, proxy = self.inventory(deployment)
            gain = self.candidate_gains(deployment, pooled, proxy, clip=False)[i]
            gain[deployment[service_id, stage_id]] = 0.
            roots = self._competition(deployment, pooled)[0][i]
            extra[:, :4] = np.column_stack([
                np.log1p(1000 * upstream), np.log1p(1000 * downstream),
                np.log1p(1000 * home), signed_log_ms(gain)])
            extra[:, 6] = self._opportunity_loss(deployment, i)
            # Approximate work pressure in this serial phase, not a node mask.
            extra[:, 7] = np.log1p(roots / np.sqrt(np.maximum(self.capacities, 1e-9)))
        combined = np.c_[node_features,extra]
        if self.history is not None:
            past,_ = history_features(self.history,self.nodes,self.services,self.stages,service_id,stage_id)
            combined = np.c_[combined,past]
        return global_features,combined

    def stage_costs(self, deployment, remaining_memory, remaining_storage):
        existing = super().stage_costs(deployment, remaining_memory, remaining_storage)
        extra = np.zeros((self.size, CHAIN_STAGE_DIM))
        selected = deployment.reshape(self.size, self.nodes)
        feasible = (~selected & self.valid[:, None]
                    & (remaining_memory[None, :] + 1e-9 >= self.memory[:, None])
                    & (remaining_storage[None, :] + 1e-9 >= self.storage[:, None]))
        _, pooled, proxy = self.inventory(deployment)
        gains = self.candidate_gains(deployment, pooled, proxy, clip=False)
        for i in np.flatnonzero(self.valid):
            allowed = feasible[i]
            if not allowed.any():
                continue
            service_id, stage_id = divmod(int(i), self.stages)
            upstream, downstream, _ = self._chain_costs(deployment, service_id, stage_id)
            costs = upstream + downstream
            extra[i] = [np.log1p(1000 * costs[allowed].min()),
                        np.log1p(1000 * costs[allowed].mean()),
                        signed_log_ms(gains[i, allowed].max()),
                        self._opportunity_loss(deployment, i)[allowed].mean()]
        combined = np.c_[existing,extra]
        if self.history is not None:
            _,past = history_features(self.history,self.nodes,self.services,self.stages)
            combined = np.c_[combined,past]
        return combined
