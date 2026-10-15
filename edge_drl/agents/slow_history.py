"""Observed per-stage routes and settled costs from the PREVIOUS window."""
from __future__ import annotations

import numpy as np

HISTORY_NODE_DIM = 6
HISTORY_STAGE_DIM = 4


class SlowRouteHistory:
    def __init__(self, nodes):
        self.nodes = nodes
        self.rows = {}

    def add(self, service_id, stage_nodes, settlement, weight):
        if weight <= 0 or not settlement.get('valid',True):
            return
        for stage,node in enumerate(stage_nodes):
            if not 0 <= int(node) < self.nodes:
                raise ValueError('history observed an invalid node')
            key = f'{int(service_id)}:{stage}'
            row = self.rows.setdefault(key,np.zeros((self.nodes,3),dtype=np.float64))
            link = 'ingress' if stage == 0 else f'stage-{stage-1}'
            compute = float(settlement.get('compute_delays',{}).get(f'stage-{stage}',0.))
            transport = float(settlement.get('link_delays',{}).get(link,0.)) + float(settlement.get('logical_propagation_delays',{}).get(link,0.))
            row[int(node)] += np.asarray([1.,compute,transport])*weight

    def snapshot(self):
        return {key:value.tolist() for key,value in self.rows.items()}


def history_features(history, nodes, services, stages, service_id=None, stage_id=None):
    node_values = np.zeros((nodes,HISTORY_NODE_DIM))
    stage_values = np.zeros((services*stages,HISTORY_STAGE_DIM))
    routes = {}
    for key,value in history.items():
        service,stage = map(int,key.split(':'))
        if not 0 <= service < services or not 0 <= stage < stages:
            raise ValueError('history service/stage differs from observation')
        row = np.asarray(value,dtype=np.float64)
        if row.shape != (nodes,3) or not np.isfinite(row).all() or np.any(row < 0):
            raise ValueError('invalid previous-window route history')
        weights = row[:,0]
        mass = float(weights.sum())
        if mass <= 0:
            continue
        routes[(service,stage)] = weights/mass
        stage_values[service*stages+stage] = [np.log1p(1000*row[:,1].sum()/mass),
            np.log1p(1000*row[:,2].sum()/mass),weights.max()/mass,1.]
        if (service,stage) == (service_id,stage_id):
            costs = np.divide(row[:,1:],weights[:,None],out=np.zeros((nodes,2)),where=weights[:,None]>0)
            node_values[:,0] = weights/mass
            node_values[:,3:5] = np.log1p(1000*costs)
            node_values[:,5] = weights > 0
    if service_id is not None and stage_id is not None:
        node_values[:,1] = routes.get((service_id,stage_id-1),np.zeros(nodes))
        node_values[:,2] = routes.get((service_id,stage_id+1),np.zeros(nodes))
    return node_values,stage_values
