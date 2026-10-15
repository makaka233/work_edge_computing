"""Value-only graph encoder with no shared actor parameters or features."""
from __future__ import annotations

import math

import torch
from torch import nn


class StructuredSlowCritic(nn.Module):
    def __init__(self, *, global_dim, node_feature_dim, edge_feature_dim, num_nodes,
                 stage_feature_dim=0, stage_count=0, hidden_dim=64):
        super().__init__()
        self.global_dim, self.node_dim, self.edge_dim = global_dim, node_feature_dim, edge_feature_dim
        self.num_nodes, self.stage_dim, self.stage_count = num_nodes, stage_feature_dim, stage_count
        self.node_offset = global_dim
        self.stage_offset = global_dim + num_nodes * node_feature_dim
        self.edge_offset = self.stage_offset + stage_count * stage_feature_dim
        self.global_encoder = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.Tanh())
        self.node_encoder = nn.Sequential(nn.Linear(node_feature_dim, hidden_dim), nn.Tanh())
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Sequential(nn.Linear(edge_feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.node_update = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.Tanh())
        self.stage_encoder = nn.Sequential(nn.Linear(stage_feature_dim, hidden_dim), nn.Tanh()) if stage_count else None
        self.readout = nn.Sequential(nn.Linear(hidden_dim * (3 if stage_count else 2), hidden_dim), nn.Tanh())
        self.output = nn.Linear(hidden_dim, 1)

    def __getitem__(self, index):
        # PPO target normalization rescales only the final affine layer.
        if index != -1:
            raise IndexError(index)
        return self.output

    def forward(self, states):
        global_emb = self.global_encoder(states[:, :self.global_dim])
        nodes = states[:, self.node_offset:self.stage_offset].reshape(-1, self.num_nodes, self.node_dim)
        edges = states[:, self.edge_offset:].reshape(-1, self.num_nodes, self.num_nodes, self.edge_dim)
        node_emb = self.node_encoder(nodes)
        scores = torch.matmul(self.query(node_emb), self.key(node_emb).transpose(1, 2)) / math.sqrt(node_emb.shape[-1])
        scores = scores + self.edge_bias(edges).squeeze(-1)
        connected = (edges[..., 0] > .5) | torch.eye(self.num_nodes, dtype=torch.bool, device=states.device)[None]
        messages = torch.softmax(scores.masked_fill(~connected, -1e9), dim=-1) @ self.message(node_emb)
        graph = self.node_update(torch.cat([node_emb, messages], dim=-1))
        # Global query conditions aggregation on the current head/stage/quota.
        node_weights = torch.softmax((graph * global_emb[:, None]).sum(-1) / math.sqrt(graph.shape[-1]), dim=-1)
        pooled = (graph * node_weights[..., None]).sum(1)
        pieces = [global_emb, pooled]
        if self.stage_count:
            raw = states[:, self.stage_offset:self.edge_offset].reshape(-1, self.stage_count, self.stage_dim)
            stages = self.stage_encoder(raw)
            valid = raw[..., 0] > .5
            weights = valid.to(stages.dtype)
            pieces.append((stages * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1))
        return self.output(self.readout(torch.cat(pieces, dim=-1)))
