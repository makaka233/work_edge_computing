from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge_drl.env.paths import PAD_NODE
from edge_drl.env.requests import Task


@dataclass(slots=True)
class ScheduledTask:
    task: Task
    path: tuple[int, int, int]


@dataclass(slots=True)
class KKTResult:
    total_delay: float
    compute_delay: float
    transmission_delay: float
    average_delay: float
    node_compute_load: np.ndarray
    link_data_load: np.ndarray
    node_active_counts: np.ndarray
    link_active_counts: np.ndarray
    invalid_count: int = 0


class KKTAllocator:
    """Analytical compute and bandwidth allocation under fixed x/y decisions."""

    def __init__(self, compute_capacity: np.ndarray, bandwidth_capacity: np.ndarray):
        self.compute_capacity = compute_capacity.astype(np.float64)
        self.bandwidth_capacity = bandwidth_capacity.astype(np.float64)
        self.num_nodes = int(compute_capacity.shape[0])

    def allocate(self, scheduled: list[ScheduledTask]) -> KKTResult:
        compute_items: list[list[tuple[int, float]]] = [[] for _ in range(self.num_nodes)]
        link_items: dict[tuple[int, int], list[float]] = {}
        node_load = np.zeros(self.num_nodes, dtype=np.float64)
        link_load = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        node_counts = np.zeros(self.num_nodes, dtype=np.int64)
        link_counts = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int64)
        invalid = 0

        for item in scheduled:
            task = item.task
            path = item.path
            active = path[: task.stage_count]
            if any(node == PAD_NODE for node in active):
                invalid += 1
                continue
            for j, node in enumerate(active):
                c = max(float(task.compute_gcycles[j]), 1e-9)
                compute_items[node].append((task.task_id, c))
                node_load[node] += c
                node_counts[node] += 1

            first = active[0]
            if first != task.source_node:
                d = max(float(task.input_mb), 1e-9)
                key = (task.source_node, first)
                link_items.setdefault(key, []).append(d)
                link_load[key] += d
                link_counts[key] += 1

            for j, (left, right) in enumerate(zip(active[:-1], active[1:])):
                if left == right:
                    continue
                d = max(float(task.output_mb[j]), 1e-9)
                key = (left, right)
                link_items.setdefault(key, []).append(d)
                link_load[key] += d
                link_counts[key] += 1

        compute_delay = self._sum_kkt_delay_by_group(
            [np.array([c for _, c in group], dtype=np.float64) for group in compute_items],
            self.compute_capacity,
        )
        transmission_delay = self._sum_link_delay(link_items)
        total = compute_delay + transmission_delay
        avg = total / max(len(scheduled), 1)
        return KKTResult(
            total_delay=total,
            compute_delay=compute_delay,
            transmission_delay=transmission_delay,
            average_delay=avg,
            node_compute_load=node_load,
            link_data_load=link_load,
            node_active_counts=node_counts,
            link_active_counts=link_counts,
            invalid_count=invalid,
        )

    @staticmethod
    def _sum_kkt_delay_by_group(groups: list[np.ndarray], capacities: np.ndarray) -> float:
        total = 0.0
        for values, capacity in zip(groups, capacities):
            if values.size == 0:
                continue
            if capacity <= 0:
                total += 1e9
                continue
            root_sum = float(np.sqrt(np.maximum(values, 1e-12)).sum())
            total += (root_sum**2) / float(capacity)
        return total

    def _sum_link_delay(self, link_items: dict[tuple[int, int], list[float]]) -> float:
        total = 0.0
        for (left, right), values in link_items.items():
            capacity = float(self.bandwidth_capacity[left, right])
            if capacity <= 0:
                total += 1e9
                continue
            data = np.asarray(values, dtype=np.float64)
            root_sum = float(np.sqrt(np.maximum(data, 1e-12)).sum())
            total += (root_sum**2) / capacity
        return total

