from __future__ import annotations

import numpy as np

from edge_drl.env.paths import PathManager
from edge_drl.env.requests import Task


class RandomScheduler:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_path(self, task: Task, mask: np.ndarray, obs: np.ndarray | None = None) -> int:
        actions = np.flatnonzero(mask)
        if actions.size == 0:
            raise RuntimeError("No feasible scheduling path is available.")
        return int(self.rng.choice(actions))


class HeuristicScheduler:
    """Pick a feasible path with a simple delay and congestion estimate."""

    def __init__(self, path_manager: PathManager, compute_capacity: np.ndarray, bandwidth: np.ndarray):
        self.path_manager = path_manager
        self.compute_capacity = compute_capacity
        self.bandwidth = bandwidth

    def select_path(
        self,
        task: Task,
        mask: np.ndarray,
        node_pressure: np.ndarray,
        link_pressure: np.ndarray,
        pending_node_load: np.ndarray,
        pending_link_load: np.ndarray,
    ) -> int:
        best_action = None
        best_score = float("inf")
        for action in np.flatnonzero(mask):
            path = self.path_manager.path(int(action))
            score = self._score(task, path, node_pressure, link_pressure, pending_node_load, pending_link_load)
            if score < best_score:
                best_score = score
                best_action = int(action)
        if best_action is None:
            raise RuntimeError("No feasible scheduling path is available.")
        return best_action

    def _score(
        self,
        task: Task,
        path: tuple[int, int, int],
        node_pressure: np.ndarray,
        link_pressure: np.ndarray,
        pending_node_load: np.ndarray,
        pending_link_load: np.ndarray,
    ) -> float:
        active = path[: task.stage_count]
        score = 0.0
        for j, node in enumerate(active):
            capacity = max(float(self.compute_capacity[node]), 1e-6)
            pressure = float(node_pressure[node]) + float(pending_node_load[node]) / capacity
            score += float(task.compute_gcycles[j]) / capacity * (1.0 + pressure)

        if active[0] != task.source_node:
            score += self._link_score(task.input_mb, task.source_node, active[0], link_pressure, pending_link_load)
        for j, (left, right) in enumerate(zip(active[:-1], active[1:])):
            if left != right:
                score += self._link_score(task.output_mb[j], left, right, link_pressure, pending_link_load)
        return score

    def _link_score(
        self,
        data_mb: float,
        left: int,
        right: int,
        link_pressure: np.ndarray,
        pending_link_load: np.ndarray,
    ) -> float:
        capacity = max(float(self.bandwidth[left, right]), 1e-6)
        pressure = float(link_pressure[left, right]) + float(pending_link_load[left, right]) / capacity
        return float(data_mb) / capacity * (1.0 + pressure)

