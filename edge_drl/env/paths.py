from __future__ import annotations

from typing import Iterable

import numpy as np

PAD_NODE = -1


class PathManager:
    """Enumerate path actions and build masks for staged task scheduling."""

    def __init__(self, num_nodes: int, max_stages: int = 3):
        self.num_nodes = num_nodes
        self.max_stages = max_stages
        self.paths: list[tuple[int, int, int]] = []
        self.stage_counts: list[int] = []
        self._build_paths()

    @property
    def num_actions(self) -> int:
        return len(self.paths)

    def _build_paths(self) -> None:
        for e1 in range(self.num_nodes):
            self.paths.append((e1, PAD_NODE, PAD_NODE))
            self.stage_counts.append(1)
        for e1 in range(self.num_nodes):
            for e2 in range(self.num_nodes):
                self.paths.append((e1, e2, PAD_NODE))
                self.stage_counts.append(2)
        for e1 in range(self.num_nodes):
            for e2 in range(self.num_nodes):
                for e3 in range(self.num_nodes):
                    self.paths.append((e1, e2, e3))
                    self.stage_counts.append(3)

    def path(self, action_id: int) -> tuple[int, int, int]:
        return self.paths[int(action_id)]

    def mask(
        self,
        source_node: int,
        service_id: int,
        stage_count: int,
        deployment: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        for idx, path in enumerate(self.paths):
            if self.stage_counts[idx] != stage_count:
                continue
            if self._is_feasible_path(source_node, service_id, stage_count, path, deployment, adjacency):
                mask[idx] = True
        return mask

    def feasible_actions(
        self,
        source_node: int,
        service_id: int,
        stage_count: int,
        deployment: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        return np.flatnonzero(self.mask(source_node, service_id, stage_count, deployment, adjacency))

    def _is_feasible_path(
        self,
        source_node: int,
        service_id: int,
        stage_count: int,
        path: tuple[int, int, int],
        deployment: np.ndarray,
        adjacency: np.ndarray,
    ) -> bool:
        active = path[:stage_count]
        for j, node in enumerate(active):
            if node < 0 or deployment[service_id, j, node] <= 0:
                return False

        if active[0] != source_node and adjacency[source_node, active[0]] <= 0:
            return False

        for left, right in zip(active[:-1], active[1:]):
            if left != right and adjacency[left, right] <= 0:
                return False
        return True

    def iter_active_edges(
        self,
        source_node: int,
        path: tuple[int, int, int],
        stage_count: int,
    ) -> Iterable[tuple[int, int, int]]:
        active = path[:stage_count]
        if active[0] != source_node:
            yield 0, source_node, active[0]
        for j, (left, right) in enumerate(zip(active[:-1], active[1:]), start=1):
            if left != right:
                yield j, left, right

