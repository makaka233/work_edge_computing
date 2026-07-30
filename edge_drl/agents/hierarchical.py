from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest


@dataclass
class SlowGreedyDeploymentPolicy:
    replicas_per_stage: int = 5
    demand_weight: float = 0.75
    capacity_weight: float = 0.25
    broad_stage_replication: bool = True

    def act(self, env: EdgeComputingEnv) -> np.ndarray:
        """Create a feasible service-stage placement for the next 4-hour window."""

        env._require_ready()
        assert env.scenario is not None

        shape = (
            env.config.num_service_types,
            env.config.max_service_stages,
            env.config.num_edge_nodes,
        )
        deployment = np.zeros(shape, dtype=bool)
        remaining_memory = np.array([n.memory_gb for n in env.scenario.nodes], dtype=np.float64)
        remaining_storage = np.array([n.storage_gb for n in env.scenario.nodes], dtype=np.float64)

        node_demand = self._forecast_node_service_demand(env)
        node_capacity = np.array([n.compute_gcycles_per_s for n in env.scenario.nodes], dtype=np.float64)
        capacity_score = node_capacity / max(node_capacity.max(), 1e-9)

        for service in env.scenario.services:
            service_demand = node_demand[:, service.service_id]
            demand_score = service_demand / max(service_demand.max(), 1e-9)
            score = self.demand_weight * demand_score + self.capacity_weight * capacity_score
            ranked_nodes = np.argsort(-score)
            for stage in service.stages:
                replicas = 0
                replica_target = env.config.num_edge_nodes if self.broad_stage_replication else self.replicas_per_stage
                for node_id in ranked_nodes:
                    if remaining_memory[node_id] < stage.memory_gb or remaining_storage[node_id] < stage.storage_gb:
                        continue
                    deployment[service.service_id, stage.stage_id, node_id] = True
                    remaining_memory[node_id] -= stage.memory_gb
                    remaining_storage[node_id] -= stage.storage_gb
                    replicas += 1
                    if replicas >= replica_target:
                        break
                if replicas == 0:
                    self._force_place_smallest_feasible(
                        deployment,
                        remaining_memory,
                        remaining_storage,
                        service.service_id,
                        stage.stage_id,
                        stage.memory_gb,
                        stage.storage_gb,
                    )

        feasible, reason = env.check_deployment_feasible(deployment)
        if not feasible:
            raise RuntimeError(f"greedy deployment failed: {reason}")
        return deployment

    def _forecast_node_service_demand(self, env: EdgeComputingEnv) -> np.ndarray:
        assert env.scenario is not None
        demand = np.zeros((env.config.num_edge_nodes, env.config.num_service_types), dtype=np.float64)
        for user in env.scenario.users:
            demand[user.home_node] += np.array(user.service_weights)
        return demand

    def _force_place_smallest_feasible(
        self,
        deployment: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        service_id: int,
        stage_id: int,
        memory_gb: float,
        storage_gb: float,
    ) -> None:
        feasible = np.where((remaining_memory >= memory_gb) & (remaining_storage >= storage_gb))[0]
        if feasible.size == 0:
            raise RuntimeError("no feasible node remains for mandatory service replica")
        node_id = int(feasible[np.argmax(remaining_memory[feasible] + remaining_storage[feasible])])
        deployment[service_id, stage_id, node_id] = True
        remaining_memory[node_id] -= memory_gb
        remaining_storage[node_id] -= storage_gb


@dataclass
class FastGreedyScheduler:
    candidate_limit_per_stage: int = 8

    def act(self, env: EdgeComputingEnv, request: TaskRequest | None = None) -> list[int]:
        env._require_ready()
        assert env.scenario is not None
        if request is None:
            assert env.current_request is not None
            request = env.current_request

        mask = env.scheduler_candidate_mask(request)
        candidate_lists: list[list[int]] = []
        for stage_id in range(mask.shape[0]):
            candidates = np.where(mask[stage_id])[0]
            if candidates.size == 0:
                return [request.home_node] * mask.shape[0]
            ranked = sorted(
                candidates.tolist(),
                key=lambda node_id: self._stage_node_pre_score(env, request, stage_id, node_id),
            )
            candidate_lists.append(ranked[: self.candidate_limit_per_stage])

        best_valid_nodes: list[int] | None = None
        best_invalid_nodes: list[int] | None = None
        best_valid_score = float("inf")
        best_invalid_score = float("inf")
        best_score = float("inf")
        for nodes in product(*candidate_lists):
            info = env.evaluate_schedule(request, nodes)
            score = info["latency_s"] + 4.0 * (not info["valid"]) + 0.05 * info["load_penalty"]
            if info["valid"] and score < best_valid_score:
                best_valid_score = score
                best_valid_nodes = list(nodes)
            elif not info["valid"] and score < best_invalid_score:
                best_invalid_score = score
                best_invalid_nodes = list(nodes)
            best_score = min(best_score, score)

        if best_valid_nodes is not None:
            return best_valid_nodes

        valid_path = self._find_reachable_path(env, request, mask)
        if valid_path is not None:
            return valid_path

        colocated = self._find_colocated_fallback(mask)
        if colocated is not None:
            return colocated

        if best_invalid_nodes is not None:
            return best_invalid_nodes
        return [request.home_node] * mask.shape[0]

    def _stage_node_pre_score(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        node_id: int,
    ) -> float:
        assert env.scenario is not None
        node = env.scenario.nodes[node_id]
        compute_delay = request.stage_compute_gcycles[stage_id] / max(node.compute_gcycles_per_s, 1e-9)
        load_penalty = env.node_compute_load[node_id]
        if stage_id == 0 and node_id != request.home_node:
            if not env.scenario.adjacency[request.home_node, node_id]:
                return float("inf")
            link_delay = request.input_mb / max(env.scenario.bandwidth_mb_s[request.home_node, node_id], 1e-9)
        else:
            link_delay = 0.0
        return compute_delay + link_delay + 0.2 * load_penalty

    def _find_colocated_fallback(self, mask: np.ndarray) -> list[int] | None:
        shared = mask.all(axis=0)
        nodes = np.where(shared)[0]
        if nodes.size == 0:
            return None
        node_id = int(nodes[0])
        return [node_id] * mask.shape[0]

    def _find_reachable_path(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        mask: np.ndarray,
    ) -> list[int] | None:
        assert env.scenario is not None
        partial_paths: list[tuple[list[int], float]] = [([], 0.0)]
        for stage_id in range(mask.shape[0]):
            candidates = np.where(mask[stage_id])[0].tolist()
            candidates.sort(key=lambda node_id: self._stage_node_pre_score(env, request, stage_id, node_id))
            next_paths: list[tuple[list[int], float]] = []
            for path, score in partial_paths:
                prev_node = request.home_node if not path else path[-1]
                for node_id in candidates:
                    if prev_node != node_id and not env.scenario.adjacency[prev_node, node_id]:
                        continue
                    next_paths.append(
                        (
                            path + [int(node_id)],
                            score + self._stage_node_pre_score(env, request, stage_id, int(node_id)),
                        )
                    )
            if not next_paths:
                return None
            next_paths.sort(key=lambda item: item[1])
            partial_paths = next_paths[:64]
        return partial_paths[0][0] if partial_paths else None


@dataclass
class HierarchicalBaselineAgent:
    slow_policy: SlowGreedyDeploymentPolicy
    fast_policy: FastGreedyScheduler

    def maybe_update_deployment(self, env: EdgeComputingEnv) -> float:
        if not env.needs_deployment_update:
            return 0.0
        deployment = self.slow_policy.act(env)
        return env.apply_deployment(deployment)

    def act(self, env: EdgeComputingEnv) -> list[int]:
        self.maybe_update_deployment(env)
        return self.fast_policy.act(env)


def build_baseline_agent() -> HierarchicalBaselineAgent:
    return HierarchicalBaselineAgent(
        slow_policy=SlowGreedyDeploymentPolicy(),
        fast_policy=FastGreedyScheduler(),
    )
