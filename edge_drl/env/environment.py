from __future__ import annotations

from typing import Any

import numpy as np

from edge_drl.env.paths import PathManager
from edge_drl.env.requests import DynamicRequestGenerator, Task
from edge_drl.solver.kkt import KKTAllocator, KKTResult, ScheduledTask


class EdgeComputingEnv:
    """Second-level simulator with event-driven task scheduling."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rng = np.random.default_rng(int(config["simulation"]["seed"]))
        self.nodes = config["nodes"]
        self.services = config["services"]
        self.num_nodes = len(self.nodes)
        self.num_services = len(self.services)
        self.max_stages = 3
        self.bandwidth = np.asarray(config["bandwidth_mb_per_s"], dtype=np.float64)
        self.adjacency = (self.bandwidth > 0).astype(np.int64)
        self.compute_capacity = np.array([n["compute_gcycles_per_s"] for n in self.nodes], dtype=np.float64)
        self.memory_capacity = np.array([n["memory_gb"] for n in self.nodes], dtype=np.float64)
        self.storage_capacity = np.array([n["storage_gb"] for n in self.nodes], dtype=np.float64)
        self.request_gen = DynamicRequestGenerator(config, self.rng)
        self.path_manager = PathManager(self.num_nodes, self.max_stages)
        self.kkt = KKTAllocator(self.compute_capacity, self.bandwidth)
        self.time_s = 0
        self.done = False
        self.deployment = np.zeros((self.num_services, self.max_stages, self.num_nodes), dtype=np.int64)
        self.last_node_compute_load = np.zeros(self.num_nodes, dtype=np.float64)
        self.last_link_data_load = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        self.last_request_counts = np.zeros((self.num_nodes, self.num_services), dtype=np.float64)
        self.last_result: KKTResult | None = None
        self._service_stage_resources = self._build_stage_resources()

    @property
    def state_dim(self) -> int:
        return int(self.observe().shape[0])

    @property
    def task_obs_dim(self) -> int:
        return (
            self.num_nodes
            + self.num_services
            + 1
            + 7
            + self.num_services * self.max_stages * self.num_nodes
            + self.num_nodes
            + self.num_nodes * self.num_nodes
            + self.num_nodes
            + self.num_nodes * self.num_nodes
        )

    @property
    def service_stage_mask(self) -> np.ndarray:
        mask = np.zeros((self.num_services, self.max_stages), dtype=bool)
        for i, svc in enumerate(self.services):
            mask[i, : len(svc["stages"])] = True
        return mask

    @property
    def stage_memory(self) -> np.ndarray:
        return self._service_stage_resources["memory"].copy()

    @property
    def stage_storage(self) -> np.ndarray:
        return self._service_stage_resources["storage"].copy()

    def reset(self) -> np.ndarray:
        self.rng = np.random.default_rng(int(self.config["simulation"]["seed"]))
        self.request_gen = DynamicRequestGenerator(self.config, self.rng)
        self.time_s = 0
        self.done = False
        self.last_node_compute_load.fill(0.0)
        self.last_link_data_load.fill(0.0)
        self.last_request_counts.fill(0.0)
        self.last_result = None
        self.deployment = self.greedy_initial_deployment()
        return self.observe()

    def greedy_initial_deployment(self) -> np.ndarray:
        deployment = np.zeros_like(self.deployment)
        mem_used = np.zeros(self.num_nodes, dtype=np.float64)
        st_used = np.zeros(self.num_nodes, dtype=np.float64)
        max_replicas = int(self.config["simulation"]["max_service_replicas"])
        node_order = np.argsort(-self.compute_capacity)

        for i, svc in enumerate(self.services):
            for j, _stage in enumerate(svc["stages"]):
                mem_req = self._service_stage_resources["memory"][i, j]
                st_req = self._service_stage_resources["storage"][i, j]
                replicas = 0
                for node in node_order:
                    if mem_used[node] + mem_req <= self.memory_capacity[node] and st_used[node] + st_req <= self.storage_capacity[node]:
                        deployment[i, j, node] = 1
                        mem_used[node] += mem_req
                        st_used[node] += st_req
                        replicas += 1
                    if replicas >= max_replicas:
                        break
                if replicas == 0:
                    raise RuntimeError(f"No feasible initial deployment for service={i} stage={j}.")
        return deployment

    def apply_deployment(self, deployment: np.ndarray) -> dict[str, float]:
        deployment = self.coverage_repaired_deployment(deployment.astype(np.int64))
        violation = self.deployment_violation(deployment)
        self.deployment = deployment
        return violation

    def coverage_repaired_deployment(self, deployment: np.ndarray) -> np.ndarray:
        """Return a deployment with basic service coverage for all source nodes.

        If a learned deployment leaves any source/service pair without a feasible
        staged path, fall back to the deterministic greedy deployment. This keeps
        early training numerically useful while Agent-D learns the slower policy.
        """
        if self._has_full_path_coverage(deployment):
            return deployment
        return self.greedy_initial_deployment()

    def _has_full_path_coverage(self, deployment: np.ndarray) -> bool:
        if self.deployment_violation(deployment)["missing"] > 0:
            return False
        for source in range(self.num_nodes):
            for service_id, svc in enumerate(self.services):
                actions = self.path_manager.feasible_actions(
                    source,
                    service_id,
                    len(svc["stages"]),
                    deployment,
                    self.adjacency,
                )
                if actions.size == 0:
                    return False
        return True

    def deployment_violation(self, deployment: np.ndarray) -> dict[str, float]:
        mem = np.einsum("ijn,ij->n", deployment, self._service_stage_resources["memory"])
        storage = np.einsum("ijn,ij->n", deployment, self._service_stage_resources["storage"])
        missing = 0
        for i, svc in enumerate(self.services):
            for j, _ in enumerate(svc["stages"]):
                if deployment[i, j].sum() <= 0:
                    missing += 1
        return {
            "memory": float(np.maximum(mem - self.memory_capacity, 0.0).sum()),
            "storage": float(np.maximum(storage - self.storage_capacity, 0.0).sum()),
            "missing": float(missing),
        }

    def step(self, scheduler, deployment_action: np.ndarray | None = None) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        deployment_penalty = {"memory": 0.0, "storage": 0.0, "missing": 0.0}
        if deployment_action is not None:
            deployment_penalty = self.apply_deployment(deployment_action)

        tasks = self.request_gen.generate(self.time_s)
        self.last_request_counts.fill(0.0)
        for task in tasks:
            self.last_request_counts[task.source_node, task.service_id] += 1.0

        scheduled, invalid = self._schedule_tasks(tasks, scheduler)
        result = self.kkt.allocate(scheduled)
        self.last_result = result
        self.last_node_compute_load = result.node_compute_load
        self.last_link_data_load = result.link_data_load

        reward = self._reward(result, invalid, deployment_penalty)
        info = {
            "time_s": self.time_s,
            "num_tasks": len(tasks),
            "invalid_schedule": invalid + result.invalid_count,
            "total_delay": result.total_delay,
            "average_delay": result.average_delay,
            "compute_delay": result.compute_delay,
            "transmission_delay": result.transmission_delay,
            "deployment_penalty": deployment_penalty,
            "user_count": int(self.request_gen.user_counts.sum()),
        }

        self.time_s += 1
        seconds = int(self.config["simulation"]["seconds_per_episode"])
        self.done = self.time_s >= seconds
        return self.observe(), reward, self.done, info

    def _schedule_tasks(self, tasks: list[Task], scheduler) -> tuple[list[ScheduledTask], int]:
        scheduled: list[ScheduledTask] = []
        invalid = 0
        pending_node = np.zeros(self.num_nodes, dtype=np.float64)
        pending_link = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        node_pressure = self.last_node_compute_load / np.maximum(self.compute_capacity, 1e-9)
        link_pressure = self.last_link_data_load / np.maximum(self.bandwidth, 1e-9)
        link_pressure[self.bandwidth <= 0] = 0.0

        for task in tasks:
            mask = self.path_manager.mask(
                task.source_node,
                task.service_id,
                task.stage_count,
                self.deployment,
                self.adjacency,
            )
            if not np.any(mask):
                invalid += 1
                continue
            if hasattr(scheduler, "select_path_with_obs"):
                task_obs = self.task_observation(task, pending_node, pending_link)
                action = scheduler.select_path_with_obs(task, mask, task_obs)
            elif hasattr(scheduler, "select_path"):
                try:
                    action = scheduler.select_path(
                        task,
                        mask,
                        node_pressure,
                        link_pressure,
                        pending_node,
                        pending_link,
                    )
                except TypeError:
                    action = scheduler.select_path(task, mask)
            else:
                raise TypeError("Scheduler must provide select_path(...).")

            if hasattr(scheduler, "record_teacher_sample"):
                task_obs = self.task_observation(task, pending_node, pending_link)
                scheduler.record_teacher_sample(task, mask, task_obs, action)

            path = self.path_manager.path(action)
            scheduled.append(ScheduledTask(task=task, path=path))
            self._add_pending_load(task, path, pending_node, pending_link)

        return scheduled, invalid

    def _add_pending_load(
        self,
        task: Task,
        path: tuple[int, int, int],
        pending_node: np.ndarray,
        pending_link: np.ndarray,
    ) -> None:
        active = path[: task.stage_count]
        for j, node in enumerate(active):
            pending_node[node] += float(task.compute_gcycles[j])
        if active[0] != task.source_node:
            pending_link[task.source_node, active[0]] += float(task.input_mb)
        for j, (left, right) in enumerate(zip(active[:-1], active[1:])):
            if left != right:
                pending_link[left, right] += float(task.output_mb[j])

    def _reward(self, result: KKTResult, invalid_schedule: int, deployment_penalty: dict[str, float]) -> float:
        delay_norm = result.average_delay
        invalid_penalty = 10.0 * invalid_schedule
        deployment_cost = (
            5.0 * deployment_penalty["missing"]
            + 0.5 * deployment_penalty["memory"]
            + 0.05 * deployment_penalty["storage"]
        )
        load_ratio = self.last_node_compute_load / np.maximum(self.compute_capacity, 1e-9)
        imbalance = float(np.std(load_ratio))
        return -float(delay_norm + invalid_penalty + deployment_cost + 0.1 * imbalance)

    def observe(self) -> np.ndarray:
        slots = float(self.config["simulation"]["slots_per_day"])
        angle = 2.0 * np.pi * (self.time_s % int(slots)) / slots
        time_features = np.array([np.sin(angle), np.cos(angle)], dtype=np.float64)
        users = self.request_gen.user_counts.astype(np.float64).ravel() / 5000.0
        node_load = self.last_node_compute_load / np.maximum(self.compute_capacity, 1e-9)
        link_load = self.last_link_data_load / np.maximum(self.bandwidth, 1e-9)
        link_load[self.bandwidth <= 0] = 0.0
        requests = self.last_request_counts.ravel() / 100.0
        deployment = self.deployment.astype(np.float64).ravel()
        return np.concatenate([time_features, users, node_load, link_load.ravel(), requests, deployment]).astype(np.float32)

    def task_observation(
        self,
        task: Task,
        pending_node_load: np.ndarray | None = None,
        pending_link_load: np.ndarray | None = None,
    ) -> np.ndarray:
        pending_node = np.zeros(self.num_nodes) if pending_node_load is None else pending_node_load
        pending_link = np.zeros((self.num_nodes, self.num_nodes)) if pending_link_load is None else pending_link_load
        source = np.zeros(self.num_nodes, dtype=np.float64)
        source[task.source_node] = 1.0
        service = np.zeros(self.num_services, dtype=np.float64)
        service[task.service_id] = 1.0
        stage_count = np.array([task.stage_count / self.max_stages], dtype=np.float64)
        task_values = np.concatenate(
            [
                np.array([task.input_mb / 100.0], dtype=np.float64),
                task.compute_gcycles / 50.0,
                task.output_mb / 100.0,
            ]
        )
        node_pressure = self.last_node_compute_load / np.maximum(self.compute_capacity, 1e-9)
        link_pressure = self.last_link_data_load / np.maximum(self.bandwidth, 1e-9)
        link_pressure[self.bandwidth <= 0] = 0.0
        return np.concatenate(
            [
                source,
                service,
                stage_count,
                task_values,
                self.deployment.ravel(),
                node_pressure,
                link_pressure.ravel(),
                pending_node / np.maximum(self.compute_capacity, 1e-9),
                pending_link.ravel() / np.maximum(self.bandwidth.ravel(), 1e-9),
            ]
        ).astype(np.float32)

    def _build_stage_resources(self) -> dict[str, np.ndarray]:
        memory = np.zeros((self.num_services, self.max_stages), dtype=np.float64)
        storage = np.zeros((self.num_services, self.max_stages), dtype=np.float64)
        for i, svc in enumerate(self.services):
            for j, stage in enumerate(svc["stages"]):
                memory[i, j] = float(stage["memory_gb"])
                storage[i, j] = float(stage["storage_gb"])
        return {"memory": memory, "storage": storage}
