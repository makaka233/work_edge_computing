from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from edge_drl.config import node_name_to_id, service_name_to_id
from edge_drl.env.time_patterns import tidal_factor


@dataclass(slots=True)
class Task:
    task_id: int
    time_s: int
    source_node: int
    user_type: int
    service_id: int
    stage_count: int
    input_mb: float
    compute_gcycles: np.ndarray
    output_mb: np.ndarray


class DynamicRequestGenerator:
    """Dynamic user-count request generator.

    Users are represented as counts per node and user type, not as individual
    objects. Each second, user counts change through arrivals/departures, and
    active users trigger service requests with Bernoulli sampling.
    """

    def __init__(self, config: dict[str, Any], rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.nodes = config["nodes"]
        self.services = config["services"]
        self.node_ids = node_name_to_id(config)
        self.service_ids = service_name_to_id(config)
        self.user_type_names = list(config["user_types"].keys())
        self.user_type_ids = {name: idx for idx, name in enumerate(self.user_type_names)}
        self.user_counts = self._initial_user_counts()
        self._task_id = 0

        self._service_probs = self._build_service_probs()
        self._base_probs = np.array(
            [config["user_types"][name]["base_request_probability"] for name in self.user_type_names],
            dtype=np.float64,
        )
        self._leave_probs = np.array(
            [config["user_types"][name]["leave_probability"] for name in self.user_type_names],
            dtype=np.float64,
        )
        self._arrival_rates = np.array(
            [config["user_types"][name]["arrival_rate"] for name in self.user_type_names],
            dtype=np.float64,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_user_types(self) -> int:
        return len(self.user_type_names)

    def reset(self) -> None:
        self.user_counts = self._initial_user_counts()
        self._task_id = 0

    def _initial_user_counts(self) -> np.ndarray:
        counts = np.zeros((len(self.nodes), len(self.config["user_types"])), dtype=np.int64)
        initial = self.config["initial_users"]
        for node_name, type_counts in initial.items():
            m = self.node_ids[node_name]
            for type_name, count in type_counts.items():
                counts[m, self.user_type_ids[type_name]] = int(count)
        return counts

    def _build_service_probs(self) -> np.ndarray:
        probs = np.zeros((len(self.user_type_names), len(self.services)), dtype=np.float64)
        for type_name, type_cfg in self.config["user_types"].items():
            q = self.user_type_ids[type_name]
            for service_name, p in type_cfg["service_probs"].items():
                probs[q, self.service_ids[service_name]] = float(p)
            total = probs[q].sum()
            if total <= 0:
                raise ValueError(f"User type {type_name} has no service probabilities.")
            probs[q] /= total
        return probs

    def _sample_lognormal(self, mean: float, cv: float) -> float:
        cv = max(float(cv), 1e-6)
        sigma2 = np.log(1.0 + cv**2)
        sigma = np.sqrt(sigma2)
        mu = np.log(max(mean, 1e-9)) - 0.5 * sigma2
        return float(self.rng.lognormal(mu, sigma))

    def _sample_task(self, time_s: int, source_node: int, user_type: int, service_id: int) -> Task:
        svc = self.services[service_id]
        stage_count = len(svc["stages"])
        compute = np.zeros(3, dtype=np.float64)
        output = np.zeros(3, dtype=np.float64)
        input_mb = self._sample_lognormal(svc["input_mb_mean"], svc["input_mb_cv"])
        current_mb = input_mb
        for j, stage in enumerate(svc["stages"]):
            compute[j] = self._sample_lognormal(stage["compute_gcycles_mean"], stage["compute_cv"])
            current_mb = max(1e-6, current_mb * float(stage["output_ratio"]))
            output[j] = current_mb
        task = Task(
            task_id=self._task_id,
            time_s=time_s,
            source_node=source_node,
            user_type=user_type,
            service_id=service_id,
            stage_count=stage_count,
            input_mb=input_mb,
            compute_gcycles=compute,
            output_mb=output,
        )
        self._task_id += 1
        return task

    def generate(self, time_s: int) -> list[Task]:
        self._update_user_counts(time_s)
        tasks: list[Task] = []
        max_prob = float(self.config["simulation"]["max_request_probability"])
        slots = int(self.config["simulation"]["slots_per_day"])

        for m, node in enumerate(self.nodes):
            factor = tidal_factor(time_s, node["tidal"], slots)
            probs = np.clip(self._base_probs * factor, 0.0, max_prob)
            requests_by_type = self.rng.binomial(self.user_counts[m], probs)
            for q, count in enumerate(requests_by_type):
                if count <= 0:
                    continue
                service_counts = self.rng.multinomial(int(count), self._service_probs[q])
                for service_id, service_count in enumerate(service_counts):
                    for _ in range(int(service_count)):
                        tasks.append(self._sample_task(time_s, m, q, service_id))

        return tasks

    def _update_user_counts(self, time_s: int) -> None:
        max_users = int(self.config["simulation"]["max_users_per_node_type"])
        slots = int(self.config["simulation"]["slots_per_day"])
        new_counts = self.user_counts.copy()

        for m, node in enumerate(self.nodes):
            factor = tidal_factor(time_s, node["tidal"], slots)
            arrivals = self.rng.poisson(self._arrival_rates * factor)
            departures = self.rng.binomial(new_counts[m], self._leave_probs)
            new_counts[m] = np.clip(new_counts[m] + arrivals - departures, 0, max_users)

        self.user_counts = new_counts.astype(np.int64)

