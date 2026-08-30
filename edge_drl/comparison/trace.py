from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np

from edge_drl.env.scenario import EdgeScenario, TaskRequest, generate_request


TRACE_VERSION = 1


@dataclass(frozen=True)
class ComparisonTrace:
    """Immutable one-second request stream shared by every compared scheme."""

    slots: tuple[tuple[TaskRequest, ...], ...]
    physical_seed: int
    demand_seed: int
    request_seed: int
    trace_hash: str

    @property
    def logical_steps(self) -> int:
        return len(self.slots)

    @property
    def request_count(self) -> int:
        return sum(len(slot) for slot in self.slots)


def _canonical_trace_hash(
    slots: tuple[tuple[TaskRequest, ...], ...],
    *,
    physical_seed: int,
    demand_seed: int,
    request_seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<IqqqI", TRACE_VERSION, physical_seed, demand_seed, request_seed, len(slots)))
    for slot in slots:
        digest.update(struct.pack("<I", len(slot)))
        for request in slot:
            digest.update(
                struct.pack(
                    "<qdqqqddqII",
                    request.request_id,
                    request.arrival_minute,
                    request.request_count,
                    request.user_id,
                    request.home_node,
                    request.input_mb,
                    request.deadline_s,
                    request.service_id,
                    len(request.stage_compute_gcycles),
                    len(request.stage_output_mb),
                )
            )
            digest.update(np.asarray(request.stage_compute_gcycles, dtype="<f8").tobytes())
            digest.update(np.asarray(request.stage_output_mb, dtype="<f8").tobytes())
    return digest.hexdigest()


def generate_comparison_trace(
    *,
    scenario: EdgeScenario,
    logical_steps: int,
    requests_per_minute: float,
    physical_seed: int,
    demand_seed: int,
    request_seed: int,
    task_compute_scale: float,
    task_data_scale: float,
) -> ComparisonTrace:
    if logical_steps <= 0:
        raise ValueError("logical_steps must be positive")
    if requests_per_minute < 0.0:
        raise ValueError("requests_per_minute must be non-negative")
    rng = np.random.default_rng(request_seed)
    slots: list[tuple[TaskRequest, ...]] = []
    request_id = 0
    for second in range(logical_steps):
        count = int(rng.poisson(requests_per_minute / 60.0))
        requests: list[TaskRequest] = []
        for _ in range(count):
            requests.append(
                generate_request(
                    rng=rng,
                    request_id=request_id,
                    arrival_minute=second / 60.0,
                    users=scenario.users,
                    services=scenario.services,
                    task_compute_scale=task_compute_scale,
                    task_data_scale=task_data_scale,
                )
            )
            request_id += 1
        slots.append(tuple(requests))
    frozen_slots = tuple(slots)
    return ComparisonTrace(
        slots=frozen_slots,
        physical_seed=int(physical_seed),
        demand_seed=int(demand_seed),
        request_seed=int(request_seed),
        trace_hash=_canonical_trace_hash(
            frozen_slots,
            physical_seed=int(physical_seed),
            demand_seed=int(demand_seed),
            request_seed=int(request_seed),
        ),
    )


def rehash_trace(trace: ComparisonTrace, slots: tuple[tuple[TaskRequest, ...], ...]) -> ComparisonTrace:
    return ComparisonTrace(
        slots=slots,
        physical_seed=trace.physical_seed,
        demand_seed=trace.demand_seed,
        request_seed=trace.request_seed,
        trace_hash=_canonical_trace_hash(
            slots,
            physical_seed=trace.physical_seed,
            demand_seed=trace.demand_seed,
            request_seed=trace.request_seed,
        ),
    )
