from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from collections.abc import Sequence

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
    requests_per_minute_schedule: Sequence[float] | None = None,
    schedule_window_seconds: int = 600,
    request_stride_seconds: int = 1,
    reload_schedule_boundaries: bool = False,
    reload_window_seconds: int = 600,
) -> ComparisonTrace:
    if logical_steps <= 0:
        raise ValueError("logical_steps must be positive")
    if requests_per_minute < 0.0:
        raise ValueError("requests_per_minute must be non-negative")
    if requests_per_minute_schedule is not None:
        if not requests_per_minute_schedule:
            raise ValueError("requests_per_minute_schedule must not be empty")
        if schedule_window_seconds <= 0:
            raise ValueError("schedule_window_seconds must be positive")
        schedule = tuple(float(value) for value in requests_per_minute_schedule)
        if any(value < 0.0 for value in schedule):
            raise ValueError("requests_per_minute_schedule values must be non-negative")
    else:
        schedule = None
    if request_stride_seconds <= 0:
        raise ValueError("request_stride_seconds must be positive")
    if reload_schedule_boundaries and reload_window_seconds <= 0:
        raise ValueError("reload_window_seconds must be positive")
    rng = np.random.default_rng(request_seed)
    slots: list[tuple[TaskRequest, ...]] = []
    request_id = 0
    for second in range(logical_steps):
        if schedule is None:
            rate = float(requests_per_minute)
        else:
            schedule_index = min(second // int(schedule_window_seconds), len(schedule) - 1)
            rate = schedule[schedule_index]
        requests: list[TaskRequest] = []
        if second % int(request_stride_seconds) != 0:
            slots.append(tuple(requests))
            continue
        # In the Proposed window collector, changing the multiplier at a
        # boundary regenerates the current second once.  The first draw (made
        # by the preceding env.step) is discarded and the second draw is the
        # one consumed by the new window.  Reproduce that RNG progression when
        # a synchronized training trace is requested.
        if (
            reload_schedule_boundaries
            and second > 0
            and second % int(reload_window_seconds) == 0
            and schedule is not None
        ):
            previous_index = min((second - 1) // int(schedule_window_seconds), len(schedule) - 1)
            if not np.isclose(schedule[previous_index], rate):
                discarded_count = int(rng.poisson(schedule[previous_index] / 60.0))
                for _ in range(discarded_count):
                    generate_request(
                        rng=rng,
                        request_id=request_id,
                        arrival_minute=second / 60.0,
                        users=scenario.users,
                        services=scenario.services,
                        task_compute_scale=task_compute_scale,
                        task_data_scale=task_data_scale,
                    )
                    request_id += 1
        count = int(rng.poisson(rate / 60.0))
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
