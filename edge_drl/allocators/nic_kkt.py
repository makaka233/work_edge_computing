"""Convex bandwidth allocation with full-duplex aggregate node NIC limits.

Within a serial phase, combine flows on edge e into S_e=sum(m*sqrt(D)).
Minimize sum(S_e**2 / b_e), subject to b_e<=B_e, sum_out(b)<=NIC,
sum_in(b)<=NIC. At fixed port duals u,v: b_e=min(B_e,S_e/sqrt(u+v)).
Alternating exact dual coordinate maximization is vectorized over ports.
Slow, tightly coupled cases use a joint convex-dual solve with the SAME
primal-dual certificate; no capacity or precision limit is relaxed.
"""
from __future__ import annotations

import numpy as np

from edge_drl.allocators.kkt import LinkDemand


def nic_link_rates(sqrt_loads, link_capacities, nic_capacities, *, initial_duals=None,
                   tolerance=1e-7, max_iterations=300):
    loads = np.asarray(sqrt_loads, dtype=np.float64)
    capacities = np.asarray(link_capacities, dtype=np.float64)
    nic = np.asarray(nic_capacities, dtype=np.float64)
    n = len(nic)
    if loads.shape != (n, n) or capacities.shape != (n, n):
        raise ValueError("NIC allocator dimensions do not match")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("NIC solver tolerance and iteration budget must be positive")
    if not np.isfinite(loads).all() or np.any(loads < 0) or np.any(nic <= 0) or np.any(np.isnan(nic)):
        raise ValueError("invalid NIC allocator loads/capacities")
    active = loads > 0
    if np.any(active & (~np.isfinite(capacities) | (capacities <= 0))):
        raise ValueError("active link requires positive finite bandwidth")
    caps = np.where(active, capacities, 0.0)
    # Infinite NICs preserve the independent-link result exactly.
    if np.all(caps.sum(axis=1) <= nic) and np.all(caps.sum(axis=0) <= nic):
        return caps, float(np.sum(np.divide(loads ** 2, caps, out=np.zeros_like(loads), where=active))), (np.zeros(n), np.zeros(n))
    outgoing, incoming = (np.zeros(n), np.zeros(n)) if initial_duals is None else (
        np.maximum(np.asarray(initial_duals[0], dtype=np.float64), 0).copy(),
        np.maximum(np.asarray(initial_duals[1], dtype=np.float64), 0).copy())

    def rates(u, v):
        return np.minimum(caps, loads / np.sqrt(np.maximum(u[:, None] + v[None, :], 1e-30)))

    def port_duals(other, axis):
        # At zero opposite-port dual, this upper bound would satisfy the
        # NIC even without link caps. Nonnegative other duals only reduce rates.
        upper = np.square(np.divide(loads.sum(axis=axis), nic,
                                   out=np.zeros(n), where=np.isfinite(nic)))
        lower = np.zeros(n)
        base = rates(np.zeros(n), other) if axis == 1 else rates(other, np.zeros(n))
        binding = base.sum(axis=axis) > nic
        result = np.zeros(n)
        if not binding.any():
            return result
        selected = np.flatnonzero(binding)
        # Only constrained ports need a root. Orient both send and receive
        # updates as rows, preserving the same 35 bisections and KKT tolerance.
        port_loads = loads[selected] if axis == 1 else loads[:, selected].T
        port_caps = caps[selected] if axis == 1 else caps[:, selected].T
        lower, upper = lower[selected], upper[selected]
        for _ in range(35):
            middle = (lower + upper) * 0.5
            candidate = np.minimum(port_caps, port_loads / np.sqrt(np.maximum(middle[:, None] + other[None, :], 1e-30)))
            excessive = candidate.sum(axis=1) > nic[selected]
            lower = np.where(excessive, middle, lower)
            upper = np.where(excessive, upper, middle)
        result[selected] = upper
        return result

    def certificate(outgoing, incoming):
        bandwidth = rates(outgoing, incoming)
        row_sum, col_sum = bandwidth.sum(axis=1), bandwidth.sum(axis=0)
        objective = float(np.sum(np.divide(loads ** 2, bandwidth, out=np.zeros_like(loads), where=active)))
        slack = np.sum(outgoing[np.isfinite(nic)] * (nic[np.isfinite(nic)] - row_sum[np.isfinite(nic)]))
        slack += np.sum(incoming[np.isfinite(nic)] * (nic[np.isfinite(nic)] - col_sum[np.isfinite(nic)]))
        violation = max(float(np.max(np.divide(row_sum, nic))), float(np.max(np.divide(col_sum, nic))))
        feasible = bandwidth * np.minimum(1.0, nic / np.maximum(row_sum, 1e-30))[:, None]
        feasible *= np.minimum(1.0, nic / np.maximum(feasible.sum(axis=0), 1e-30))[None, :]
        feasible_objective = float(np.sum(np.divide(loads ** 2, feasible, out=np.zeros_like(loads), where=active)))
        # Dual objective is raw objective minus lambda*(capacity-usage).
        # Compare it to a *feasible* primal, rather than requiring an overly
        # strict raw port residual on nearly degenerate low-weight links.
        gap = feasible_objective - objective + float(slack)
        return feasible, feasible_objective, gap, violation

    def certified(gap, objective):
        return -1e-10 * max(objective, 1e-12) <= gap <= tolerance * max(objective, 1e-12)

    # Common warm-start cases finish in a few sweeps. Tightly coupled ports
    # should switch algorithms early, not pay hundreds of fruitless sweeps for
    # every selected stage. The certificate, not this budget, controls accuracy.
    for _ in range(min(max_iterations, 32)):
        outgoing = port_duals(incoming, 1)
        incoming = port_duals(outgoing, 0)
        feasible, feasible_objective, gap, violation = certificate(outgoing, incoming)
        if certified(gap, feasible_objective):
            return feasible, feasible_objective, (outgoing, incoming)

    # Alternating port updates can be arbitrarily slow when both endpoints
    # bind and their duals trade off almost freely. Optimize the SAME convex
    # dual jointly rather than relaxing ports or accepting an uncertified cost.
    from scipy.optimize import minimize

    selected_out = np.flatnonzero(caps.sum(axis=1) > nic)
    selected_in = np.flatnonzero(caps.sum(axis=0) > nic)
    budgets = np.concatenate((nic[selected_out], nic[selected_in]))
    objective_scale = max(feasible_objective, 1e-12)
    dual_scales = objective_scale / budgets

    def unpack(x):
        u, v = np.zeros(n), np.zeros(n)
        u[selected_out] = x[:len(selected_out)] * dual_scales[:len(selected_out)]
        v[selected_in] = x[len(selected_out):] * dual_scales[len(selected_out):]
        return u, v

    def negative_dual(x):
        u, v = unpack(x)
        bandwidth = rates(u, v)
        row_sum, col_sum = bandwidth.sum(axis=1), bandwidth.sum(axis=0)
        objective = float(np.sum(np.divide(loads ** 2, bandwidth,
                                          out=np.zeros_like(loads), where=active)))
        residuals = budgets - np.concatenate((row_sum[selected_out], col_sum[selected_in]))
        # Scale each variable by its port budget. The normalized gradient is
        # a relative capacity residual, avoiding tiny raw duals in MB/s units.
        value = (-objective + np.dot(x * dual_scales, residuals)) / objective_scale
        return value, residuals / budgets

    x0 = np.concatenate((outgoing[selected_out], incoming[selected_in])) / dual_scales
    result = minimize(negative_dual, x0, jac=True, method="L-BFGS-B",
                      bounds=[(0.0, None)] * len(x0),
                      options={"ftol": 1e-15, "gtol": 1e-10,
                               "maxiter": 3000, "maxls": 50, "maxcor": 30})
    outgoing, incoming = unpack(result.x)
    feasible, feasible_objective, gap, violation = certificate(outgoing, incoming)
    if certified(gap, feasible_objective):
        return feasible, feasible_objective, (outgoing, incoming)

    # L-BFGS-B may stop because consecutive objective values are indistinguish-
    # able, while a few endpoint capacity residuals still fail our certificate.
    # Its success flag is therefore NOT acceptance. Starting from the joint
    # solution, exact port roots remove those residuals without changing the
    # physical constraints, objective, or primal-dual tolerance.
    for _ in range(64):
        outgoing = port_duals(incoming, 1)
        incoming = port_duals(outgoing, 0)
        feasible, feasible_objective, gap, violation = certificate(outgoing, incoming)
        if certified(gap, feasible_objective):
            return feasible, feasible_objective, (outgoing, incoming)

    # Rare coupled cases need another joint step after port polishing. Disable
    # objective-change stopping for this refinement; still check the SAME
    # independent certificate rather than trusting optimizer termination.
    x0 = np.concatenate((outgoing[selected_out], incoming[selected_in])) / dual_scales
    result = minimize(negative_dual, x0, jac=True, method="L-BFGS-B",
                      bounds=[(0.0, None)] * len(x0),
                      options={"ftol": 0.0, "gtol": 1e-12,
                               "maxiter": 10000, "maxls": 100, "maxcor": 30})
    outgoing, incoming = unpack(result.x)
    for iteration in range(max_iterations + 1):
        if iteration:
            outgoing = port_duals(incoming, 1)
            incoming = port_duals(outgoing, 0)
        feasible, feasible_objective, gap, violation = certificate(outgoing, incoming)
        if certified(gap, feasible_objective):
            return feasible, feasible_objective, (outgoing, incoming)
    raise RuntimeError("aggregate NIC KKT solver did not converge: "
                       f"max_port_capacity_ratio={violation:.9g}, "
                       f"relative_gap={gap/max(feasible_objective,1e-12):.3g}, "
                       f"joint_dual_status={result.message}")


def allocate_link_nic_kkt(demands: list[LinkDemand], link_capacities, nic_capacities):
    loads = np.zeros_like(link_capacities, dtype=np.float64)
    for demand in demands:
        if demand.src_node != demand.dst_node and demand.data_mb > 0:
            loads[demand.src_node, demand.dst_node] += demand.multiplicity * np.sqrt(demand.data_mb)
    bandwidth, objective, _ = nic_link_rates(loads, link_capacities, nic_capacities)
    allocations, delays = {}, {}
    for demand in demands:
        src, dst = demand.src_node, demand.dst_node
        if src == dst or demand.data_mb <= 0:
            allocations[demand.demand_id] = delays[demand.demand_id] = 0.0
        else:
            rate = bandwidth[src, dst] * demand.multiplicity * np.sqrt(demand.data_mb) / loads[src, dst]
            allocations[demand.demand_id] = float(rate)
            delays[demand.demand_id] = float(demand.data_mb * demand.multiplicity / rate)
    return allocations, delays, objective
