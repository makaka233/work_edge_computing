# KKT continuous-resource allocation comparison

This experiment isolates the contribution of the continuous KKT allocator. It
does not train a new policy. One frozen Proposed checkpoint generates every Slow
deployment and Fast path action on the ordinary KKT trajectory. The exact same
actions are replayed in four synchronized copies of the same immutable request
trace and physical scenario.

| Name | Compute settlement | Link settlement |
|---|---|---|
| Proposed | KKT square-root allocation | KKT square-root allocation |
| NoKKT-Compute | Equal share | KKT square-root allocation |
| NoKKT-Link | KKT square-root allocation | Equal share |
| NoKKT-All | Equal share | Equal share |

The experiment changes only continuous compute and bandwidth settlement. The
checkpoint, deployment, scheduled stage nodes, request trace, serial-phase
separation, topology, capacity, propagation, radio access, invalid-action rule,
and sampled-time weighting are shared. Every output row records the same
`action_trace_hash`, making accidental action divergence visible.

## Allocation rules

For concurrent demands on one resource, let capacity be `F`, per-request work
be `C_i`, and the number of identical requests represented by group `i` be
`m_i`. KKT assigns aggregate group rate

```
r_i = F * m_i * sqrt(C_i) / sum_j(m_j * sqrt(C_j)).
```

For links, `C_i` is replaced by transferred data. This minimizes the modeled
weighted sum delay for that resource and serial phase.

Equal Share is a non-optimizing processor-sharing baseline. Every individual
request or flow receives the same rate, so a request group receives

```
r_i = F * m_i / sum_j(m_j).
```

The group representation is therefore equivalent to expanding it into `m_i`
identical requests. Equal Share is applied independently in each causal service
stage, just like KKT. It does not mean unlimited capacity or no contention.

## Interpretation

`NoKKT-Compute - Proposed` measures the effect of replacing optimized compute
allocation. `NoKKT-Link - Proposed` measures the corresponding bandwidth effect.
`NoKKT-All - Proposed` measures their combined causal effect, including later
load-state consequences. The last difference need not equal the sum of the two
individual differences because dynamic node/link loads interact over time.

KKT minimizes the modeled weighted total resource delay, so its total compute or
link delay should not exceed Equal Share for identical instantaneous capacities
and actions. End-to-end P95 is not covered by that guarantee because KKT's
objective is a sum, and the replay environments develop different load histories.
The stage-heterogeneity, high-load, low-compute, and low-bandwidth points should
show the most informative differences. Report mean/P95 latency, compute/link
components, deadline violations, paired differences, trace hash, and action hash.

This is a component comparison of the fixed system, not a separately trained
NoKKT controller. It answers how much the deployed checkpoint benefits from KKT
at inference under the same chosen actions. It does not answer how well a policy
retrained specifically for Equal Share could perform.

## Commands

Quick single-seed nominal check with training-equivalent temporal sampling:

```powershell
$runId = "kkt_replay_phase1_s6_$(Get-Date -Format 'yyyyMMdd_HHmmss')"; python .\run_kkt_comparison.py `
  --checkpoint ".\runs\global_incremental_cf_s6_u180_20260903_192623\checkpoints\best.pt" `
  --phase 1 `
  --sampled-seconds-per-window 6 `
  --device cuda `
  --run-id $runId
```

Single-seed five-scenario screening:

```powershell
$runId = "kkt_replay_5scenes_s6_$(Get-Date -Format 'yyyyMMdd_HHmmss')"; python .\run_kkt_comparison.py `
  --checkpoint ".\runs\global_incremental_cf_s6_u180_20260903_192623\checkpoints\best.pt" `
  --phase 2 `
  --eval-seeds 2026 `
  --sampled-seconds-per-window 6 `
  --device cuda `
  --run-id $runId
```

Results are written under `results/kkt_comparison/<run-id>/`: per-scenario raw
rows, paired differences, aggregate summaries, metadata, and a trace manifest.

## Initial action-replay result (2026-09-29)

The implementation was screened at
`results/kkt_comparison/kkt_action_replay_phase1_s6_20260929` using phase 1,
seed 2026, nominal request load, ten logical minutes, and six sampled settlement
seconds. All four rows have action hash
`4b08381fb04f20aaa4c9db3464808c6fd0f331f36480c2ccc858569ec3ca0b2f`
and zero invalid actions.

| Mode | Mean latency, ms | P95, ms | Compute, ms | Link, ms | Violation rate |
|---|---:|---:|---:|---:|---:|
| Proposed | 481.1421 | 774.8621 | 341.0801 | 39.4367 | 0.3234 |
| NoKKT-Compute | 492.3174 | 872.0732 | 352.2554 | 39.4367 | 0.3473 |
| NoKKT-Link | 483.0161 | 792.2684 | 341.0801 | 41.3107 | 0.3293 |
| NoKKT-All | 494.1914 | 875.5043 | 352.2554 | 41.3107 | 0.3533 |

Relative to NoKKT-All, full KKT reduces mean latency by 2.64% and P95 by
11.50% at this point. Removing compute KKT accounts for 11.1753 ms of the mean
increase; removing link KKT accounts for 1.8740 ms. Deadline violations increase
by 2.99 percentage points when both are removed. This one sampled seed is a
correctness and effect-size screen, not a five-scenario statistical conclusion.
