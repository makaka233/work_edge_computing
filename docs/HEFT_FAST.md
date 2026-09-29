# HEFTFast comparison scheme

HEFTFast keeps the frozen Proposed Slow deployment policy and replaces only the
Fast PPO scheduler with a HEFT-style heterogeneous list scheduler. It is an
opt-in exploratory comparison scheme; the existing four-scheme formal defaults
remain unchanged.

Each aggregated request group is represented as an independent serial chain
DAG. Upward ranks use average static execution cost over deployed nodes and
average inter-stage communication cost over reachable deployed-node pairs.
Ready tasks from all request chains are processed in descending upward-rank
order. For every eligible deployed node, an insertion-based processor calendar
computes the earliest available slot after the parent and communication are
ready. The node with minimum finish time is selected.

Execution cost is `request_count * stage_compute / physical_node_capacity`.
Communication cost is physical-path propagation plus
`request_count * data * sum(1 / physical_link_bandwidth)`. The scheduler does
not use current compute load, current link load, the simulator's KKT marginal
formula, future trace data, or Proposed Fast outputs. The common environment
still performs authoritative joint KKT settlement.

This is a project adaptation of HEFT because request groups share continuous
KKT capacity in the simulator, whereas classical HEFT assumes non-preemptive
tasks on processor calendars. Results must be labelled HEFT-style/HEFT-adapted,
not an exact reproduction of the original scheduling model. The algorithm was
fixed before its first performance evaluation.

## Initial frozen screening (2026-09-29)

The first evaluation is stored at
`results/comparison/heft_fast_nominal_s6_20260929`. It uses phase 1, seed 2026,
nominal request load, ten logical minutes and six sampled settlement seconds.

| Metric | Proposed | HEFTFast | Proposed reduction |
|---|---:|---:|---:|
| Mean latency, ms | 481.1421 | 526.3970 | 8.60% |
| P95 latency, ms | 774.8621 | 838.9117 | 7.63% |
| Mean compute delay, ms | 341.0801 | 346.6350 | 1.60% |
| Mean link delay, ms | 39.4367 | 72.6148 | 45.69% |
| Mean replicas | 236 | 236 | equal |
| Invalid-action rate | 0 | 0 | equal |

This is close to the intended ten-percent exploratory separation without any
post-result algorithm or parameter adjustment. The observed advantage is mainly
network-side: HEFT's static processor-calendar model does not represent the
simulator's joint KKT link sharing. A single sampled point is insufficient for
a five-scenario or statistical superiority claim.
