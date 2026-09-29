# PopularityEFT exploratory baseline

An opt-in, checkpoint-free deployment and scheduling heuristic. Existing four
scheme defaults, Proposed training, and public KKT settlement are unchanged.
This is a project-specific heuristic, not a reproduction of PRP or HEFT.

## Fixed algorithm definition

At each deployment boundary, forecast service demand from the same user service
weights available to the existing Algorithmic planner. Use current load-adjusted
compute capacities and physical network paths. A coverage MILP places exactly
one replica per valid stage subject to the common memory/storage constraints.
Its cost is demand-weighted ingress/locality transport plus stage execution.
The solver has a five-second limit and 1% relative gap, as in Algorithmic;
absence of a feasible solution is an explicit failure.

Then repeatedly choose a feasible stage/node addition. For each stage, its
candidate node minimizes the same locality/execution cost over all feasible
nodes. Rank stages by demand-weighted work divided by currently deployed pooled
compute capacity and the candidate's normalized memory/storage footprint.
Continue until no positive-demand feasible addition exists. No artificial
replica cap, discarded feasible candidate shortlist, or test-result tuning is
used. This is proportional resource allocation, not marginal full-chain search.

In each sampled second, process requests in trace order and stages in dependency
order. Pick the deployed, reachable node minimizing incoming transfer time plus
predicted local execution time. Execution is estimated as (previously reserved
phase work + this request group's work) / effective compute capacity. Links use
the analogous data-volume ledger and propagation delay. Update these ledgers
after every stage. Reachability pruning prevents selecting a dead-end prefix.
This is an online completion-time heuristic: it does not claim to implement
HEFT's upward ranking, processor availability calendar, or insertion scheduling.

The heuristic's scores only select actions. Mean/p95 latency and invalid rates
come exclusively from the same public environment settlement as Proposed.
No future trace, learned checkpoint, or Proposed performance enters the heuristic.

## Scope of conclusions

The baseline tests the complete learned controller against demand/locality
placement and myopic work-based scheduling. A win cannot by itself attribute
the advantage to Slow or Fast, nor establish superiority to KKT dynamic
programming. Keep Hybrid and Algorithmic as stronger references. Ranking is
an empirical result, not an acceptance condition that changes the algorithm.
Single-seed sampled screening is exploratory and cannot establish all-scenario
superiority.

## Commands

```powershell
python .\run_comparison.py --checkpoint .\runs\global_incremental_cf_s6_u180_20260903_192623\checkpoints\best.pt --schemes Proposed PopularityEFT --phase 1 --sampled-seconds-per-window 6 --device cuda --run-id popularity_eft_phase1
```

For five-scenario screening, use phase 2 with `--eval-seeds 2026` and a new run
ID. Include `Monolithic` and its explicit checkpoint when desired. Plotting
supports PopularityEFT alongside the existing schemes. Neither command trains
or modifies the Proposed checkpoint.

## Initial screening (2026-09-29)

The first implementation was tested without outcome-driven parameter changes:
`results/comparison/popularity_eft_nominal_s6_20260929` (phase 1, seed 2026,
10 logical minutes, six settlement samples, nominal request load).

| Metric | Proposed | PopularityEFT |
|---|---:|---:|
| Mean latency, ms | 481.1421 | 499.1952 |
| P95 latency, ms | 774.8621 | 798.2292 |

Proposed's mean is 3.62% lower and P95 is 2.93% lower at this point. This does
not establish superiority across the five scenario families or attribute the
improvement to one controller component. Monolithic was not evaluated in this
screening. The implementation remains fixed for subsequent comparisons.

## PopularityNearest locality baseline

`PopularityNearest` reuses the frozen PopularityPlacement definition above and
changes only scheduling. Every request follows its serial chain in trace order;
each stage chooses the reachable deployed replica with minimum physical-path
propagation delay from the request home or previous stage. It does not observe
current compute load, link load, batch ledgers, or KKT marginal cost. Ties use
node ID. This represents a locality-first edge heuristic and tests the value of
Proposed's dynamic load-aware scheduling. Its definition was fixed before its
first performance run and is recorded separately from PopularityEFT.

Its first frozen screening is stored in
`results/comparison/popularity_nearest_nominal_s6_20260929` under the same phase
1, seed 2026, nominal-load and six-sample protocol used above.

| Metric | Proposed | PopularityNearest | Proposed reduction |
|---|---:|---:|---:|
| Mean latency, ms | 481.1421 | 728.8259 | 33.98% |
| P95 latency, ms | 774.8621 | 1328.5537 | 41.68% |
| Mean compute delay, ms | 341.0801 | 591.9861 | 42.38% |
| Mean replicas | 236 | 236 | equal |
| Invalid-action rate | 0 | 0 | equal |

This passes a 10% screening threshold at this one point. It shows that dynamic
load-aware scheduling is useful relative to the locality-only rule here; it
does not prove a 10% advantage over five scenario families or over stronger
KKT-aware Algorithmic/Hybrid schemes.
