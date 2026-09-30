# Slow Count replica-ablation credit

## Problem addressed

In the global-incremental Slow policy, Count chooses which service stage gets
another replica, or `STOP`; Placement chooses the replica's node. The older
Count local return reused a **routing** comparison: how much better a selected
node was than another already-deployed node for one stage. That can value a
route without answering whether the *new replica* was necessary. `STOP` had
zero local credit. This made idle additional replicas and poor stopping
decisions difficult to distinguish.

## Opt-in correction

`--slow-replica-ablation-samples K` (with `K > 0`) changes only the
global-incremental **Count** local credit. It requires a positive
`--slow-counterfactual-credit-coef`. Zero retains the exact previous training
path. Placement's routing credit, the measured shared window return, the Slow
critic, Fast PPO, and the environment are unchanged.

For one nonempty sampled request tick in each Slow window, the trainer uses
the frozen deterministic Fast policy and the same pre-step load state to:

1. Route the whole request batch under the chosen deployment and jointly
   settle compute and link allocation with the existing KKT implementation.
2. Temporarily remove up to `K` chosen ADD replicas, route the *same* batch
   again, and measure `latency_without - latency_with`. A positive value means
   that replica helped. Existing mandatory coverage replicas are never removed.
3. If Count chose `STOP`, temporarily add up to `K` feasible candidate
   replicas and measure `best_tested_addition_latency - stop_latency`. A
   positive value favors stopping among the tested additions.

The temporary deployments are restored before the actual environment step.
The comparison makes no Fast PPO buffer entries and changes no optimizer or
checkpoint. The STOP candidates are demand-ranked stages with nodes chosen by
the frozen deterministic Placement actor; this is a bounded probe, **not** an
exhaustive search or proof that stopping is optimal. ADD probes rotate across
windows and episodes so a small `K` does not permanently favor early stage IDs.
Unprobed ADD actions receive zero *ablation* credit, but keep their resource
cost and shared window return. This avoids imputing the old routing proxy as
replica value.

This is a one-tick estimate, not an exact full-window causal effect: demand
and loads later in the window can differ, and training Fast samples actions
while the diagnostic replays deterministic Fast. Higher `K` covers more
actions but requires more Fast routing and KKT evaluations. Start with `K=2`
and compare against the `K=0` control using identical seeds and scenario
settings; do not assume the new reward guarantees better latency.

The training and rollout CSVs contain `slow_replica_ablation_samples`,
`slow_replica_ablation_mean_s`, and `slow_stop_ablation_value_s`. Interpret
them alongside ADD/STOP action counts, used-replica fraction, mean latency,
and the five-scenario evaluation. A fresh run is preferable to resuming an
old checkpoint because the reward definition has changed; the checkpoint
format itself remains compatible.
