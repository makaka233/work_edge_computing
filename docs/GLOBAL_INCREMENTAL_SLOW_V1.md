# Global incremental Slow allocation

## Why this change exists

The former Slow Count policy chose a complete replica count for one stage at a
time, in fixed service/stage order.  Under the realistic constrained profile,
service memory consumes most of the allowed pool.  Early stages could therefore
claim the remaining capacity before later stages were evaluated.  The policy
was learning counts, but it was not learning the global question: which stage
should receive the next scarce replica?

Recent training also reduced latency mainly by colocating adjacent stages.  It
lowered link latency and cross-node transitions much more than compute latency,
while the tail and dispersion of node compute load barely improved.  This is
why continuing the former run was unlikely to expose the intended advantage of
stage-wise placement.

## New optional training path

`--slow-allocation-mode global-incremental` changes only the Slow allocation
decision process:

1. Establish one feasible replica for every valid stage so coverage is not a
   learned accident.
2. Present every valid stage to one shared graph actor.
3. At each decision, choose the globally most useful stage for one additional
   replica, or choose `STOP`.
4. Use the existing Placement PPO to choose the node for that replica.
5. Rebuild the state and feasibility mask before the next decision so the actor
   observes the budget already consumed by earlier choices.

This removes service-ID priority and makes Count a sequential resource-budget
policy.  The legacy path remains the default and is unchanged unless the new
flag is selected.

## Same-window marginal credit

`--slow-counterfactual-credit-coef` adds a node-local marginal signal to Slow
Count and Placement.  For a sampled request and stage, the selected deployed
node is compared with the best other deployed feasible node while preserving
the same request, chain prefix, and current resource state.  Positive credit
means the selected node beats its available alternative; negative credit means
another deployed node would have been better.

This signal addresses two ambiguities in the old observational return:

- a replica is no longer rewarded merely because Fast happened to use it;
- Count receives credit for the actual node created by an incremental action,
  rather than only a service-stage average.

The calculation is a same-trace proxy, not a second exact environment rollout.
It uses the scheduler's latency model and is deliberately clipped, while the
measured window return and Slow critic remain the global objective.

## Sparse temporal sampling correction

`--sampled-load-update-mode interval-average` updates compute/link EWMA loads
once from the sampled interval's aggregate utilization.  The former
`legacy-repeat` path repeatedly applied a nonlinear clipped update for the same
sampled second.  That could make a 6-second sample behave like many independent
load observations and distort the congestion state seen by PPO.

The legacy method remains available and is still the default for reproducible
old experiments.

## Logs and checkpoints

Training and rollout CSV files now include:

- `slow_incremental_add_actions`
- `slow_incremental_stop_actions`
- `slow_counterfactual_marginal_value`
- `slow_counterfactual_samples`

Checkpoints save the Slow allocation architecture.  A global-incremental
checkpoint can resume a global-incremental run, and legacy checkpoints remain
loadable by legacy runs.  Because the Count action space and network differ,
loading a legacy Count head into a global-incremental run is intentionally
rejected with a clear error.  Start the first global-incremental experiment
without `--load-checkpoint`.

## What a short validation run must establish

Before a 320-update experiment, verify that:

- invalid actions remain zero and every stage retains coverage;
- incremental add and STOP counts are both visible and do not collapse at the
  first few updates;
- marginal credit has nonzero samples and finite values;
- compute latency or compute-load dispersion improves, rather than obtaining
  nearly all latency gain from fewer cross-node transitions;
- Fast/Slow loss, entropy, KL, and checkpoint files remain finite.

The new path is an experimental architecture and requires fresh training.  It
does not claim that one short run is sufficient to establish superiority over
Monolithic or the legacy Proposed policy.
