# Monolithic optimization protocol

This revision addresses two limitations observed in the constrained Edge-AI
scenario without changing the physical environment or the definition of a
monolithic service.

1. **Strict Fast reservation ledger.** A Monolithic action carries the sum of
   all stage compute. New Monolithic training therefore defaults to
   `fast_reservation_microbatch_size=1`, so every request updates predicted
   node work before the next request is scheduled. The earlier checkpoint used
   16 and allowed requests in one microbatch to make decisions against the same
   stale ledger.
2. **Stable Fast PPO for strict accounting.** Monolithic training uses a
   comparison-local Fast learning rate of `1e-4` (instead of the source
   checkpoint's `2e-4`) and a normalized entropy target of `0.50`. This keeps
   the four PPO epochs usable under the existing `0.015` KL limit and starts
   entropy protection before the route policy becomes nearly deterministic.
3. **Optional full feasible resource use with useful credit.**
   `--allow-full-resource-use` removes deployment memory/storage occupancy
   costs. Hard memory and storage constraints remain enforced. Small penalties
   are retained only for ineffective deployment: idle replicas `0.01`, unused
   count replicas `0.005`, and idle placement `0.005`. Thus useful replicas may
   consume all feasible capacity, while Slow still receives a signal that
   distinguishes useful capacity from replicas that never serve a request.

These controls are local to `train_monolithic.py`; Proposed training defaults
and the shared PPO implementation are unchanged.

The comparison runner also supports a diagnostic stochastic Fast mode with
explicit routing seeds and repeat averaging. It applies equally to Proposed,
Monolithic and other schemes that reuse the learned Fast policy. Deterministic
evaluation remains the default, preserving all historical results.

Neither implementation contains a performance cap or reads Proposed results.
The expectation that Monolithic remains weaker is an experimental hypothesis,
not an enforced outcome.

## Screening evidence before retraining

- Three fixed-seed stochastic repeats did not materially improve the old
  checkpoint: Monolithic 749.65 ms versus Proposed 470.80 ms.
- Applying a strict reservation ledger to the old checkpoint reduced
  Monolithic from 747.42 ms to 732.68 ms at the nominal Phase-1 point.
- Selecting Update 160 on an independent validation seed reduced the held-out
  nominal result only to 745.06 ms.

These checks show that a new training run is required; evaluation-only changes
cannot remove the structural performance gap.
