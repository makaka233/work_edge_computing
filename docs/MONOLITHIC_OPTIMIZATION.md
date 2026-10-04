# Monolithic optimization protocol

V4 corrects the Monolithic learning semantics without changing the physical
environment or the definition of a monolithic service.

1. **Strict Fast reservation ledger.** A Monolithic action carries the sum of
   all stage compute. New Monolithic training therefore defaults to
   `fast_reservation_microbatch_size=1`, so every request updates predicted
   node work before the next request is scheduled. The earlier checkpoint used
   16 and allowed requests in one microbatch to make decisions against the same
   stale ledger.
2. **Batch-level Fast trajectories.** A one-stage request used to make every
   Fast action terminal, reducing PPO to a contextual bandit even though later
   requests see the KKT reservations created by earlier requests. V4 keeps all
   requests settled together in one second in the same trajectory and marks
   only the final request terminal.
3. **Remaining-demand state.** Fast now observes the service, origin-node,
   compute, and input-data composition still waiting to be scheduled. This
   makes the strict sequential KKT ledger observable instead of asking the
   critic to infer hidden batch state.
4. **Training/evaluation Slow parity.** Evaluation preserves the previous
   complete-service deployment and feeds the previous window's latency,
   violation, load, and resource-use statistics to Slow, matching training.
5. **Restored Proposed controls.** Monolithic now inherits the source
   checkpoint's Fast learning rate, entropy controls, load-group weights, and
   Slow replica-ablation sampling. The V3 temperature policy remains loadable
   for historical checkpoints but is not used by V4.
6. **Checkpoint identity.** `best.pt` is saved before the PPO update because it
   is the policy that actually generated the selecting rollout. It is for
   evaluation only. `latest.pt`, `last.pt`, and periodic checkpoints are saved
   after the update and remain the resume points.
7. **Optional full feasible resource use with useful credit.**
   `--allow-full-resource-use` removes deployment memory/storage occupancy
   costs. Hard memory and storage constraints remain enforced. Small penalties
   are retained only for ineffective deployment: idle replicas `0.01`, unused
   count replicas `0.005`, and idle placement `0.005`. Thus useful replicas may
   consume all feasible capacity, while Slow still receives a signal that
   distinguishes useful capacity from replicas that never serve a request.

Monolithic-specific behavior is opt-in through the `aggregate-kkt-v4`
checkpoint tag. Proposed policy observations, actions, rewards, and optimizer
settings remain unchanged.

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
