# Placement stability recovery

Baseline commit: `0917387`.

The first hardware-constrained run reached its load-adjusted best checkpoint at
update 129, then regressed across every load stratum after update 170.  The
diagnostics separated that regression from Fast KL and the Slow window critic:

- Fast normally completed all four PPO epochs.
- Slow critic train and holdout explained variance remained high.
- Placement entropy fell below its 1.10 target and stayed low after its 0.015
  coefficient ceiling saturated.
- The explicit cross-stage transition penalty remained constant even though
  end-to-end latency already included the measured link cost.
- Placement's very small per-update KL permanently blocked the degradation LR
  guard, despite harmful cumulative policy drift.

Trajectory-simultaneous training now applies three coordinated controls:

1. Explicit colocation credit is early shaping only. It remains at 0.05 for 64
   updates, linearly reaches zero over the next 96 updates, and is then removed.
   Actual compute and link latency remain in the objective throughout training.
2. Placement entropy recovery uses a 0.030 coefficient ceiling and a 0.004
   adaptation rate. Legacy training keeps its original defaults unless these
   options are explicitly selected.
3. When Placement entropy is below target, sustained score degradation may
   bypass the low-KL block and reduce only the Placement learning rate. Count
   retains its independent low-KL protection.

New logs expose `slow_colocation_credit_coef` for every rollout and
`slow_placement_lr_low_entropy_overrides` for every update. Existing checkpoint
formats remain loadable because new controller counters are optional on restore,
and all schedules derive from the completed-update offset.
