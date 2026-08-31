# Proposed experiment freeze (2026-08-31)

This record identifies the immutable Proposed baseline used by the formal
scheme-comparison workflow.

- Branch: `improve-slow-critic-credit-v2`
- Proposed source commit: `d4dc589` (`Stabilize joint PPO training and add rollout diagnostics`)
- Run: `runs/joint_1to1_stability_large_s30_u320_20260831_121230`
- Selected checkpoint: `checkpoints/best.pt`
- Selected update: 307
- Load-adjusted rolling score: 0.20103654785409977 s
- Checkpoint SHA-256: `EF90BD1908B23F5AD87FFD1C5F7A0149988EC4B7DBD37FB4951D91652E826612`

The run completed 320 updates, 640 sixty-minute episodes and 3,840 ten-minute
rollout windows.  The checkpoint metadata is authoritative for training
settings.  In particular, it records 6 representative settlement seconds per
ten-minute training window; the `s30` text in the historical run name is not a
configuration source.  Formal comparison evaluation remains unsampled and
executes all 3,600 logical/settlement seconds in every 60-minute episode.

The formal comparison freezes one Proposed checkpoint and one independently
trained Monolithic checkpoint.  Both are evaluated unchanged over all five
scenario families and 28 parameter points.  Per-point retraining is retained
only as an explicitly labelled exploratory protocol.
