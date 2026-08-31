# IEEE formal scheme comparison

This framework is isolated under `edge_drl/comparison/`. It reuses the existing
physical simulator, radio model, multi-hop routing, contention model, and KKT
settlement without changing the training entry point or the Proposed policy.

The paper-facing schemes are exactly:

- **Proposed**: deterministic inference from an explicit checkpoint; no retraining.
- **Monolithic**: a separately trained copy of the same dual-scale PPO/KKT
  controller, with each service represented as one aggregated stage. It is not
  an MILP baseline and is not presented as a reproduction of SD3.

During evaluation, the original multi-stage request shape is retained for the
shared physical simulator: compute is `(sum(stage_compute), 0, ...)`, all
inter-stage outputs are zero, and the aggregate Fast decision is projected to
the same node for every original stage. Thus the only structural difference is
the stage aggregation, not the trace, resource accounting, or simulator.
- **DMDR**: native AES-JDR/RDMP adaptation following Peng et al., IEEE TSC 2024.
  Integer instance multiplicity is retained as a native diagnostic; the common
  physical simulator projects it to `N_integer > 0`, and memory/storage are
  enforced on that indicator.
- **SICP**: adapted JPS-CP chain placement with OR-Tools CP-SAT; TSN gate
  scheduling is outside this MEC simulator and is explicitly omitted.

Greedy remains available only as a development sanity check and is never added
to formal plots or the main result tables.

Every evaluation point and seed first creates a complete immutable one-second
request trace. Each scheme receives the identical trace and an independent copy
of the same transformed physical scenario. Formal episodes contain 3,600
logical and settlement steps (60 minutes); Phase 1 intentionally uses 600 steps
for correctness screening. No temporal sampling is used in comparison runs.

## Phases

1. Phase 1: nominal load, one seed, 10 minutes.
2. Phase 2: all five parameter sweeps, three paired seeds, 60 minutes.
3. Phase 3: all sweeps, twenty paired seeds, 60 minutes; DMDR routing is repeated
   three times and averaged within each seed before Student-t confidence intervals.
   The runner refuses Phase 3 unless `--phase2-validation-run` points to a complete,
   failure-free Phase 2 result directory.

Run Phase 1 from PowerShell:

```powershell
python run_comparison.py `
  --checkpoint "runs\joint_1to1_stability_large_s30_u320_20260831_121230\checkpoints\best.pt" `
  --monolithic-checkpoint "runs\monolithic_checkpoints\request_load_1\checkpoints\best.pt" `
  --monolithic-checkpoint-mode fixed `
  --phase 1 `
  --device cuda
```

Train the Monolithic checkpoint separately before comparison.  The trainer
inherits the Proposed checkpoint's physical seed, demand random streams, load
strata, episode length, and temporal sampling by default; the only training
structure change is the one-stage collapse:

```powershell
python .\train_monolithic.py `
  --base-checkpoint ".\runs\joint_1to1_stability_large_s30_u320_20260831_121230\checkpoints\best.pt" `
  --updates 320 `
  --episode-minutes 60 `
  --run-root ".\runs\monolithic_checkpoints" `
  --run-name request_load_1 `
  --device cuda `
  --progress-interval-seconds 10
```

The omitted seed, episode length, and sampled-seconds options are copied from
the Proposed checkpoint (the reference run uses 30 representative settlement
seconds per window).  Monolithic collects all six ten-minute windows of a
60-minute episode and uses the same per-window load assignments and random
streams before collapsing the generated trace.  Evaluation remains unsampled.
The run writes `training_manifest.json`, including the physical seed, demand
seed, environment/request seed, load assignments, and trace hash for every
episode so parity can be audited.

For the formal Phase 2/3 protocol, freeze one Proposed checkpoint and one
Monolithic checkpoint trained on the same nominal training distribution and
budget.  Reuse those two checkpoints unchanged over all five scenario families
and 28 parameter points.  This makes stage decomposition the learning schemes'
only structural difference and uses the sweeps to measure generalization.
`--monolithic-checkpoint-mode fixed` is therefore the default and requires a
checkpoint file.  The legacy `per-point` directory/template resolver remains
available only for explicitly labelled exploratory experiments; its results
must not be mixed into the formal fixed-checkpoint tables.

Plot a completed run:

```powershell
python scripts\plot_comparison.py "results\comparison\<run_id>"
```
