# IEEE formal scheme comparison

This framework is isolated under `edge_drl/comparison/`. It reuses the existing
physical simulator, radio model, multi-hop routing, contention model, and KKT
settlement without changing the training entry point or the Proposed policy.

The paper-facing schemes are exactly:

- **Proposed**: deterministic inference from an explicit checkpoint; no retraining.
- **Monolithic**: a separately trained copy of the same dual-scale PPO/KKT
  controller, with each service represented as one aggregated stage. It is not
  an MILP baseline and is not presented as a reproduction of SD3.
- **Hybrid**: the frozen Proposed Slow checkpoint controls stage deployment;
  a deterministic batch-aware shortest-delay algorithm schedules every
  request. This is a component-substitution comparison, not an independently
  trained learning method.
- **Algorithmic**: a demand-aware binary MILP places one mandatory replica per
  stage under memory/storage constraints, followed by marginal-benefit replica
  expansion. Its Fast scheduler is exactly the same one used by Hybrid. This
  is a model-driven comparison method and requires no training checkpoint.

The shared algorithmic scheduler routes each request through serial stages.
Its score is the marginal batch-delay cost predicted by the same square-root
KKT rule that the simulator uses for each serial phase, plus physical-path
propagation. It uses current node/link loads and only requests already seen
in the current batch; the simulator performs the final joint allocation.
The Algorithmic planner uses the service catalogue and user service-weight
distribution available in the current scenario. It never reads future slots
from the immutable evaluation trace. Planning time, solver status, added
replicas, and actual physical performance are logged separately.

The Algorithmic deployment method has two steps. First, a binary MILP picks
exactly one node for every valid `(service, stage)` pair. Each node's selected
stages must fit its service memory and storage capacities. Its linear costs
estimate demand-weighted access, stage computation, and intermediate transfer
using the current physical paths and load. Second, it repeatedly adds the
feasible replica with the largest reduction in a serial-chain forecast per
unit of consumed memory/storage fraction. The forecast includes a pooled
compute-capacity congestion term proportional to the current expected request
rate. Candidate gains for one service are recalculated only when that service's
deployment changes; the search stops when no positive feasible gain remains.
There is no additional replica-count limit or wall-clock truncation in this
second step. The mandatory MILP itself has a five-second solver limit; a
missing feasible solution is reported as a failed evaluation, not silently
replaced with a different method.

For the shared Fast algorithm, each incoming request is processed in trace
order. A dynamic program finds the lowest-cost reachable node chain over the
currently deployed replicas. The cost of adding demand `d` to a compute node
or physical link in its causal phase is `((S + sqrt(d))^2 - S^2) / C`, where
`S` is the square-root workload already reserved in the current second and
`C` is its load-adjusted capacity. Multi-hop propagation is added on the same
physical route used by the simulator. After choosing a chain, the algorithm
reserves its work before scheduling the next request in the batch. The actual
metric still comes from the simulator's joint KKT settlement. This procedure
is designed for the project's binary deployment and serial workload model;
it has no inherited approximation guarantee from DMDR or SICP.

During evaluation, the original multi-stage request shape is retained for the
shared physical simulator: compute is `(sum(stage_compute), 0, ...)`, all
inter-stage outputs are zero, and the aggregate Fast decision is projected to
the same node for every original stage. Thus the only structural difference is
the stage aggregation, not the trace, resource accounting, or simulator.

Greedy remains available only as a development sanity check and is never added
to formal plots or the main result tables.

Every evaluation point and seed first creates a complete immutable one-second
request trace. Each scheme receives the identical trace and an independent copy
of the same transformed physical scenario. Formal episodes contain 3,600
logical and settlement steps (60 minutes); Phase 1 intentionally uses 600 steps
for correctness screening. By default comparison runs are unsampled. Set
`--sampled-seconds-per-window 6` for training-equivalent weighted settlement;
this is an approximate evaluation and does not produce formal validation.

## Phases

1. Phase 1: nominal load, one seed, 10 minutes.
2. Phase 2: all five parameter sweeps, three paired seeds, 60 minutes.
3. Phase 3: all sweeps, twenty paired seeds, 60 minutes. Each scheme is evaluated
   once per point and seed; statistics use paired seeds.
   The runner refuses Phase 3 unless `--phase2-validation-run` points to a complete,
   failure-free Phase 2 result directory with the same four-scheme registry.

Run Phase 1 from PowerShell:

```powershell
python run_comparison.py `
  --checkpoint "runs\global_incremental_cf_s6_u180_20260903_192623\checkpoints\best.pt" `
  --monolithic-checkpoint "runs\monolithic_global_incremental_cf_s6_u180_20260904_094837\checkpoints\best.pt" `
  --monolithic-checkpoint-mode fixed `
  --phase 1 `
  --device cuda
```

For a quick six-settlement correctness check of the two new schemes, use a
different run ID and temporal sampling. This exploratory run cannot validate
Phase 2:

```powershell
$runId = "algorithmic_smoke_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

python .\run_comparison.py `
  --checkpoint ".\runs\global_incremental_cf_s6_u180_20260903_192623\checkpoints\best.pt" `
  --schemes Hybrid Algorithmic `
  --phase 1 `
  --sampled-seconds-per-window 6 `
  --device cuda `
  --run-id $runId
```

Train the Monolithic checkpoint separately before comparison. The trainer
inherits the Proposed checkpoint's physical seed, demand random streams, load
strata and weights, episode length, temporal sampling, and PPO controls by
default. The service is collapsed to one stage; V4 then uses batch-level Fast
credit and a remaining-demand observation so that the one-stage action still
has valid sequential PPO semantics:

```powershell
python .\train_monolithic.py `
  --base-checkpoint ".\runs\global_incremental_cf_s6_u180_20260903_192623\checkpoints\best.pt" `
  --updates 320 `
  --episode-minutes 60 `
  --run-root ".\runs\monolithic_checkpoints" `
  --run-name request_load_1 `
  --device cuda `
  --progress-interval-seconds 10
```

The omitted seed, episode length, and sampled-seconds options are copied from
the Proposed checkpoint (the frozen reference uses 6 representative settlement
seconds per window).  Monolithic collects all six ten-minute windows of a
60-minute episode and uses the same per-window load assignments and random
streams before collapsing the generated trace.  Evaluation is unsampled unless explicitly configured otherwise.
The run writes `training_manifest.json`, including the physical seed, demand
seed, environment/request seed, load assignments, and trace hash for every
episode so parity can be audited.

Monolithic fixes NumPy and PyTorch initialization from the same master seed
before constructing the policy. Every resumable update-boundary checkpoint
also stores the model, optimizers, adaptive entropy state, Slow critic replay,
Slow learning rate controller, load/demand/environment random streams, and
global counters. `best.pt` is an evaluation-only snapshot of the exact
pre-update policy that produced its selecting rollout; resume from `latest.pt`,
`last.pt`, or a periodic `update_XXXX.pt` checkpoint.
If a run is interrupted after update 66 and 254 updates remain, resume into a
new run directory as follows (`--updates` always means additional updates when
`--load-checkpoint` is present):

```powershell
$runName = "monolithic_resume_u66_r254_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

python .\train_monolithic.py `
  --base-checkpoint ".\runs\global_incremental_cf_s6_u180_20260903_192623\checkpoints\best.pt" `
  --load-checkpoint ".\runs\monolithic_checkpoints\<interrupted-run>\checkpoints\latest.pt" `
  --updates 254 `
  --episode-minutes 60 `
  --episodes-per-update 2 `
  --sampled-seconds-per-window 6 `
  --scenario-family request_load `
  --scenario-value 1.0 `
  --run-root ".\runs\monolithic_checkpoints" `
  --run-name $runName `
  --device cuda `
  --progress-interval-seconds 10
```

For the formal Phase 2/3 protocol, freeze one Proposed checkpoint and one
Monolithic checkpoint trained on the same nominal training distribution and
budget. Reuse those two checkpoints unchanged over all five scenario families
and 28 parameter points.  This makes stage decomposition the learning schemes'
only structural difference and uses the sweeps to measure generalization.
Hybrid loads the same Proposed checkpoint and replaces only Fast. Algorithmic
uses the same scenario and causal path scheduler as Hybrid, but computes its
own deployment at each Slow boundary. Its MILP is an adaptation to this
project's binary placement model; neither it nor Hybrid is a reproduction of
DMDR or SICP. Compare the four schemes with the common environment metrics,
and report the model-driven planning time alongside latency and replica use.
`--monolithic-checkpoint-mode fixed` is therefore the default and requires a
checkpoint file.  The legacy `per-point` directory/template resolver remains
available only for explicitly labelled exploratory experiments; its results
must not be mixed into the formal fixed-checkpoint tables.

Plot a completed run:

```powershell
python scripts\plot_comparison.py "results\comparison\<run_id>"
```
