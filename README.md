# Edge DRL for Staged Edge Services

This project implements the first runnable scaffold for the staged edge-service
optimization model described in `数学模型.docx` and `KKT条件推导.docx`.

The engineering layout mainly follows the useful shape of
`acsicuib/DRL-AC-Allocation`: environment logic is isolated from policy logic,
instances are generated reproducibly, and training/evaluation entrypoints are
kept thin. The problem model is different, so the code is specialized for:

- 10,000 to 15,000 users in a city-scale edge network.
- 32 heterogeneous edge nodes for the main experiment setting.
- Staged services with at most 3 stages.
- 10 realistic service classes: speech, AR, video analytics, industrial
  inspection, traffic perception, retail events, robot control, medical vital
  anomaly detection, drone inspection, and connected-vehicle planning.
- Slow service-stage deployment every 10 minutes.
- Fast scheduling for every individual request in each one-second step.
- KKT closed-form allocation for continuous compute and link bandwidth.
- MEC-scale task latency calibration: service requests use small input payloads,
  single-digit Gcycle staged compute demand, 150 Mbps uplink, and 10 ms radio
  RTT so average single-task latency is expected to fall in the tens to hundreds
  of milliseconds range.
- City-scale traffic derived from active users. The default training episode is
  a stationary 4-hour trajectory containing 24 slow-deployment windows; the legacy daily
  morning/lunch/evening curve remains available with `--arrival-profile daily`.
  Use `--traffic-scale` above 1.0 to create heavier congestion.
- Demand pressure can be deliberately batched across PPO rollouts with
  `--load-multipliers`, so one update can cover multiple traffic levels and
  independent rollouts.
- Fully connected wired metro links between edge nodes, with heterogeneous
  bottleneck, ordinary metro, and backbone-like bandwidth classes so placement
  and scheduling still have visible network tradeoffs.
- Compute and wired-link pressure can be calibrated without changing the fixed
  physical topology by using `--node-compute-capacity-scale` and
  `--wired-link-bandwidth-scale`.
- The physical edge infrastructure is fixed by `--physical-seed`: edge-node
  positions, compute capacity, memory, storage, service catalogue, and wired
  link bandwidth/propagation do not change during scenario refresh.
- A normal `env.step` equals one simulated second. Arrivals remain as individual
  `TaskRequest` objects with `request_count=1`; no node-service aggregation or
  representative-request reassignment is performed.
- The fast policy schedules every request before settlement. Compute and link
  demands from all requests then enter one joint KKT allocation, loads update
  once, and time advances one second. This preserves batch contention while
  allowing different requests to use different deployed replicas.
- Fast-policy state includes the current second's total request count, request
  event count, service share, and per-access-node demand share, so each request
  action can respond to the batch it will compete with.

## Current Modules

- `edge_drl/env/scenario.py`: realistic synthetic MEC scenario generator.
- `edge_drl/env/environment.py`: event-driven environment and action masks.
- `edge_drl/allocators/kkt.py`: KKT sqrt-rule resource allocator.
- `edge_drl/agents/hierarchical.py`: slow greedy deployment and fast greedy scheduler baseline.
- `edge_drl/agents/drl.py`: trainable hierarchical dual-agent PPO scaffold.
- `edge_drl/models/ppo.py`: masked categorical PPO core, following the rollout-memory style used by DRL-AC-Allocation.
- `train_dual_ppo.py`: slow deployment PPO + fast scheduling PPO training entrypoint.
- `tests/test_env_smoke.py`: KKT and environment smoke tests.
- `tests/test_dual_ppo_smoke.py`: dual-agent PPO rollout/update tests.

## Run

```powershell
python train_dual_ppo.py --train-mode fast-only --rollout-unit requests --updates 2 --requests-per-update 64
python train_dual_ppo.py --train-mode joint --rollout-unit window --episode-hours 4 --deployment-interval-minutes 10 --sampled-seconds-per-window 60 --arrival-profile stationary --demand-sampling-mode episode --fast-windows-per-update 1 --slow-windows-per-update 12 --updates 80 --num-users 12000 --num-edge-nodes 32 --num-service-types 10 --physical-seed 2026 --traffic-scale 1.0 --load-multipliers 1.0 --eval-rollout-unit window --eval-interval 10 --eval-seeds 1 --reward-mode latency --reward-scale 10 --fast-policy-kind gat_node_scorer --max-replicas-per-stage 0 --slow-lr 0.0001 --slow-k-epochs 2 --slow-count-entropy-coef 0.02 --slow-placement-entropy-coef 0.005 --fast-k-epochs 2 --fast-minibatch-size 512 --device cuda --run-name joint_gat_decoupled_updates --save-best --progress-interval-seconds 10
python scripts/run_full_training.py --scenario-refresh-episodes 20 --traffic-scale 1.6
python scripts/summarize_full_training.py runs
python scripts/analyze_convergence.py runs/phase2_joint/logs/training.csv
python -m pytest tests
```

For the current convergence experiments, one 4-hour episode contains 14,400
one-second steps and 24 ten-minute slow-deployment windows. Prefer
`--rollout-unit window --episode-hours 4` with `--demand-sampling-mode rollout`:
each rollout samples one independent ten-minute demand window while the physical
edge network stays fixed. `train_dual_ppo.py` prints
in-rollout terminal progress by default every 10 seconds. The progress line
reports update progress, real request count, request event count, simulated
hours, deployment updates, average latency, elapsed time, and ETA. Use
`--progress-interval-seconds 0` to disable it or a smaller value for more
frequent refreshes.

Two agent families are available:

- `HierarchicalBaselineAgent`: deterministic baseline for sanity checks.
- `HierarchicalPPOAgent`: trainable dual-agent DRL scaffold.

The DRL version has a slow deployment agent for service-stage deployment and a
fast PPO agent for request-level stage scheduling. The slow deployment agent
contains two PPO policies: `count_ppo` first chooses the replica count
`k in [1, --max-replicas-per-stage]`, then `placement_ppo` chooses `k` distinct
nodes under memory/storage masks. Use `--max-replicas-per-stage 0` to remove
the artificial replica cap; the physical maximum then becomes the number of
edge nodes because duplicate placement on the same node is masked. Continuous
compute and bandwidth allocation remains outside the neural policy and is solved
by the KKT module.

Replica count is an ordered action rather than 32 unrelated classes. The Count
head predicts a distribution center and scale, which define a masked discretized
Gaussian over feasible replica counts. Nearby counts therefore share statistical
strength and the deterministic expected-count decoder remains consistent with the
stochastic training policy. Its initial scale is one sixth of the action range,
which preserves local exploration without keeping a 32-count policy nearly uniform.

Slow deployment is trained from one 10-minute window at a time, but Count and
Placement no longer receive one undifferentiated return for every component
action. Placement receives service-stage mean/P95 latency, deadline return, and
the observed cross-node transition rate of the adjacent stages owned by that
placement action. The colocation term is therefore stage-local actor credit rather
than only a global window diagnostic.
Count receives a separate dense U-shaped return. Service-stage mean/P95 latency
and deadline violations penalize under-provisioning, while a continuous effective-
replica measure based on request shares penalizes replicas that add little usable
capacity. This avoids treating a replica as fully useful merely because it handled
one request during a high-traffic window. Placement still trains its value head.
Count instead uses direct stage-centered returns with
`--slow-count-value-coef 0` by default, so its failed critic cannot dominate the
shared graph encoder; value loss and explained variance remain logged for ablation
diagnostics but are not optimized. The
separate window critic remains as a high-level diagnostic. Fast PPO receives an additive stage-local latency reward
plus the exact KKT difference reward for compute/link congestion imposed on other
requests. Requests are inferred in microbatches (default 16); after each
microbatch a virtual workload ledger reserves the selected-node work so later
requests do not see stale batch load. Both Slow actors use a topology-aware
graph-attention encoder; the placement head scores nodes and the count head scores
replica-count actions. The Slow state includes node/link capacity and load plus
the previous window's mean latency, P95 latency, penalty latency, and deadline
feedback. The global slow window diagnostic combines mean and P95 latency according
to `--slow-tail-latency-coef` (default 0.35) with deployment, idle-replica, deadline,
and migration costs.
Migration changes remain logged, but `--slow-migration-coef` defaults to zero for
the current convergence experiments.
Fast and Slow PPO use alternating frozen-controller phases by default. Four Fast
warm-up updates first learn under the conservative expected-count Slow deployment.
Four Slow warm-up updates then collect 32 windows each with the now-trained Fast
policy frozen. Afterwards, three Fast updates collect stochastic Fast actions
under deterministic Slow deployment;
the next Slow update freezes Fast deterministically and collects 32 independent
window returns. Because replica count is ordinal, deterministic Slow deployment
uses the ceiling of the Count distribution's expected replica count instead of an
unstable categorical argmax; use `--slow-deterministic-count-mode mode` only for an
explicit mode ablation. This prevents one Slow PPO batch from mixing returns
produced by several different Fast policies. Count actor advantages use direct
Monte-Carlo returns centered within each service stage before normalization, so
intrinsic latency differences between easy and expensive stages do not overwhelm
the replica-count comparison. Set a nonzero `--slow-count-value-coef` only for a
critic ablation; doing so reintroduces shared-backbone critic gradients.
Configure the cadence with `--fast-warmup-updates`, `--slow-warmup-updates`,
`--fast-updates-per-cycle`,
`--fast-windows-per-update`, and `--slow-windows-per-update`; use
`--joint-training-schedule simultaneous` only for legacy ablations. The log field
`training_phase` records which policy was active.
For a synchronized training block, use `--synchronized-window-block 4`; this
sets both PPO update periods to four windows and selects simultaneous mode. Four windows are useful for a quick
smoke run, but are usually too few for Slow PPO when four load multipliers are
cycled because each pressure level contributes only one sample. Prefer independent
alternating collection with at least 32 Slow windows for convergence runs. With `--demand-sampling-mode episode`
and multiple `--load-multipliers`, the fixed user distribution is retained while
each window in the block receives the next load multiplier, for example
`0.8,1.1,1.4,1.7`.
Training windows use stratified temporal approximation by default:
`--sampled-seconds-per-window 60` performs 60 neural-policy/KKT settlements that
represent all 600 logical seconds in a ten-minute window. Instantaneous KKT
contention still uses one sampled second of arrivals; request metrics, the Slow
return, and EWMA load evolution use the represented-time weight. This creates
one Slow transition per logical window, not 60 Slow transitions. Periodic eval
always executes the full window. Use `--sampled-seconds-per-window 0` to restore
full training rollouts. Logs distinguish `settlement_steps`, `logical_steps`,
and `temporal_sampling_fraction`.
`--service-resource-fraction` fixes the share of each physical
node's memory/storage available to this controller, representing system and
co-tenant reservations without imposing a per-service replica-count cap.
The physical metro graph is a connected symmetric k-nearest-neighbor topology
(`--topology-k-nearest 6` by default). Transfers use cached multi-hop routes and
consume bandwidth plus propagation delay on every physical edge, rather than
treating all edge nodes as directly connected.

When `--fixed-scenario` is omitted, demand-side variation can be sampled in two
ways while the physical edge network remains fixed by `--physical-seed`.
`--demand-sampling-mode episode` reuses one demand scenario for
`--scenario-refresh-episodes N` training episodes. `--demand-sampling-mode
rollout` samples a new demand scenario for every PPO rollout. With the default
4-hour horizon, an episode contains multiple rollout windows. In both modes, only user
locations, home-node assignment, and service
preferences may change; nodes, capacities, service catalogue, and wired links do
not change. Request samples still change every rollout. Eval seeds therefore
check demand generalization on the same edge infrastructure, not a different
physical network.

Training logs include deployment size and resource diagnostics: node compute
EWMA load, wired-link EWMA load, memory/storage deployment utilization, and the
fraction of nodes with at least one deployed stage. They also include load
standard deviation, active-node/link rates, hot-node/link rates, and the idle
deployed-node rate so resource efficiency can be diagnosed separately from
latency. Scheduler utilization diagnostics report whether fast scheduling uses
the deployed service replicas: `used_replica_rate`, `idle_replica_rate`,
`used_replicas_per_stage`, normalized replica-use entropy,
top-1 replica-use share, and cross-node stage transition rate.

Demand-side load can be raised without changing physical infrastructure. Use
`--traffic-scale`, `--active-user-ratio`, and
`--active-user-request-rate-per-minute` to increase arrival volume. Use
`--task-compute-scale` and `--task-data-scale` to make each sampled task heavier
in CPU cycles or transferred data. Use `--node-compute-capacity-scale` and
`--wired-link-bandwidth-scale` below 1.0 when the fixed physical environment is
too over-provisioned for the desired experiment. These capacity scales are part
of the fixed physical scenario for a run and remain tied to `--physical-seed`.

For reproducible pressure experiments, `--pressure-profile mec-moderate` applies
the following fixed-run operating point: 20% active users, 1.75 requests per active
user per minute, 1.65x task compute, 2.5x task data, 0.65x node compute capacity,
0.15x wired-link bandwidth, 0.25 service resource fraction, and rollout load
multipliers `0.8,1.1,1.4,1.7` with 2.75x deadline scale. This keeps the topology, node locations, node
tiers, service catalogue, and link classes fixed while making both compute and
network pressure visible. `--pressure-profile mec-stress` is a stronger bounded
ablation. Explicit scale flags override a profile, and the resolved values are
stored in `metadata.json`.

The moderate profile is intended as the first pressure test. It targets a
moderate operating point rather than immediate saturation: if diagnostics show
persistent infeasible actions or penalty latency, use the moderate profile as a
calibration point before trying `mec-stress`.
For demand-randomized convergence experiments, load multipliers can still be
cycled across consecutive Fast windows. The stationary profile ignores rollout
start modes. Use
`--arrival-profile daily --episode-hours 24 --rollout-start-mode cycle-window`
only for legacy experiments that intentionally model within-day timing.

The optimizer reward remains latency-centered by default:
`train_reward = -latency_s`. Resource-efficiency shaping can be enabled with
explicit coefficients:
`--compute-hotspot-coef`, `--link-hotspot-coef`,
`--compute-imbalance-coef`, `--link-imbalance-coef`, and
`--idle-deployed-node-coef`. These terms penalize hotspots, imbalance, and idle
deployed nodes instead of directly rewarding more replicas.

At each eval interval, `eval_*` fields report one held-out deterministic rollout
by default. Set `--eval-seeds` above 1 only when an explicit multi-seed sweep is
needed. Periodic training no longer launches separate seen-demand and policy
diagnostic rollouts.

For high-variance demand-randomized PPO, `--fast-windows-per-update K` can still
collect multiple independent ten-minute windows before one Fast optimizer
update. `--rollouts-per-update` remains a compatibility alias.
Slow deployment exploration can also be controlled separately with
`--slow-count-entropy-coef` and `--slow-placement-entropy-coef`; the count policy
is especially sensitive because its action space is only the replica count.
