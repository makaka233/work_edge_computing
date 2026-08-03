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
- Slow service-stage deployment every 4 hours.
- Fast request-level scheduling when each task request arrives.
- KKT closed-form allocation for continuous compute and link bandwidth.
- MEC-scale task latency calibration: service requests use small input payloads,
  single-digit Gcycle staged compute demand, 150 Mbps uplink, and 10 ms radio
  RTT so average single-task latency is expected to fall in the tens to hundreds
  of milliseconds range.
- City-scale traffic derived from active users by default, with a daily
  morning/lunch/evening curve. Use `--traffic-scale` above 1.0 to create
  heavier congestion.
- Demand pressure can be deliberately batched across PPO rollouts with
  `--load-multipliers` and `--rollout-start-mode cycle-window`, so one update
  can cover multiple traffic levels and 4-hour deployment windows.
- Fully connected wired metro links between edge nodes, with heterogeneous
  bottleneck, ordinary metro, and backbone-like bandwidth classes so placement
  and scheduling still have visible network tradeoffs.
- Compute and wired-link pressure can be calibrated without changing the fixed
  physical topology by using `--node-compute-capacity-scale` and
  `--wired-link-bandwidth-scale`.
- The physical edge infrastructure is fixed by `--physical-seed`: edge-node
  positions, compute capacity, memory, storage, service catalogue, and wired
  link bandwidth/propagation do not change during scenario refresh.
- Request aggregation is enabled by default. The environment groups arrivals
  within a short time window by `(home_node, service_id)` and stores the number
  of underlying requests in `request_count`. Per-task latency is evaluated with
  one task's compute and data demand, while metrics and dynamic load updates are
  weighted by `request_count`.
- Representative group sampling caps the number of aggregate events per window
  while rescaling selected groups so the underlying request count is preserved.

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
python train_dual_ppo.py --updates 2 --requests-per-update 64
python train_dual_ppo.py --train-mode joint --rollout-unit window --demand-sampling-mode rollout --rollouts-per-update 4 --updates 80 --num-users 12000 --num-edge-nodes 32 --num-service-types 10 --physical-seed 2026 --traffic-scale 4.0 --load-multipliers 1.0,1.35,1.7,2.1 --rollout-start-mode cycle-window --eval-rollout-unit window --eval-rollout-start-mode same --task-compute-scale 2.8 --task-data-scale 5.0 --node-compute-capacity-scale 0.45 --wired-link-bandwidth-scale 0.10 --eval-interval 10 --eval-seeds 4 --reward-mode latency --reward-scale 20 --fast-policy-kind gat_node_scorer --max-replicas-per-stage 0 --max-representative-groups-per-window 8 --compute-hotspot-threshold 0.45 --link-hotspot-threshold 0.35 --compute-hotspot-coef 0.12 --link-hotspot-coef 0.08 --compute-imbalance-coef 0.04 --link-imbalance-coef 0.03 --idle-deployed-node-coef 0.04 --slow-lr 0.0001 --slow-k-epochs 2 --slow-count-entropy-coef 0.02 --slow-placement-entropy-coef 0.005 --slow-value-coef 0.25 --fast-k-epochs 1 --fast-minibatch-size 1024 --device cuda --run-name joint_gat_city32_svc10_strong_pressure --save-best --progress-interval-seconds 10
python scripts/run_full_training.py --scenario-refresh-episodes 20 --traffic-scale 1.6
python scripts/summarize_full_training.py runs
python scripts/analyze_convergence.py runs/phase2_joint/logs/training.csv
python -m pytest tests
```

For the current convergence experiments, prefer `--rollout-unit window` with
`--demand-sampling-mode rollout`: each rollout samples an independent 4h demand
window while the physical edge network stays fixed. `train_dual_ppo.py` prints
in-rollout terminal progress by default every 10 seconds. The progress line
reports update progress, real request count, aggregate event count, simulated
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

When `--fixed-scenario` is omitted, demand-side variation can be sampled in two
ways while the physical edge network remains fixed by `--physical-seed`.
`--demand-sampling-mode episode` reuses one demand scenario for
`--scenario-refresh-episodes N` training episodes. `--demand-sampling-mode
rollout` samples a new demand scenario for every PPO rollout/update, which is
useful for convergence checks that intentionally ignore within-day temporal
structure. In both modes, only user locations, home-node assignment, and service
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
For convergence experiments, prefer `--rollout-start-mode cycle-window` with a
four-value `--load-multipliers` list when `--rollouts-per-update 4`; this makes
each PPO update see several pressure levels instead of repeatedly sampling the
same low-load 0:00-4:00 window.

The optimizer reward remains latency-centered by default:
`train_reward = -latency_s`. Resource-efficiency shaping can be enabled with
explicit coefficients:
`--compute-hotspot-coef`, `--link-hotspot-coef`,
`--compute-imbalance-coef`, `--link-imbalance-coef`, and
`--idle-deployed-node-coef`. These terms penalize hotspots, imbalance, and idle
deployed nodes instead of directly rewarding more replicas.

At each eval interval, `eval_*` fields report held-out demand seeds, while
`seen_eval_*` fields report fixed demand seeds from the training distribution.
This separates optimization on familiar demand profiles from demand
generalization on unseen profiles.

For high-variance demand-randomized PPO, `--rollouts-per-update K` can collect
multiple independent 4h rollout windows before one optimizer update. This is
closer to batched PPO sampling than updating after a single demand seed, and it
makes the logged training latency less dominated by one sampled demand profile.
Slow deployment exploration can also be controlled separately with
`--slow-count-entropy-coef` and `--slow-placement-entropy-coef`; the count policy
is especially sensitive because its action space is only the replica count.
