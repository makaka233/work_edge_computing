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
- Fully connected wired metro links between edge nodes, with heterogeneous
  bottleneck, ordinary metro, and backbone-like bandwidth classes so placement
  and scheduling still have visible network tradeoffs.
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
- `train.py`: runnable rollout entrypoint.
- `train_dual_ppo.py`: slow deployment PPO + fast scheduling PPO training smoke entrypoint.
- `tests/test_env_smoke.py`: KKT and environment smoke tests.
- `tests/test_dual_ppo_smoke.py`: dual-agent PPO rollout/update tests.

## Run

```powershell
python train.py --max-requests 1000
python train_dual_ppo.py --updates 2 --requests-per-update 64
python train_dual_ppo.py --updates 20 --requests-per-update 48 --eval-interval 5 --eval-seeds 2 --reward-scale 10
python train_dual_ppo.py --train-mode joint --rollout-unit episode --episode-hours 24 --updates 200 --num-users 12000 --num-edge-nodes 32 --num-service-types 10 --scenario-refresh-episodes 20 --traffic-scale 1.6 --eval-interval 20 --eval-rollout-unit episode --eval-seeds 5 --reward-mode latency --reward-scale 10 --fast-policy-kind gat_node_scorer --max-replicas-per-stage 5 --max-representative-groups-per-window 16 --run-name joint_gat_city32_svc10_day200 --save-best --progress-interval-seconds 30
python scripts/run_full_training.py --scenario-refresh-episodes 20 --traffic-scale 1.6
python scripts/summarize_full_training.py runs
python scripts/analyze_convergence.py runs/phase2_joint/logs/training.csv
python -m pytest tests
```

Use `--rollout-unit episode` for serious training: each PPO update then collects
one complete environment episode before optimizing, so the logged `update` and
`episode` advance together. `train_dual_ppo.py` prints in-rollout terminal
progress by default every 10 seconds. The progress line reports update progress,
real request count, aggregate event count, simulated hours, episode fraction,
deployment updates, average latency, elapsed time, and ETA. Use
`--progress-interval-seconds 0` to disable it or a smaller value for more
frequent refreshes.

Two agent families are available:

- `HierarchicalBaselineAgent`: deterministic baseline for sanity checks.
- `HierarchicalPPOAgent`: trainable dual-agent DRL scaffold.

The DRL version has a slow deployment agent for service-stage deployment and a
fast PPO agent for request-level stage scheduling. The slow deployment agent
contains two PPO policies: `count_ppo` first chooses the replica count
`k in [1, --max-replicas-per-stage]`, then `placement_ppo` chooses `k` distinct
nodes under memory/storage masks. Continuous compute and bandwidth allocation
remains outside the neural policy and is solved by the KKT module.

When `--fixed-scenario` is omitted, `--scenario-refresh-episodes N` refreshes
only demand-side variation after N training episodes: user locations, home-node
assignment, and service preferences may change, while the physical edge network
remains fixed by `--physical-seed`. Request samples still change every episode.
Eval seeds therefore check demand generalization on the same edge
infrastructure, not a different physical network.
