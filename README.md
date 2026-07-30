# Edge DRL for Staged Edge Services

This project implements the first runnable scaffold for the staged edge-service
optimization model described in `数学模型.docx` and `KKT条件推导.docx`.

The engineering layout mainly follows the useful shape of
`acsicuib/DRL-AC-Allocation`: environment logic is isolated from policy logic,
instances are generated reproducibly, and training/evaluation entrypoints are
kept thin. The problem model is different, so the code is specialized for:

- 10,000 to 15,000 users in a city-scale edge network.
- Staged services with at most 3 stages.
- Slow service-stage deployment every 4 hours.
- Fast request-level scheduling when each task request arrives.
- KKT closed-form allocation for continuous compute and link bandwidth.
- City-scale traffic derived from active users by default. For 10,000 users,
  the default traffic model produces about 29 requests/s on average and about
  43 requests/s at peak before optional CLI scaling.

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
python train_dual_ppo.py --updates 20 --requests-per-update 48 --eval-interval 5 --eval-seeds 2 --reward-scale 0.1
python scripts/run_full_training.py --fixed-scenario
python scripts/summarize_full_training.py runs
python scripts/analyze_convergence.py runs/phase2_joint/logs/training.csv
python -m pytest tests
```

Two agent families are available:

- `HierarchicalBaselineAgent`: deterministic baseline for sanity checks.
- `HierarchicalPPOAgent`: trainable dual-agent DRL scaffold.

The DRL version has a slow PPO agent for service-stage deployment and a fast PPO
agent for request-level stage scheduling. Continuous compute and bandwidth
allocation remains outside the neural policy and is solved by the KKT module.
