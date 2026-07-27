# World-Model Dual-Agent DRL for Edge Computing

This project implements a simulation and learning framework for staged service
deployment, task scheduling, and KKT-based resource allocation in an edge
computing network.

Core design:

- 10 edge nodes with heterogeneous compute, memory, storage, and links.
- Dynamic user counts and user-driven task generation.
- Services have at most 3 sequential stages.
- Agent-D updates service deployment on a slow time scale, such as 4 hours.
- Agent-S schedules each arriving task immediately, while the simulator settles
  resource allocation and reward once per second.
- Continuous compute and bandwidth allocation is solved analytically with KKT.
- A world model predicts next-second state and reward as a diagnostic module.

Run a smoke simulation:

```powershell
python train.py --config config/default.yaml --episodes 1 --seconds 30 --mode heuristic
```

Run the integrated neural trainer. This enables Agent-D slow deployment,
Agent-S event-driven scheduling, KKT allocation, PPO updates, and world-model
diagnostics in one loop:

```powershell
python train.py --config config/default.yaml --episodes 1 --seconds 10 --mode neural --run-name smoke_neural
```

Run Agent-S behavior cloning before neural training:

```powershell
python train.py --config config/default.yaml --mode neural --bc-seconds 10 --bc-epochs 3 --episodes 5 --seconds 60 --run-name bc_then_ppo
```

Run BC pretraining followed by PPO with validation-based checkpointing:

```powershell
python train.py --config config/default.yaml --mode neural --seed 7 --agent-s-top-k-actions 16 --bc-seconds 60 --bc-epochs 50 --bc-max-samples 12000 --episodes 100 --seconds 300 --agent-d-warmup-episodes 100 --ppo-lr 0.00005 --ppo-entropy-coef 0.001 --val-seconds 300 --val-every 5 --val-freeze-agent-d --run-name convergence_seed7_topk16
```

Evaluate a baseline:

```powershell
python evaluate.py --config config/default.yaml --mode heuristic --episodes 1 --seconds 10
```

Evaluate a saved neural checkpoint:

```powershell
python evaluate.py --mode neural --checkpoint runs/smoke_neural/checkpoints/best.pt --episodes 1 --seconds 10
```

Run tests:

```powershell
python -m unittest discover tests
```
