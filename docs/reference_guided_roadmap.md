# Reference-Guided Roadmap

This roadmap pauses narrow hyperparameter tuning and redirects the project
toward a more evidence-based architecture. The goal is stable convergence for a
single-seed training pipeline before broad robustness checks.

## Reference Lessons

### AirFogSim-style simulation structure

AirFogSim is useful mainly as an environment-design reference. The important
lesson is to keep the simulator modular and event-driven:

- request generation
- service-chain definitions
- network and compute resources
- deployment control
- task scheduling
- resource allocation
- metric collection

Current project impact: `EdgeComputingEnv` already contains most of these
concepts, but too much of the metric and policy-facing logic is still embedded
inside one class. The next environment refactor should extract metric
collection and policy action filtering into separate modules.

### NFVdeep / SFC deployment lesson

Service function chain deployment has a large discrete action space. Relevant
systems avoid selecting the whole placement matrix as one flat action. Instead,
they serialize placement decisions and use feasibility repair or backtracking.

Current project impact: Agent-D should not keep learning by sampling a full
service-stage-node deployment tensor at once. That makes the action too sparse,
hard to credit, and hard to repair meaningfully. Agent-D should become a
sequential placement policy:

```text
for each service stage in a deployment cycle:
    observe global state + partial placement
    choose one node or replica action
    apply feasibility mask
    update partial placement
repair / reject only when no feasible action remains
```

This keeps the slow-timescale deployment idea, but makes the action model closer
to known SFC deployment approaches.

### MARL offloading + convex allocation lesson

Several edge-computing works decouple task offloading from resource allocation:
DRL selects task placement/offloading, while the continuous allocation problem
is solved analytically or by convex optimization.

Current project impact: the current split is sound:

- Agent-S chooses staged service paths.
- KKT solves continuous compute and bandwidth allocation.

Do not merge KKT into the policy network. The learning problem should focus on
discrete scheduling/deployment, while resource allocation remains analytical.

### SLA-aware scheduling lesson

SLA-aware DRL systems do not optimize only average latency. They introduce
deadline or risk features and evaluate violation rates, tail latency, and
unsafe actions.

Current project impact: average delay is too weak as the only convergence
target. The environment should expose:

- average delay
- P95 / P99 delay
- SLA violation rate
- invalid/rejected task rate
- node utilization imbalance
- link utilization imbalance
- deployment cost or churn

Training can still start with a compact reward, but convergence should be
judged by this fuller metric set.

### Experiment-management lesson

Open MEC RL projects usually separate:

- training
- evaluation
- result aggregation
- plotting/reporting

Current project impact: the new `summarize_convergence.py` is a first step, but
we still need a fixed experiment protocol that compares the same seed,
same request trajectory, same deployment mode, and same metric set.

## Project Direction

### Keep

- user-driven request generation with tidal patterns
- 10 heterogeneous edge nodes
- staged services with at most 3 stages
- event-driven Agent-S decisions
- one-second simulator update
- KKT continuous resource allocation
- slow Agent-D deployment interval, at least 14400 simulated seconds
- strict same-seed evaluation

### Rework

- Agent-D full-matrix deployment action
- environment metric collection
- convergence reporting
- reward design
- training phase boundaries

### Avoid For Now

- more Top-K sweeps
- world-model imagined rollout optimization
- task-level heuristic auxiliary rewards
- simultaneous Agent-S and Agent-D training before Agent-S is stable

## New Training Architecture

### Phase 1: Agent-S Stabilization

Purpose: prove that event-driven scheduling can converge under fixed deployment.

Settings:

- one seed only
- Agent-D frozen
- fixed validation seed
- KKT enabled
- BC pretraining allowed
- PPO training allowed
- validation every fixed number of episodes

Required evidence:

- best checkpoint beats heuristic on the same validation trajectory
- last or restored policy does not regress above heuristic
- invalid task count remains zero or bounded
- value loss and explained variance do not indicate collapse

### Phase 2: Metric-Rich Reward

Purpose: stop optimizing only mean latency.

Add metric collector outputs:

- `avg_delay`
- `p95_delay`
- `sla_violation_rate`
- `invalid_total`
- `node_imbalance`
- `link_imbalance`

Reward should become a weighted objective:

```text
reward = -(
    avg_delay
    + alpha * p95_delay
    + beta * sla_violation_rate
    + gamma * invalid_count
    + eta * imbalance
)
```

The exact weights are less important than making each metric explicit and
logged.

### Phase 3: Agent-D Sequential Deployment

Purpose: replace full-matrix deployment sampling with sequential placement.

Action design:

- one service-stage placement action at a time
- feasibility mask over nodes
- optional replica add/remove action
- partial deployment included in observation
- repair only as a final safeguard, not the normal path

Training:

- episode length must cover at least two deployment intervals
- recommended minimum: `seconds >= 28800`
- deployment interval remains `14400`
- Agent-S can be loaded from the stable Phase 1 checkpoint

### Phase 4: Joint Training

Purpose: train both agents without destabilizing the scheduler.

Rules:

- start from stable Agent-S checkpoint
- train Agent-D first with Agent-S deterministic or low-learning-rate
- then allow small Agent-S updates
- validate with both fixed deployment and learned deployment

## Completion Criteria

The project should not be called converged until current evidence shows:

- same-seed validation beats heuristic on average delay
- same-seed validation beats heuristic on P95 or SLA metric
- final/restored policy is not worse than best checkpoint by more than a small tolerance
- no invalid scheduling explosion
- Agent-D learned deployment improves over greedy initial deployment over at least two slow cycles
- all claims are reproducible from logged commands and saved checkpoints

## Next Code Steps

1. Extract metric collection from `EdgeComputingEnv`.
2. Add per-task delay distribution output from KKT/environment.
3. Add SLA/tail-latency metrics and convergence report fields.
4. Replace Agent-D full-matrix action with sequential deployment prototype.
5. Add a Phase 1/Phase 3 experiment runner with explicit commands.
