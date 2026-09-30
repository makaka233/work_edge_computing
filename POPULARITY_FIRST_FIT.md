# FitDeploy

This deployment-only baseline uses the frozen Proposed Fast PPO and the same
public KKT settlement. It requires no additional training. The original
PopularityDeploy remains available and unchanged.

At each deployment window, estimate per-service demand from the same causal
demand information available to the other planners. First place one replica
of every valid stage, ordered by descending service demand, then service ID
and stage ID. Place each replica on the first feasible node in ascending node
ID order, checking both service memory and storage capacities.

If First-Fit fragments capacity and cannot cover all stages, discard the partial
placement and solve a binary feasibility problem with one replica per stage
and a zero objective. This repair uses no latency or performance ranking and
is reported as `coverage_repaired` in the planning diagnostics. If no feasible
solution is found within the solver limit, report an error rather than skip a
stage or evaluate an invalid deployment.

Then add replicas greedily. Choose the feasible stage with greatest service
demand divided by its current replica count, breaking ties by service/stage
order, and place on its first feasible node. Stop when no positive-demand
stage can accept another replica. There is no arbitrary replica cap.

The placement score does not use node compute capacity, node load, bandwidth,
propagation delay or KKT marginal latency. These remain available to the frozen
Fast policy and the common settlement. Fixed node ordering is a stated packing
rule, not an ordering selected from evaluation results.

Select with `--schemes Proposed Monolithic HEFTFast FitDeploy`
in `run_comparison.py`, retaining the same checkpoints, seeds, scenarios and
sampling configuration. Use a new run ID to preserve historical results.
No relative performance ranking or minimum gap is guaranteed.
