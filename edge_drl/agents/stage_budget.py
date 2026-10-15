"""Learned stage ordering and per-stage continuous replica quotas.

No template, candidate truncation, expert action, or latency proxy reward.
Coverage and RAM/SSD are hard constraints. All remaining stage/node choices
are made by the three heads of the same Slow PPO agent.
"""
from __future__ import annotations

import numpy as np


def plan_stage_budgets(slow, env, deployment, memory, storage, demand,
                      window_id, deterministic, record):
    stages = [(service, stage) for service in env.scenario.services for stage in service.stages]
    by_key = {(s.service_id, t.stage_id): t for s, t in stages}
    # Cover hard-to-pack stages first, but learn every coverage node placement.
    order = sorted(stages, key=lambda item: (
        int(np.count_nonzero((memory >= item[1].memory_gb) & (storage >= item[1].storage_gb))),
        -max(item[1].memory_gb / max(float(memory.max()), 1e-9),
             item[1].storage_gb / max(float(storage.max()), 1e-9)),
        item[0].service_id, item[1].stage_id,
    ))
    for service, stage in order:
        node = slow._place_slow_replica(
            env, deployment=deployment, remaining_memory=memory, remaining_storage=storage,
            demand=demand, service_id=service.service_id, stage_id=stage.stage_id,
            replica_idx=0, window_id=window_id, deterministic=deterministic, record=record)
        if node is None:
            # Never train log probabilities against a heuristic replacement.
            raise RuntimeError("stage-budget-v3 coverage fragmented: no feasible learned placement")

    visited = np.zeros(slow.num_service_types * slow.max_service_stages + 1, dtype=bool)
    slow._stage_budget_visited = visited[:-1]
    for _ in stages:
        mask = slow._build_incremental_stage_mask(env, memory, storage, deployment)
        mask[-1] = False
        mask &= ~visited
        if not mask.any():
            break
        state = slow._build_incremental_state(env, demand, memory, storage, deployment)
        stage_action, lp, value = slow.stage_ppo.act(state, mask, deterministic=deterministic)
        stage_action = int(stage_action)
        visited[stage_action] = True
        service_id, stage_id = divmod(stage_action, slow.max_service_stages)
        key = (service_id, stage_id)
        stage = by_key[key]
        feasible = ((memory >= stage.memory_gb) & (storage >= stage.storage_gb)
                    & ~deployment[service_id, stage_id])
        maximum = min(int(feasible.sum()), slow.replicas_per_stage - int(deployment[key].sum()))
        if maximum <= 0:
            raise RuntimeError("Stage mask admitted an infeasible quota")
        if record:
            slow.stage_ppo.buffer.states.append(state)
            slow.stage_ppo.buffer.masks.append(mask.copy())
            slow.stage_ppo.buffer.actions.append(stage_action)
            slow.stage_ppo.buffer.logprobs.append(lp)
            slow.stage_ppo.buffer.values.append(value)
            slow.pending_stage_indices.append(len(slow.stage_ppo.buffer.actions)-1)
            slow.pending_stage_keys.append(key)
            # Stage credit is window-team credit, not a particular node ablation.
            slow.pending_stage_node_ids.append(-1)
            slow.stage_window_ids.append(window_id)

        # Count observes the stage chosen BEFORE its own action. The current
        # stage's compute/data/placement context is explicit in this state.
        count_state = slow._build_state(env, demand, memory, storage, service_id, stage_id,
                                       int(deployment[key].sum()), planned_deployment=deployment)
        count_mask = np.ones(1, dtype=bool)
        fraction, lp, value = slow.count_ppo.act(count_state, count_mask, deterministic=deterministic)
        fraction = float(fraction)
        # A continuous action is sampled and stored with its exact Beta density;
        # the environment deterministically converts it to an integer quota.
        # This is not a fixed list of deployment/budget candidates.
        quota = min(int(np.floor(fraction * (maximum+1))), maximum)
        slow.last_stage_budget_diagnostics.append({
            "service_id":service_id, "stage_id":stage_id, "order":int(visited.sum()),
            "fraction":fraction, "feasible_extra_replicas":maximum, "extra_replicas":quota,
            "remaining_memory_gb":float(memory.sum()), "remaining_storage_gb":float(storage.sum())})
        slow.pending_budget_fraction += fraction / max(len(stages), 1)
        slow.pending_budget_requested_replicas += quota
        slow.pending_budget_available_memory_gb += maximum * stage.memory_gb
        slow.pending_budget_available_storage_gb += maximum * stage.storage_gb
        slow.pending_budget_target_memory_gb += quota * stage.memory_gb
        slow.pending_budget_target_storage_gb += quota * stage.storage_gb
        if record:
            slow.count_ppo.buffer.states.append(count_state)
            slow.count_ppo.buffer.masks.append(count_mask)
            slow.count_ppo.buffer.actions.append(fraction)
            slow.count_ppo.buffer.logprobs.append(lp)
            slow.count_ppo.buffer.values.append(value)
            slow.pending_count_indices.append(len(slow.count_ppo.buffer.actions)-1)
            slow.pending_count_stage_keys.append(key)
            slow.pending_count_node_ids.append(None)
            slow.count_window_ids.append(window_id)
        for _ in range(quota):
            node = slow._place_slow_replica(
                env, deployment=deployment, remaining_memory=memory, remaining_storage=storage,
                demand=demand, service_id=service_id, stage_id=stage_id,
                replica_idx=int(deployment[key].sum()), window_id=window_id,
                deterministic=deterministic, record=record)
            if node is None:
                raise RuntimeError("A physically feasible stage quota was not executable")
            slow.pending_budget_used_memory_gb += stage.memory_gb
            slow.pending_budget_used_storage_gb += stage.storage_gb
    valid, reason = env.check_deployment_feasible(deployment)
    if not valid:
        raise RuntimeError(f"stage-budget-v3 deployment invalid: {reason}")
    return deployment
