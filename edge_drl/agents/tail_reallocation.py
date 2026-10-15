"""Learned finite reallocation after the first pass of integer stage quotas."""
from __future__ import annotations

import numpy as np


def record_stage(slow, state, mask, action, logprob, value, window_id, record):
    if not record:
        return
    slow.stage_ppo.buffer.states.append(state)
    slow.stage_ppo.buffer.masks.append(mask.copy())
    slow.stage_ppo.buffer.actions.append(action)
    slow.stage_ppo.buffer.logprobs.append(logprob)
    slow.stage_ppo.buffer.values.append(value)
    slow.pending_stage_indices.append(len(slow.stage_ppo.buffer.actions) - 1)
    key = (-1, -1) if action == len(mask)-1 else divmod(action, slow.max_service_stages)
    slow.pending_stage_keys.append(key)
    slow.pending_stage_node_ids.append(-1)
    slow.stage_window_ids.append(window_id)


def reallocate(slow, env, deployment, memory, storage, demand, window_id,
               deterministic, record, allocate):
    """Each non-STOP action consumes a slot or closes a distinct stage.

    No stage ranking, candidate pruning, proxy reward or forced resource fill.
    The bound is a proof obligation, not an extra policy planning budget.
    """
    slow._reallocation_phase = True
    closed = slow._reallocation_closed
    valid = sum(len(service.stages) for service in env.scenario.services)
    slots = valid * slow.replicas_per_stage - int(deployment.sum())
    for _ in range(slots + valid + 1):
        slow._quota_state = (0, 0, 0, "stage")
        mask = slow._build_incremental_stage_mask(env, memory, storage, deployment)
        mask[:-1] &= ~closed
        if not mask[:-1].any():
            return
        mask[-1] = True
        state = slow._build_incremental_state(env, demand, memory, storage, deployment)
        action, logprob, value = slow.stage_ppo.act(state, mask, deterministic=deterministic)
        action = int(action)
        if not 0 <= action < len(mask) or not mask[action]:
            raise RuntimeError("Reallocation Stage sampled an illegal action")
        record_stage(slow, state, mask, action, logprob, value, window_id, record)
        if action == len(mask)-1:
            slow.last_stage_budget_diagnostics.append({"phase":"reallocation", "action_kind":"stop",
                "service_id":-1, "stage_id":-1, "extra_replicas":0,
                "remaining_memory_gb":float(memory.sum()), "remaining_storage_gb":float(storage.sum())})
            return
        before = int(deployment.sum())
        quota = allocate(action, "reallocation")
        if quota == 0:
            closed[action] = True
        elif int(deployment.sum()) - before != quota:
            raise RuntimeError("Reallocation did not consume its executed integer quota")
    raise RuntimeError("Reallocation violated its physical finite-termination bound")
