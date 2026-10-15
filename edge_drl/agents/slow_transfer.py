"""Explicit actor-only initialization for the integer Slow observation ablation."""
from __future__ import annotations

import hashlib

import torch


def copy_slow_actor(source, destination, *, scope="all"):
    """Copy selected actor parameters; new observation columns start at 0.

    No optimizer, critic, replay or selection history is transferred. This is
    a fresh experiment, never full continuation of the source training run.
    Beta-to-integer transfer preserves the parameter network, while changing
    the probability law from a continuous density to normalized bin masses.
    """
    if scope not in {"all", "stage-placement"}:
        raise ValueError("unknown Slow actor initialization scope")
    modes = {"stage-budget-v3", "stage-budget-v4", "stage-budget-v5", "stage-budget-v6"}
    if destination.allocation_mode not in modes or source.allocation_mode not in modes:
        raise ValueError("Slow actor initialization requires stage quotas")
    compatible_count = (source.allocation_mode == "stage-budget-v3" and destination.allocation_mode in {"stage-budget-v5", "stage-budget-v6"}) or (
        source.allocation_mode == "stage-budget-v5" and destination.allocation_mode == "stage-budget-v6")
    if scope == "all" and source.allocation_mode != destination.allocation_mode and not compatible_count:
        raise ValueError("Count architecture differs; select stage-placement initialization")
    if (source.num_nodes, source.num_service_types, source.max_service_stages,
            source.replicas_per_stage, source.cost_context) != (
            destination.num_nodes, destination.num_service_types, destination.max_service_stages,
            destination.replicas_per_stage, destination.cost_context):
        raise ValueError("Slow actor initialization dimensions/physical replica limit differ")
    if source.chain_context and not destination.chain_context:
        raise ValueError("Slow actor initialization cannot discard learned chain observations")
    if source.tail_reallocation and not destination.tail_reallocation:
        raise ValueError("Slow actor initialization cannot discard reallocation state")
    head_names = ("count_ppo", "stage_ppo", "placement_ppo") if scope == "all" else ("stage_ppo", "placement_ppo")
    if destination.count_ppo.optimizer.state:
        raise ValueError("Slow actor initialization requires fresh optimizers")
    prepared = []
    for head_name in head_names:
        original, target = getattr(source, head_name), getattr(destination, head_name)
        if target.optimizer.state:
            raise ValueError("Slow actor initialization requires fresh optimizers")
        old = original.policy.state_dict()
        updated = target.policy.state_dict()
        for key, value in old.items():
            if key.startswith("critic."):
                continue
            if key not in updated:
                raise ValueError(f"Slow actor initialization missing parameter {key}")
            if updated[key].shape == value.shape:
                updated[key] = value.detach().clone()
            elif (key == "global_encoder.0.weight" and updated[key].shape[0] == value.shape[0]
                  and updated[key].shape[1] > value.shape[1]
                  and updated[key].shape[1] - value.shape[1] == destination._global_dim() - source._global_dim()):
                updated[key] = torch.zeros_like(updated[key])
                updated[key][:, :value.shape[1]] = value
            elif (((not source.chain_context and destination.chain_context)
                   or (not source.tail_reallocation and destination.tail_reallocation))
                  and key in {"node_encoder.0.weight", "stage_encoder.0.weight"}
                  and updated[key].shape[0] == value.shape[0]
                  and updated[key].shape[1] > value.shape[1]):
                updated[key] = torch.zeros_like(updated[key])
                updated[key][:, :value.shape[1]] = value
            else:
                raise ValueError(f"Slow actor initialization architecture mismatch: {key}")
        prepared.append((target, updated))
    for target, updated in prepared:
        target.policy.load_state_dict(updated, strict=True)


def initialize_slow_actors(env, agent, checkpoint_path, *, scope="all"):
    from edge_drl.comparison.checkpoint import (
        CRITICAL_ENV_KEYS, assert_no_critical_conflicts, build_proposed_agent, load_checkpoint_configuration,
    )
    from dataclasses import asdict
    path, args, metadata = load_checkpoint_configuration(checkpoint_path)
    current = asdict(env.config)
    assert_no_critical_conflicts(args, {k: current.get(k) for k in CRITICAL_ENV_KEYS})
    physical_seed = args.get("seed", 2026) if args.get("physical_seed") is None else args["physical_seed"]
    if int(physical_seed) != int(env.config.physical_seed):
        raise ValueError("Slow actor initialization physical seed differs")
    # Constructing the source networks must not change the fresh run's RNG.
    with torch.random.fork_rng(devices=[]):
        source, _ = build_proposed_agent(env, path, args, device="cpu")
        copy_slow_actor(source.slow_agent, agent.slow_agent, scope=scope)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_update": metadata.get("completed_updates", metadata.get("update")),
        "source_context": source.slow_agent.deployment_context,
        "target_context": agent.slow_agent.deployment_context,
        "source_allocation": source.slow_agent.allocation_mode,
        "target_allocation": agent.slow_agent.allocation_mode,
        "scope": ("three_slow_actors_only_new_critics_and_optimizers" if scope == "all"
                  else "stage_placement_actors_only_new_count_critics_and_optimizers"),
        "new_observation_columns": "zero_initialized", "resume": False}
