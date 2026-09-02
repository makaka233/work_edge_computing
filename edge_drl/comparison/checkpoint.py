from __future__ import annotations

from dataclasses import fields
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from train_dual_ppo import load_checkpoint


CRITICAL_ENV_KEYS = (
    "num_users",
    "num_edge_nodes",
    "num_service_types",
    "episode_minutes",
    "deployment_interval_minutes",
    "arrival_profile",
    "active_user_ratio",
    "active_user_request_rate_per_minute",
    "traffic_scale",
    "task_compute_scale",
    "task_data_scale",
    "node_compute_capacity_scale",
    "wired_link_bandwidth_scale",
    "topology_k_nearest",
    "deadline_scale",
    "service_resource_fraction",
    "load_ewma_tau_minutes",
    "wireless_uplink_mbps",
    "radio_rtt_ms",
)


def load_checkpoint_configuration(checkpoint_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    internal = dict(checkpoint.get("metadata", {}))
    embedded_args = internal.get("args")
    if isinstance(embedded_args, dict) and embedded_args:
        return path, dict(embedded_args), internal

    sidecar_path = path.parent.parent / "metadata.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            "checkpoint has no embedded args and requires sibling run metadata: "
            f"{sidecar_path}"
        )
    with sidecar_path.open("r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    args = dict(sidecar.get("args", {}))
    if not args:
        raise ValueError(f"checkpoint sidecar has no args: {sidecar_path}")
    return path, args, internal


def edge_config_from_checkpoint(
    checkpoint_args: dict[str, Any],
    *,
    episode_minutes: int,
    environment_seed: int,
    demand_seed: int,
    demand_load_multiplier: float,
) -> EdgeEnvConfig:
    accepted = {field.name for field in fields(EdgeEnvConfig)}
    kwargs = {key: value for key, value in checkpoint_args.items() if key in accepted and value is not None}
    kwargs.update(
        {
            "seed": int(environment_seed),
            "physical_seed": int(
                checkpoint_args.get("seed", 2026)
                if checkpoint_args.get("physical_seed") is None
                else checkpoint_args["physical_seed"]
            ),
            "scenario_seed": int(demand_seed),
            "episode_minutes": int(episode_minutes),
            "episode_hours": None,
            "demand_load_multiplier": float(demand_load_multiplier),
            "request_aggregation_window_seconds": 1.0,
        }
    )
    return EdgeEnvConfig(**kwargs)


def assert_no_critical_conflicts(
    checkpoint_args: dict[str, Any], overrides: dict[str, Any]
) -> None:
    conflicts = []
    for key in CRITICAL_ENV_KEYS:
        if key in overrides and overrides[key] is not None and checkpoint_args.get(key) != overrides[key]:
            conflicts.append(f"{key}: checkpoint={checkpoint_args.get(key)!r}, override={overrides[key]!r}")
    if conflicts:
        raise ValueError("critical checkpoint configuration conflict: " + "; ".join(conflicts))


def build_proposed_agent(
    env: EdgeComputingEnv,
    checkpoint_path: str | Path,
    checkpoint_args: dict[str, Any],
    *,
    device: str,
) -> tuple[HierarchicalPPOAgent, dict[str, Any]]:
    signature = inspect.signature(HierarchicalPPOAgent.from_env)
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if name in {"env", "cls"}:
            continue
        if name in checkpoint_args and checkpoint_args[name] is not None:
            kwargs[name] = checkpoint_args[name]
    kwargs["device"] = device
    replicas = int(checkpoint_args.get("replicas_per_stage", 5))
    kwargs["replicas_per_stage"] = env.config.num_edge_nodes if replicas <= 0 else replicas
    agent = HierarchicalPPOAgent.from_env(env, **kwargs)
    metadata = load_checkpoint(agent, Path(checkpoint_path))
    agent.slow_agent.count_ppo.policy.eval()
    agent.slow_agent.placement_ppo.policy.eval()
    agent.slow_agent.window_critic.eval()
    agent.fast_agent.ppo.policy.eval()
    return agent, metadata
