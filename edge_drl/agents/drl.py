from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from edge_drl.env.environment import EdgeComputingEnv
from edge_drl.env.scenario import TaskRequest
from edge_drl.models.ppo import PPOAgent


SLOW_GLOBAL_DIM = 17
SLOW_NODE_BASE_FEATURE_DIM = 6
SLOW_EDGE_FEATURE_DIM = 4
SLOW_STAGE_FEATURE_DIM = 10


def _discounted_window_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Discount ten-minute Slow rewards without crossing episode boundaries."""

    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=bool)
    if rewards_np.ndim != 1 or dones_np.shape != rewards_np.shape:
        raise ValueError("Slow rewards and dones must have the same 1D shape")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Slow window gamma must be in [0, 1]")
    discounted = np.zeros_like(rewards_np)
    running_return = 0.0
    for index in range(len(rewards_np) - 1, -1, -1):
        if dones_np[index]:
            running_return = 0.0
        running_return = float(rewards_np[index]) + gamma * running_return
        discounted[index] = running_return
    return discounted


def slow_node_feature_dim(num_service_types: int) -> int:
    return SLOW_NODE_BASE_FEATURE_DIM + num_service_types


def slow_obs_dim(num_nodes: int, num_service_types: int) -> int:
    return (
        SLOW_GLOBAL_DIM
        + num_nodes * slow_node_feature_dim(num_service_types)
        + num_nodes * num_nodes * SLOW_EDGE_FEATURE_DIM
    )


def slow_incremental_obs_dim(
    num_nodes: int,
    num_service_types: int,
    max_service_stages: int,
) -> int:
    return slow_obs_dim(num_nodes, num_service_types) + (
        num_service_types * max_service_stages * SLOW_STAGE_FEATURE_DIM
    )


def fast_feature_dims(chain_context: str = "legacy") -> tuple[int, int]:
    if chain_context == "legacy":
        return FAST_GLOBAL_DIM, FAST_NODE_FEATURE_DIM
    if chain_context == "chain-v2":
        return FAST_CHAIN_GLOBAL_DIM, FAST_CHAIN_NODE_FEATURE_DIM
    raise ValueError("fast chain context must be 'legacy' or 'chain-v2'")


def fast_obs_dim(
    num_nodes: int,
    policy_kind: str = "gat_node_scorer",
    chain_context: str = "legacy",
) -> int:
    global_dim, node_feature_dim = fast_feature_dims(chain_context)
    dim = global_dim + num_nodes * node_feature_dim
    if policy_kind == "gat_node_scorer":
        dim += num_nodes * num_nodes * FAST_EDGE_FEATURE_DIM
    return dim


FAST_GLOBAL_DIM = 12
FAST_NODE_FEATURE_DIM = 9
FAST_EDGE_FEATURE_DIM = 3
FAST_CHAIN_GLOBAL_DIM = 16
FAST_CHAIN_NODE_FEATURE_DIM = 13
FAST_PATH_GLOBAL_DIM = 9
FAST_PATH_STAGE_FEATURE_DIM = 5
FAST_PATH_SUMMARY_DIM = 6
FAST_STRUCT_NODE_FEATURE_DIM = 9
FAST_STRUCT_EDGE_FEATURE_DIM = 5
FAST_KKT_GLOBAL_DIM = 18
FAST_KKT_NODE_FEATURE_DIM = 11
FAST_KKT_EDGE_FEATURE_DIM = 9


def fast_path_feature_dim(max_service_stages: int) -> int:
    return FAST_PATH_SUMMARY_DIM + max_service_stages * FAST_PATH_STAGE_FEATURE_DIM


def fast_path_obs_dim(max_service_stages: int, candidate_limit: int) -> int:
    return FAST_PATH_GLOBAL_DIM + candidate_limit * fast_path_feature_dim(max_service_stages)


def fast_structured_obs_dim(num_nodes: int, max_service_stages: int) -> int:
    return (
        FAST_PATH_GLOBAL_DIM
        + max_service_stages * num_nodes * FAST_STRUCT_NODE_FEATURE_DIM
        + (max_service_stages - 1) * num_nodes * num_nodes * FAST_STRUCT_EDGE_FEATURE_DIM
    )


def fast_kkt_autoregressive_obs_dim(num_nodes: int, max_service_stages: int) -> int:
    return (
        FAST_KKT_GLOBAL_DIM
        + max_service_stages * num_nodes * FAST_KKT_NODE_FEATURE_DIM
        + (max_service_stages - 1) * num_nodes * num_nodes * FAST_KKT_EDGE_FEATURE_DIM
    )


class SlowWindowCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


@dataclass
class SlowDeploymentPPOAgent:
    num_nodes: int
    num_service_types: int
    max_service_stages: int
    replicas_per_stage: int = 5
    coverage_repair: bool = True
    lr: float = 3e-4
    count_lr: float = 2e-4
    placement_lr: float | None = None
    k_epochs: int = 3
    entropy_coef: float = 0.001
    count_entropy_coef: float | None = None
    placement_entropy_coef: float | None = 0.005
    placement_entropy_final_coef: float | None = 0.0035
    placement_entropy_hold_updates: int = 64
    placement_entropy_decay_updates: int = 64
    placement_entropy_target: float | None = 1.8
    placement_entropy_max_coef: float = 0.05
    placement_entropy_adaptation_rate: float = 2e-3
    count_global_advantage_coef: float = 0.25
    placement_global_advantage_coef: float = 0.35
    placement_global_attribution_coef: float = 0.50
    global_advantage_ev_full: float = 0.20
    value_coef: float = 0.5
    count_value_coef: float = 0.0
    critic_lr: float | None = 5e-4
    critic_k_epochs: int = 8
    critic_replay_windows: int = 96
    critic_replay_decay: float = 0.90
    critic_holdout_windows: int = 12
    critic_holdout_episodes: int = 2
    critic_gradient_clip: float = 5.0
    window_gamma: float = 0.95
    target_kl: float | None = 0.03
    count_target_kl: float | None = 0.015
    placement_target_kl: float | None = None
    minibatch_size: int = 2048
    tail_latency_coef: float = 0.35
    deterministic_count_mode: str = "expected"
    allocation_mode: str = "legacy-factorized"
    counterfactual_credit_coef: float = 0.0
    device: str = "cpu"
    count_ppo: PPOAgent = field(init=False)
    placement_ppo: PPOAgent = field(init=False)
    ppo: PPOAgent = field(init=False)
    window_critic: SlowWindowCritic = field(init=False)
    critic_optimizer: torch.optim.Optimizer = field(init=False)
    pending_count_indices: list[int] = field(default_factory=list)
    pending_placement_indices: list[int] = field(default_factory=list)
    pending_count_stage_keys: list[tuple[int, int]] = field(default_factory=list)
    pending_count_node_ids: list[int | None] = field(default_factory=list)
    pending_placement_stage_keys: list[tuple[int, int]] = field(default_factory=list)
    count_action_returns: list[float] = field(default_factory=list)
    placement_action_returns: list[float] = field(default_factory=list)
    count_action_stage_keys: list[tuple[int, int]] = field(default_factory=list)
    placement_action_stage_keys: list[tuple[int, int]] = field(default_factory=list)
    count_window_ids: list[int] = field(default_factory=list)
    placement_window_ids: list[int] = field(default_factory=list)
    window_states: list[np.ndarray] = field(default_factory=list)
    window_old_values: list[float] = field(default_factory=list)
    window_returns: list[float] = field(default_factory=list)
    window_dones: list[bool] = field(default_factory=list)
    window_episode_ids: list[int] = field(default_factory=list)
    pending_window_id: int | None = None
    last_window_feedback: dict[str, float] = field(default_factory=dict)
    placement_updates_completed: int = field(default=0, init=False)
    placement_entropy_current_coef: float = field(default=0.0, init=False)
    last_window_critic_holdout_ev: float = field(default=0.0, init=False)
    critic_return_mean: float = field(default=0.0, init=False)
    critic_return_std: float = field(default=1.0, init=False)
    critic_update_index: int = field(default=0, init=False)
    critic_replay_states: list[np.ndarray] = field(default_factory=list)
    critic_replay_returns: list[float] = field(default_factory=list)
    critic_replay_update_ids: list[int] = field(default_factory=list)
    critic_replay_episode_ids: list[int] = field(default_factory=list)
    critic_episode_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.tail_latency_coef <= 1.0:
            raise ValueError("tail_latency_coef must be in [0, 1]")
        if self.deterministic_count_mode not in {"expected", "mode"}:
            raise ValueError("deterministic_count_mode must be 'expected' or 'mode'")
        if self.allocation_mode not in {"legacy-factorized", "global-incremental"}:
            raise ValueError(
                "allocation_mode must be 'legacy-factorized' or 'global-incremental'"
            )
        if self.counterfactual_credit_coef < 0.0:
            raise ValueError("counterfactual_credit_coef must be non-negative")
        placement_entropy_start = self._placement_entropy_start()
        placement_entropy_final = self._placement_entropy_final()
        if placement_entropy_start < 0.0 or placement_entropy_final < 0.0:
            raise ValueError("placement entropy coefficients must be non-negative")
        if placement_entropy_final > placement_entropy_start:
            raise ValueError("placement_entropy_final_coef must not exceed the initial coefficient")
        if self.placement_entropy_hold_updates < 0:
            raise ValueError("placement_entropy_hold_updates must be >= 0")
        if self.placement_entropy_decay_updates < 1:
            raise ValueError("placement_entropy_decay_updates must be >= 1")
        if self.placement_entropy_target is not None and self.placement_entropy_target < 0.0:
            raise ValueError("placement_entropy_target must be non-negative")
        if self.placement_entropy_max_coef < placement_entropy_start:
            raise ValueError("placement_entropy_max_coef must be at least the initial coefficient")
        if self.placement_entropy_adaptation_rate < 0.0:
            raise ValueError("placement_entropy_adaptation_rate must be non-negative")
        if self.count_global_advantage_coef < 0.0 or self.placement_global_advantage_coef < 0.0:
            raise ValueError("Slow global advantage coefficients must be non-negative")
        if self.placement_global_attribution_coef < 0.0:
            raise ValueError("placement_global_attribution_coef must be non-negative")
        if self.global_advantage_ev_full <= 0.0:
            raise ValueError("global_advantage_ev_full must be positive")
        if not 0.0 <= self.window_gamma <= 1.0:
            raise ValueError("window_gamma must be in [0, 1]")
        if self.critic_replay_windows < 1:
            raise ValueError("critic_replay_windows must be positive")
        if self.critic_holdout_windows < 1:
            raise ValueError("critic_holdout_windows must be positive")
        if self.critic_holdout_episodes < 1:
            raise ValueError("critic_holdout_episodes must be positive")
        if self.critic_replay_windows <= self.critic_holdout_windows:
            raise ValueError("critic_replay_windows must exceed critic_holdout_windows")
        if not 0.0 < self.critic_replay_decay <= 1.0:
            raise ValueError("critic_replay_decay must be in (0, 1]")
        if self.critic_gradient_clip <= 0.0:
            raise ValueError("critic_gradient_clip must be positive")
        self.placement_entropy_current_coef = placement_entropy_start
        incremental_allocation = self.allocation_mode == "global-incremental"
        self.count_ppo = PPOAgent(
            obs_dim=(
                slow_incremental_obs_dim(
                    self.num_nodes,
                    self.num_service_types,
                    self.max_service_stages,
                )
                if incremental_allocation
                else slow_obs_dim(self.num_nodes, self.num_service_types)
            ),
            action_dim=(
                self.num_service_types * self.max_service_stages + 1
                if incremental_allocation
                else self.replicas_per_stage
            ),
            hidden_dim=128,
            lr=self.count_lr,
            gamma=0.99,
            k_epochs=self.k_epochs,
            entropy_coef=self.entropy_coef if self.count_entropy_coef is None else self.count_entropy_coef,
            # Count uses direct stage-centered Monte-Carlo returns.  Its
            # consistently failed value head must not dominate the shared
            # graph encoder with critic gradients.
            value_coef=self.count_value_coef,
            target_kl=self.count_target_kl,
            minibatch_size=self.minibatch_size,
            policy_kind="slow_gat_stage" if incremental_allocation else "slow_gat_count",
            global_dim=SLOW_GLOBAL_DIM,
            node_feature_dim=slow_node_feature_dim(self.num_service_types),
            edge_feature_dim=SLOW_EDGE_FEATURE_DIM,
            stage_feature_dim=SLOW_STAGE_FEATURE_DIM if incremental_allocation else None,
            num_nodes=self.num_nodes,
            detach_critic_backbone=True,
            device=self.device,
        )
        self.placement_ppo = PPOAgent(
            obs_dim=slow_obs_dim(self.num_nodes, self.num_service_types),
            action_dim=self.num_nodes,
            hidden_dim=128,
            lr=self.lr if self.placement_lr is None else self.placement_lr,
            gamma=0.99,
            k_epochs=self.k_epochs,
            entropy_coef=self.placement_entropy_coefficient(),
            value_coef=self.value_coef,
            target_kl=self.target_kl if self.placement_target_kl is None else self.placement_target_kl,
            minibatch_size=self.minibatch_size,
            policy_kind="slow_gat_node",
            global_dim=SLOW_GLOBAL_DIM,
            node_feature_dim=slow_node_feature_dim(self.num_service_types),
            edge_feature_dim=SLOW_EDGE_FEATURE_DIM,
            num_nodes=self.num_nodes,
            detach_critic_backbone=True,
            device=self.device,
        )
        self.ppo = self.placement_ppo
        critic_device = torch.device(self.device)
        self.window_critic = SlowWindowCritic(
            slow_obs_dim(self.num_nodes, self.num_service_types),
            hidden_dim=128,
        ).to(critic_device)
        self.critic_optimizer = torch.optim.Adam(
            self.window_critic.parameters(),
            lr=self.lr if self.critic_lr is None else self.critic_lr,
        )

    def _placement_entropy_start(self) -> float:
        return float(self.entropy_coef if self.placement_entropy_coef is None else self.placement_entropy_coef)

    def _placement_entropy_final(self) -> float:
        start = self._placement_entropy_start()
        return float(start if self.placement_entropy_final_coef is None else self.placement_entropy_final_coef)

    def placement_entropy_schedule_coefficient(self) -> float:
        progress = min(
            max(float(self.placement_updates_completed - self.placement_entropy_hold_updates), 0.0)
            / float(self.placement_entropy_decay_updates),
            1.0,
        )
        start = self._placement_entropy_start()
        return start + progress * (self._placement_entropy_final() - start)

    def placement_entropy_coefficient(self) -> float:
        return max(
            float(self.placement_entropy_current_coef),
            self.placement_entropy_schedule_coefficient(),
        )

    def _adapt_placement_entropy_coefficient(self, observed_entropy: float) -> float:
        self.placement_updates_completed += 1
        schedule_floor = self.placement_entropy_schedule_coefficient()
        if self.placement_entropy_target is None or not np.isfinite(observed_entropy):
            next_coef = schedule_floor
        else:
            next_coef = self.placement_entropy_current_coef + self.placement_entropy_adaptation_rate * (
                float(self.placement_entropy_target) - float(observed_entropy)
            )
            next_coef = float(np.clip(next_coef, schedule_floor, self.placement_entropy_max_coef))
        self.placement_entropy_current_coef = float(next_coef)
        self.placement_ppo.entropy_coef = self.placement_entropy_current_coef
        return self.placement_entropy_current_coef

    def global_advantage_reliability(self) -> float:
        """Scale global credit by predictive performance on unseen recent windows."""

        return float(
            np.clip(
                max(self.last_window_critic_holdout_ev, 0.0)
                / self.global_advantage_ev_full,
                0.0,
                1.0,
            )
        )

    def plan_deployment(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> np.ndarray:
        env._require_ready()
        assert env.scenario is not None
        current = env.deployment if env.deployment is not None else np.zeros(
            (self.num_service_types, self.max_service_stages, self.num_nodes), dtype=bool
        )
        deployment = np.zeros_like(current, dtype=bool)
        remaining_memory = env.service_memory_capacities()
        remaining_storage = env.service_storage_capacities()
        demand = self._node_service_demand(env)
        self.pending_count_indices.clear()
        self.pending_placement_indices.clear()
        self.pending_count_stage_keys.clear()
        self.pending_count_node_ids.clear()
        self.pending_placement_stage_keys.clear()
        window_id: int | None = None
        if record:
            if self.pending_window_id is not None:
                raise RuntimeError("previous slow deployment window has no completed return")
            window_state = self._build_window_state(env, demand, remaining_memory, remaining_storage)
            state_t = torch.as_tensor(window_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                normalized_value = float(self.window_critic(state_t).item())
                old_value = normalized_value * self.critic_return_std + self.critic_return_mean
            window_id = len(self.window_states)
            self.window_states.append(window_state)
            self.window_old_values.append(old_value)
            self.pending_window_id = window_id

        if self.allocation_mode == "global-incremental":
            return self._plan_deployment_incremental(
                env,
                deployment=deployment,
                remaining_memory=remaining_memory,
                remaining_storage=remaining_storage,
                demand=demand,
                window_id=window_id,
                deterministic=deterministic,
                record=record,
            )

        stage_sequence = [
            (service, stage)
            for service in env.scenario.services
            for stage in service.stages
        ]
        for stage_index, (service, stage) in enumerate(stage_sequence):
            future_stages = [future_stage for _, future_stage in stage_sequence[stage_index + 1 :]]
            reserved_memory = float(sum(future_stage.memory_gb for future_stage in future_stages))
            reserved_storage = float(sum(future_stage.storage_gb for future_stage in future_stages))
            count_state = self._build_state(
                env,
                demand,
                remaining_memory,
                remaining_storage,
                service.service_id,
                stage.stage_id,
                0,
                planned_deployment=deployment,
            )
            count_mask = self._build_count_mask(
                stage,
                remaining_memory,
                remaining_storage,
                reserved_memory=reserved_memory,
                reserved_storage=reserved_storage,
            )
            if not count_mask.any():
                continue
            if deterministic and self.deterministic_count_mode == "expected":
                # Replica count is ordinal.  The categorical mode is unstable while
                # the policy is still broad: a tiny logit difference can turn a
                # nearly uniform policy into an arbitrarily small deployment.  Use
                # the conservative (ceiling) posterior mean for deterministic
                # collection/evaluation, while PPO training still samples exactly
                # from the categorical policy.
                count_probs, count_value = self.count_ppo.action_probabilities(
                    count_state,
                    count_mask,
                )
                expected_replicas = float(
                    np.dot(count_probs, np.arange(1, self.replicas_per_stage + 1, dtype=np.float32))
                )
                target_replicas = int(np.ceil(expected_replicas - 1e-8))
                target_replicas = int(np.clip(target_replicas, 1, int(np.count_nonzero(count_mask))))
                count_action = target_replicas - 1
                count_logprob = float(np.log(max(float(count_probs[count_action]), 1e-12)))
            else:
                count_action, count_logprob, count_value = self.count_ppo.act(
                    count_state,
                    count_mask,
                    deterministic=deterministic,
                )
            if not count_mask[count_action]:
                count_action = int(np.where(count_mask)[0][0])
            target_replicas = count_action + 1
            if record:
                self.count_ppo.buffer.states.append(count_state)
                self.count_ppo.buffer.masks.append(count_mask.astype(bool))
                self.count_ppo.buffer.actions.append(count_action)
                self.count_ppo.buffer.logprobs.append(count_logprob)
                self.count_ppo.buffer.values.append(count_value)
                self.pending_count_indices.append(len(self.count_ppo.buffer.actions) - 1)
                self.pending_count_stage_keys.append((service.service_id, stage.stage_id))
                self.pending_count_node_ids.append(None)
                assert window_id is not None
                self.count_window_ids.append(window_id)

            for replica_idx in range(target_replicas):
                state = self._build_state(
                    env,
                    demand,
                    remaining_memory,
                    remaining_storage,
                    service.service_id,
                    stage.stage_id,
                    replica_idx,
                    planned_deployment=deployment,
                )
                already = deployment[service.service_id, stage.stage_id]
                feasible_new = (
                    (remaining_memory >= stage.memory_gb)
                    & (remaining_storage >= stage.storage_gb)
                    & ~already
                )
                mask = feasible_new
                if not mask.any():
                    break

                action, logprob, value = self.placement_ppo.act(state, mask, deterministic=deterministic)
                if not mask[action]:
                    action = int(np.where(mask)[0][0])

                if record:
                    self.placement_ppo.buffer.states.append(state)
                    self.placement_ppo.buffer.masks.append(mask.astype(bool))
                    self.placement_ppo.buffer.actions.append(action)
                    self.placement_ppo.buffer.logprobs.append(logprob)
                    self.placement_ppo.buffer.values.append(value)
                    self.pending_placement_indices.append(len(self.placement_ppo.buffer.actions) - 1)
                    self.pending_placement_stage_keys.append((service.service_id, stage.stage_id))
                    assert window_id is not None
                    self.placement_window_ids.append(window_id)

                deployment[service.service_id, stage.stage_id, action] = True
                remaining_memory[action] -= stage.memory_gb
                remaining_storage[action] -= stage.storage_gb

        if self.coverage_repair:
            self._coverage_repair(env, deployment, remaining_memory, remaining_storage)

        feasible, reason = env.check_deployment_feasible(deployment)
        if not feasible:
            deployment = self._repair_with_current_or_full(env, deployment)
        return deployment

    def _plan_deployment_incremental(
        self,
        env: EdgeComputingEnv,
        *,
        deployment: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        demand: np.ndarray,
        window_id: int | None,
        deterministic: bool,
        record: bool,
    ) -> np.ndarray:
        """Allocate coverage first, then spend the shared budget one replica at a time.

        The legacy policy commits every requested replica for service 0 before
        service 1 is considered.  Under a nearly full memory budget that makes
        service id, rather than marginal value, decide who gets the remaining
        capacity.  This mode first establishes one feasible replica per stage;
        a topology-aware stage actor then chooses the owner of every additional
        replica and can stop before filling the budget.
        """

        assert env.scenario is not None
        stages = [
            (service, stage)
            for service in env.scenario.services
            for stage in service.stages
        ]
        memory_capacity = np.maximum(env.service_memory_capacities(), 1e-9)
        storage_capacity = np.maximum(env.service_storage_capacities(), 1e-9)

        def coverage_priority(item) -> tuple[float, float, int, int]:
            service, stage = item
            feasible = np.count_nonzero(
                (remaining_memory >= stage.memory_gb)
                & (remaining_storage >= stage.storage_gb)
            )
            resource_pressure = max(
                float(np.min(stage.memory_gb / memory_capacity)),
                float(np.min(stage.storage_gb / storage_capacity)),
            )
            # Most constrained stages go first.  IDs are only deterministic
            # tie breakers and no longer control allocation of extra replicas.
            return (
                float(feasible),
                -resource_pressure,
                int(service.service_id),
                int(stage.stage_id),
            )

        for service, stage in sorted(stages, key=coverage_priority):
            placed = self._place_slow_replica(
                env,
                deployment=deployment,
                remaining_memory=remaining_memory,
                remaining_storage=remaining_storage,
                demand=demand,
                service_id=int(service.service_id),
                stage_id=int(stage.stage_id),
                replica_idx=0,
                window_id=window_id,
                deterministic=deterministic,
                record=record,
            )
            if placed is None:
                break

        stop_action = self.num_service_types * self.max_service_stages
        maximum_expansions = self.num_nodes * len(stages)
        for _ in range(maximum_expansions):
            count_state = self._build_incremental_state(
                env,
                demand,
                remaining_memory,
                remaining_storage,
                deployment,
            )
            count_mask = self._build_incremental_stage_mask(
                env,
                remaining_memory,
                remaining_storage,
                deployment,
            )
            if not count_mask[:-1].any():
                break
            count_action, count_logprob, count_value = self.count_ppo.act(
                count_state,
                count_mask,
                deterministic=deterministic,
            )
            if not count_mask[count_action]:
                count_action = stop_action

            stage_key = (
                (-1, -1)
                if count_action == stop_action
                else divmod(int(count_action), self.max_service_stages)
            )
            count_buffer_index: int | None = None
            if record:
                self.count_ppo.buffer.states.append(count_state)
                self.count_ppo.buffer.masks.append(count_mask.astype(bool))
                self.count_ppo.buffer.actions.append(int(count_action))
                self.count_ppo.buffer.logprobs.append(float(count_logprob))
                self.count_ppo.buffer.values.append(float(count_value))
                count_buffer_index = len(self.count_ppo.buffer.actions) - 1
                self.pending_count_indices.append(count_buffer_index)
                self.pending_count_stage_keys.append(stage_key)
                self.pending_count_node_ids.append(None)
                assert window_id is not None
                self.count_window_ids.append(window_id)

            if count_action == stop_action:
                break

            service_id, stage_id = stage_key
            replica_idx = int(deployment[service_id, stage_id].sum())
            node_id = self._place_slow_replica(
                env,
                deployment=deployment,
                remaining_memory=remaining_memory,
                remaining_storage=remaining_storage,
                demand=demand,
                service_id=service_id,
                stage_id=stage_id,
                replica_idx=replica_idx,
                window_id=window_id,
                deterministic=deterministic,
                record=record,
            )
            if node_id is None:
                break
            if record:
                assert count_buffer_index is not None
                self.pending_count_node_ids[-1] = int(node_id)

        if self.coverage_repair:
            self._coverage_repair(env, deployment, remaining_memory, remaining_storage)
        feasible, _ = env.check_deployment_feasible(deployment)
        if not feasible:
            deployment = self._repair_with_current_or_full(env, deployment)
        return deployment

    def _place_slow_replica(
        self,
        env: EdgeComputingEnv,
        *,
        deployment: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        demand: np.ndarray,
        service_id: int,
        stage_id: int,
        replica_idx: int,
        window_id: int | None,
        deterministic: bool,
        record: bool,
    ) -> int | None:
        assert env.scenario is not None
        stage = env.scenario.services[service_id].stages[stage_id]
        already = deployment[service_id, stage_id]
        mask = (
            (remaining_memory >= stage.memory_gb)
            & (remaining_storage >= stage.storage_gb)
            & ~already
        )
        if not mask.any():
            return None
        state = self._build_state(
            env,
            demand,
            remaining_memory,
            remaining_storage,
            service_id,
            stage_id,
            replica_idx,
            planned_deployment=deployment,
        )
        action, logprob, value = self.placement_ppo.act(
            state,
            mask,
            deterministic=deterministic,
        )
        if not mask[action]:
            action = int(np.flatnonzero(mask)[0])
        if record:
            self.placement_ppo.buffer.states.append(state)
            self.placement_ppo.buffer.masks.append(mask.astype(bool))
            self.placement_ppo.buffer.actions.append(action)
            self.placement_ppo.buffer.logprobs.append(logprob)
            self.placement_ppo.buffer.values.append(value)
            self.pending_placement_indices.append(len(self.placement_ppo.buffer.actions) - 1)
            self.pending_placement_stage_keys.append((service_id, stage_id))
            assert window_id is not None
            self.placement_window_ids.append(window_id)
        deployment[service_id, stage_id, action] = True
        remaining_memory[action] -= stage.memory_gb
        remaining_storage[action] -= stage.storage_gb
        return int(action)

    def assign_pending_reward(
        self,
        reward: float,
        done: bool,
        *,
        stage_returns: dict[tuple[int, int], float] | None = None,
        count_stage_returns: dict[tuple[int, ...], float] | None = None,
        placement_stage_returns: dict[tuple[int, ...], float] | None = None,
    ) -> None:
        if self.pending_window_id is None:
            return
        if self.pending_window_id != len(self.window_returns):
            raise RuntimeError("slow deployment windows must be completed in collection order")
        self.window_returns.append(float(reward))
        self.window_dones.append(bool(done))
        self.window_episode_ids.append(self.critic_episode_index)
        if done:
            self.critic_episode_index += 1
        stage_returns = {} if stage_returns is None else stage_returns
        count_stage_returns = stage_returns if count_stage_returns is None else count_stage_returns
        placement_stage_returns = stage_returns if placement_stage_returns is None else placement_stage_returns
        self.count_action_stage_keys.extend(self.pending_count_stage_keys)
        if len(self.pending_placement_indices) != len(self.pending_placement_stage_keys):
            raise ValueError("slow placement action indices and stage keys are misaligned")
        if len(self.pending_count_indices) != len(self.pending_count_node_ids):
            raise ValueError("slow count action indices and node ids are misaligned")
        if len(self.pending_count_stage_keys) != len(self.pending_count_node_ids):
            raise ValueError("slow count stage keys and node ids are misaligned")
        if self.pending_count_node_ids:
            self.count_action_returns.extend(
                float(
                    count_stage_returns.get(
                        (stage_key[0], stage_key[1], int(node_id)),
                        count_stage_returns.get(stage_key, reward),
                    )
                )
                if node_id is not None
                else float(count_stage_returns.get(stage_key, reward))
                for stage_key, node_id in zip(
                    self.pending_count_stage_keys,
                    self.pending_count_node_ids,
                )
            )
        else:
            self.count_action_returns.extend(
                float(count_stage_returns.get(stage_key, reward))
                for stage_key in self.pending_count_stage_keys
            )
        self.placement_action_stage_keys.extend(self.pending_placement_stage_keys)
        for action_index, stage_key in zip(
            self.pending_placement_indices,
            self.pending_placement_stage_keys,
        ):
            node_id = int(self.placement_ppo.buffer.actions[action_index])
            node_key = (stage_key[0], stage_key[1], node_id)
            self.placement_action_returns.append(
                float(
                    placement_stage_returns.get(
                        node_key,
                        placement_stage_returns.get(stage_key, reward),
                    )
                )
            )
        self.pending_window_id = None
        self.pending_count_indices.clear()
        self.pending_placement_indices.clear()
        self.pending_count_stage_keys.clear()
        self.pending_count_node_ids.clear()
        self.pending_placement_stage_keys.clear()

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        if self.pending_window_id is not None:
            raise ValueError("slow deployment buffer contains a window without a return")
        if not self.window_returns:
            return self.empty_update_metrics()
        if len(self.window_states) != len(self.window_returns):
            raise ValueError("slow deployment window states and returns are misaligned")
        if len(self.window_dones) != len(self.window_returns):
            raise ValueError("slow deployment window dones and returns are misaligned")
        if len(self.window_episode_ids) != len(self.window_returns):
            raise ValueError("slow deployment window episode ids and returns are misaligned")

        immediate_returns = np.asarray(self.window_returns, dtype=np.float32)
        trajectory_returns = _discounted_window_returns(
            immediate_returns,
            np.asarray(self.window_dones, dtype=bool),
            self.window_gamma,
        )
        old_values = np.asarray(self.window_old_values, dtype=np.float32)
        window_advantages = trajectory_returns - old_values
        window_advantage_mean = float(window_advantages.mean())
        window_advantage_std = float(window_advantages.std())

        count_ids = np.asarray(self.count_window_ids, dtype=np.int64)
        placement_ids = np.asarray(self.placement_window_ids, dtype=np.int64)
        if len(self.count_action_returns) != len(self.count_ppo.buffer.actions):
            raise ValueError("slow count action returns are misaligned")
        if len(self.count_action_stage_keys) != len(self.count_ppo.buffer.actions):
            raise ValueError("slow count action stage keys are misaligned")
        if len(self.placement_action_returns) != len(self.placement_ppo.buffer.actions):
            raise ValueError("slow placement action returns are misaligned")
        if len(self.placement_action_stage_keys) != len(self.placement_ppo.buffer.actions):
            raise ValueError("slow placement action stage keys are misaligned")
        count_returns = np.asarray(self.count_action_returns, dtype=np.float32)
        placement_returns = np.asarray(self.placement_action_returns, dtype=np.float32)
        placement_group_ids = np.asarray(
            [
                service_id * self.max_service_stages + stage_id
                for service_id, stage_id in self.placement_action_stage_keys
            ],
            dtype=np.int64,
        )
        placement_action_weights = self._equal_window_action_weights(placement_ids)
        placement_attribution = self._placement_credit_attribution(
            placement_returns,
            placement_group_ids,
            placement_action_weights,
        )
        global_advantage_reliability = self.global_advantage_reliability()
        effective_count_global_coef = (
            self.count_global_advantage_coef * global_advantage_reliability
        )
        effective_placement_global_coef = (
            self.placement_global_advantage_coef * global_advantage_reliability
        )
        count_advantage_groups = (
            np.zeros(len(self.count_action_stage_keys), dtype=np.int64)
            if self.allocation_mode == "global-incremental"
            else np.asarray(
                [
                    service_id * self.max_service_stages + stage_id
                    for service_id, stage_id in self.count_action_stage_keys
                ],
                dtype=np.int64,
            )
        )
        count_metrics = self.count_ppo.update_from_returns(
            count_returns,
            sample_weights=self._equal_window_action_weights(count_ids),
            advantage_group_ids=count_advantage_groups,
            actor_use_value_baseline=False,
            auxiliary_advantages=window_advantages[count_ids],
            auxiliary_advantage_coef=effective_count_global_coef,
            progress_label=f"{progress_label} count",
            progress_interval_seconds=progress_interval_seconds,
        )
        placement_entropy_coef = self.placement_entropy_coefficient()
        self.placement_ppo.entropy_coef = placement_entropy_coef
        placement_metrics = self.placement_ppo.update_from_returns(
            placement_returns,
            sample_weights=placement_action_weights,
            advantage_group_ids=placement_group_ids,
            actor_use_value_baseline=False,
            auxiliary_advantages=window_advantages[placement_ids],
            auxiliary_advantage_coef=effective_placement_global_coef,
            auxiliary_attribution=placement_attribution,
            auxiliary_attribution_coef=self.placement_global_attribution_coef,
            progress_label=f"{progress_label} placement",
            progress_interval_seconds=progress_interval_seconds,
        )
        critic_metrics = self._update_window_critic(trajectory_returns)
        self.last_window_critic_holdout_ev = float(
            critic_metrics["holdout_explained_variance"]
        )
        placement_entropy_next_coef = self._adapt_placement_entropy_coefficient(
            placement_metrics.get("entropy", 0.0)
        )
        window_count = len(self.window_returns)
        self.count_window_ids.clear()
        self.placement_window_ids.clear()
        self.count_action_returns.clear()
        self.placement_action_returns.clear()
        self.count_action_stage_keys.clear()
        self.placement_action_stage_keys.clear()
        self.window_states.clear()
        self.window_old_values.clear()
        self.window_returns.clear()
        self.window_dones.clear()
        self.window_episode_ids.clear()
        return {
            "loss": count_metrics["loss"] + placement_metrics["loss"] + self.value_coef * critic_metrics["value_loss"],
            "policy_loss": count_metrics["policy_loss"] + placement_metrics["policy_loss"],
            "value_loss": count_metrics["value_loss"] + placement_metrics["value_loss"],
            "entropy": count_metrics["entropy"] + placement_metrics["entropy"],
            "approx_kl": max(count_metrics.get("approx_kl", 0.0), placement_metrics.get("approx_kl", 0.0)),
            "window_count": float(window_count),
            "window_return_mean": float(immediate_returns.mean()),
            "window_return_std": float(immediate_returns.std()),
            "trajectory_return_mean": float(trajectory_returns.mean()),
            "trajectory_return_std": float(trajectory_returns.std()),
            "count_return_mean": float(count_returns.mean()) if count_returns.size else 0.0,
            "count_return_std": float(count_returns.std()) if count_returns.size else 0.0,
            "placement_return_mean": float(placement_returns.mean()) if placement_returns.size else 0.0,
            "placement_return_std": float(placement_returns.std()) if placement_returns.size else 0.0,
            "advantage_mean": 0.5
            * (count_metrics.get("advantage_mean", 0.0) + placement_metrics.get("advantage_mean", 0.0)),
            "advantage_std": 0.5
            * (count_metrics.get("advantage_std", 0.0) + placement_metrics.get("advantage_std", 0.0)),
            "window_advantage_mean": window_advantage_mean,
            "window_advantage_std": window_advantage_std,
            "critic_explained_variance": 0.5
            * (
                count_metrics.get("explained_variance", 0.0)
                + placement_metrics.get("explained_variance", 0.0)
            ),
            "window_critic_explained_variance": critic_metrics["explained_variance"],
            "window_critic_value_loss": critic_metrics["value_loss"],
            "window_critic_raw_value_mse": critic_metrics["raw_value_mse"],
            "window_critic_train_explained_variance": critic_metrics[
                "train_explained_variance"
            ],
            "window_critic_holdout_explained_variance": critic_metrics[
                "holdout_explained_variance"
            ],
            "window_critic_replay_size": critic_metrics["replay_size"],
            "window_critic_train_size": critic_metrics["train_size"],
            "window_critic_holdout_size": critic_metrics["holdout_size"],
            "window_critic_train_episode_count": critic_metrics["train_episode_count"],
            "window_critic_holdout_episode_count": critic_metrics["holdout_episode_count"],
            "window_critic_return_mean": self.critic_return_mean,
            "window_critic_return_std": self.critic_return_std,
            "global_advantage_reliability": global_advantage_reliability,
            "count_loss": count_metrics["loss"],
            "count_advantage_mean": count_metrics.get("advantage_mean", 0.0),
            "count_advantage_std": count_metrics.get("advantage_std", 0.0),
            "count_global_advantage_mean": count_metrics.get("auxiliary_advantage_mean", 0.0),
            "count_global_advantage_std": count_metrics.get("auxiliary_advantage_std", 0.0),
            "count_global_advantage_coef": effective_count_global_coef,
            "count_global_advantage_configured_coef": self.count_global_advantage_coef,
            "count_combined_advantage_std": count_metrics.get("combined_advantage_std", 0.0),
            "count_policy_loss": count_metrics["policy_loss"],
            "count_value_loss": count_metrics["value_loss"],
            "count_entropy": count_metrics["entropy"],
            "count_approx_kl": count_metrics.get("approx_kl", 0.0),
            "count_explained_variance": count_metrics.get("explained_variance", 0.0),
            "count_post_explained_variance": count_metrics.get("post_explained_variance", 0.0),
            "placement_loss": placement_metrics["loss"],
            "placement_advantage_mean": placement_metrics.get("advantage_mean", 0.0),
            "placement_advantage_std": placement_metrics.get("advantage_std", 0.0),
            "placement_global_advantage_mean": placement_metrics.get("auxiliary_advantage_mean", 0.0),
            "placement_global_advantage_std": placement_metrics.get("auxiliary_advantage_std", 0.0),
            "placement_global_advantage_coef": effective_placement_global_coef,
            "placement_global_advantage_configured_coef": self.placement_global_advantage_coef,
            "placement_global_attribution_coef": self.placement_global_attribution_coef,
            "placement_global_attribution_mean_abs": placement_metrics.get(
                "auxiliary_attribution_mean_abs", 0.0
            ),
            "placement_combined_advantage_std": placement_metrics.get("combined_advantage_std", 0.0),
            "placement_policy_loss": placement_metrics["policy_loss"],
            "placement_value_loss": placement_metrics["value_loss"],
            "placement_entropy": placement_metrics["entropy"],
            "placement_entropy_coef": placement_entropy_coef,
            "placement_entropy_next_coef": placement_entropy_next_coef,
            "placement_entropy_schedule_coef": self.placement_entropy_schedule_coefficient(),
            "placement_entropy_target": (
                float(self.placement_entropy_target) if self.placement_entropy_target is not None else 0.0
            ),
            "placement_entropy_shortfall": max(
                float(self.placement_entropy_target or 0.0)
                - float(placement_metrics.get("entropy", 0.0)),
                0.0,
            ),
            "placement_entropy_saturated": float(
                np.isclose(placement_entropy_next_coef, self.placement_entropy_max_coef)
            ),
            "placement_updates_completed": float(self.placement_updates_completed),
            "placement_approx_kl": placement_metrics.get("approx_kl", 0.0),
            "placement_explained_variance": placement_metrics.get("explained_variance", 0.0),
            "placement_post_explained_variance": placement_metrics.get("post_explained_variance", 0.0),
        }

    def empty_update_metrics(self) -> dict[str, float]:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "window_count": 0.0,
            "window_return_mean": 0.0,
            "window_return_std": 0.0,
            "trajectory_return_mean": 0.0,
            "trajectory_return_std": 0.0,
            "count_return_mean": 0.0,
            "count_return_std": 0.0,
            "placement_return_mean": 0.0,
            "placement_return_std": 0.0,
            "advantage_mean": 0.0,
            "advantage_std": 0.0,
            "window_advantage_mean": 0.0,
            "window_advantage_std": 0.0,
            "critic_explained_variance": 0.0,
            "window_critic_explained_variance": 0.0,
            "window_critic_value_loss": 0.0,
            "window_critic_raw_value_mse": 0.0,
            "window_critic_train_explained_variance": 0.0,
            "window_critic_holdout_explained_variance": self.last_window_critic_holdout_ev,
            "window_critic_replay_size": float(len(self.critic_replay_returns)),
            "window_critic_train_size": 0.0,
            "window_critic_holdout_size": 0.0,
            "window_critic_train_episode_count": 0.0,
            "window_critic_holdout_episode_count": 0.0,
            "window_critic_return_mean": self.critic_return_mean,
            "window_critic_return_std": self.critic_return_std,
            "global_advantage_reliability": self.global_advantage_reliability(),
            "count_loss": 0.0,
            "count_advantage_mean": 0.0,
            "count_advantage_std": 0.0,
            "count_global_advantage_mean": 0.0,
            "count_global_advantage_std": 0.0,
            "count_global_advantage_coef": 0.0,
            "count_global_advantage_configured_coef": self.count_global_advantage_coef,
            "count_combined_advantage_std": 0.0,
            "count_policy_loss": 0.0,
            "count_value_loss": 0.0,
            "count_entropy": 0.0,
            "count_approx_kl": 0.0,
            "count_explained_variance": 0.0,
            "count_post_explained_variance": 0.0,
            "placement_loss": 0.0,
            "placement_advantage_mean": 0.0,
            "placement_advantage_std": 0.0,
            "placement_global_advantage_mean": 0.0,
            "placement_global_advantage_std": 0.0,
            "placement_global_advantage_coef": 0.0,
            "placement_global_advantage_configured_coef": self.placement_global_advantage_coef,
            "placement_global_attribution_coef": self.placement_global_attribution_coef,
            "placement_global_attribution_mean_abs": 0.0,
            "placement_combined_advantage_std": 0.0,
            "placement_policy_loss": 0.0,
            "placement_value_loss": 0.0,
            "placement_entropy": 0.0,
            "placement_entropy_coef": self.placement_entropy_coefficient(),
            "placement_entropy_next_coef": self.placement_entropy_coefficient(),
            "placement_entropy_schedule_coef": self.placement_entropy_schedule_coefficient(),
            "placement_entropy_target": (
                float(self.placement_entropy_target) if self.placement_entropy_target is not None else 0.0
            ),
            "placement_entropy_shortfall": 0.0,
            "placement_entropy_saturated": float(
                np.isclose(self.placement_entropy_coefficient(), self.placement_entropy_max_coef)
            ),
            "placement_updates_completed": float(self.placement_updates_completed),
            "placement_approx_kl": 0.0,
            "placement_explained_variance": 0.0,
            "placement_post_explained_variance": 0.0,
        }

    def _equal_window_action_weights(self, window_ids: np.ndarray) -> np.ndarray:
        if len(window_ids) == 0:
            return np.asarray([], dtype=np.float32)
        counts = np.bincount(window_ids, minlength=len(self.window_returns)).astype(np.float32)
        return np.asarray([1.0 / max(float(counts[idx]), 1.0) for idx in window_ids], dtype=np.float32)

    @staticmethod
    def _placement_credit_attribution(
        local_returns: np.ndarray,
        group_ids: np.ndarray,
        sample_weights: np.ndarray,
    ) -> np.ndarray:
        """Score node choices relative to comparable service-stage actions.

        The Slow critic emits one residual per ten-minute window.  Without this
        attribution every node chosen in that window receives exactly the same
        global credit.  The standardized local return preserves the mean window
        signal while shifting bonus toward good nodes and penalty toward bad
        nodes.
        """

        values = np.asarray(local_returns, dtype=np.float32)
        groups = np.asarray(group_ids)
        weights = np.asarray(sample_weights, dtype=np.float32)
        if values.shape != groups.shape or values.shape != weights.shape:
            raise ValueError("placement credit arrays must have matching shapes")
        attribution = np.zeros_like(values)
        for group_id in np.unique(groups):
            indices = np.flatnonzero(groups == group_id)
            group_weights = weights[indices]
            group_values = values[indices]
            mean = float(np.average(group_values, weights=group_weights))
            variance = float(
                np.average(np.square(group_values - mean), weights=group_weights)
            )
            std = float(np.sqrt(variance))
            if std > 1e-8:
                attribution[indices] = (group_values - mean) / (std + 1e-8)
        return np.clip(attribution, -2.0, 2.0)

    def _update_window_critic(self, returns: np.ndarray) -> dict[str, float]:
        current_states = np.stack(self.window_states).astype(np.float32, copy=False)
        current_returns = np.asarray(returns, dtype=np.float32)
        self.critic_update_index += 1
        self.critic_replay_states.extend(state.copy() for state in current_states)
        self.critic_replay_returns.extend(float(value) for value in current_returns)
        self.critic_replay_update_ids.extend(
            [self.critic_update_index] * len(current_returns)
        )
        self.critic_replay_episode_ids.extend(self.window_episode_ids)
        while len(self.critic_replay_returns) > self.critic_replay_windows:
            oldest_episode = self.critic_replay_episode_ids[0]
            drop_count = 0
            while (
                drop_count < len(self.critic_replay_episode_ids)
                and self.critic_replay_episode_ids[drop_count] == oldest_episode
            ):
                drop_count += 1
            del self.critic_replay_states[:drop_count]
            del self.critic_replay_returns[:drop_count]
            del self.critic_replay_update_ids[:drop_count]
            del self.critic_replay_episode_ids[:drop_count]

        replay_states = np.stack(self.critic_replay_states).astype(np.float32, copy=False)
        replay_returns = np.asarray(self.critic_replay_returns, dtype=np.float32)
        replay_update_ids = np.asarray(self.critic_replay_update_ids, dtype=np.int64)
        replay_episode_ids = np.asarray(self.critic_replay_episode_ids, dtype=np.int64)
        unique_episodes = np.unique(replay_episode_ids)
        holdout_episode_count = min(self.critic_holdout_episodes, len(unique_episodes))
        holdout_episode_ids = unique_episodes[-holdout_episode_count:]
        holdout_mask = np.isin(replay_episode_ids, holdout_episode_ids)
        train_indices = np.flatnonzero(~holdout_mask)
        holdout_indices = np.flatnonzero(holdout_mask)

        value_loss = 0.0
        if len(train_indices) > 0:
            train_weights = np.power(
                float(self.critic_replay_decay),
                (self.critic_update_index - replay_update_ids[train_indices]).astype(
                    np.float64
                ),
            ).astype(np.float32)
            train_weights /= max(float(train_weights.mean()), 1e-8)
            weighted_mean = float(
                np.average(replay_returns[train_indices], weights=train_weights)
            )
            weighted_variance = float(
                np.average(
                    np.square(replay_returns[train_indices] - weighted_mean),
                    weights=train_weights,
                )
            )
            self.critic_return_mean = weighted_mean
            self.critic_return_std = max(float(np.sqrt(weighted_variance)), 1e-3)

            minibatch_size = min(32, len(train_indices))
            for _ in range(max(self.critic_k_epochs, 1)):
                order = np.random.permutation(train_indices)
                for start in range(0, len(train_indices), minibatch_size):
                    batch_indices = order[start : start + minibatch_size]
                    states = torch.as_tensor(
                        replay_states[batch_indices],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    normalized_targets = torch.as_tensor(
                        (replay_returns[batch_indices] - self.critic_return_mean)
                        / self.critic_return_std,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    weights = torch.as_tensor(
                        train_weights[batch_indices],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    predictions = self.window_critic(states)
                    losses = nn.functional.smooth_l1_loss(
                        predictions,
                        normalized_targets,
                        reduction="none",
                    )
                    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.window_critic.parameters(),
                        self.critic_gradient_clip,
                    )
                    self.critic_optimizer.step()
                    value_loss = float(loss.item())

        def raw_predictions(states_np: np.ndarray) -> np.ndarray:
            if len(states_np) == 0:
                return np.asarray([], dtype=np.float32)
            with torch.no_grad():
                normalized = self.window_critic(
                    torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
                ).detach().cpu().numpy()
            return normalized * self.critic_return_std + self.critic_return_mean

        def explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float:
            if len(targets) == 0 or len(predictions) == 0:
                return 0.0
            target_variance = float(np.var(targets))
            if target_variance <= 1e-8:
                return 0.0
            return 1.0 - float(np.var(targets - predictions)) / target_variance

        current_predictions = raw_predictions(current_states)
        train_predictions = raw_predictions(replay_states[train_indices])
        holdout_predictions = raw_predictions(replay_states[holdout_indices])
        return {
            "value_loss": value_loss,
            "raw_value_mse": float(
                np.mean(np.square(current_returns - current_predictions))
            ),
            "explained_variance": explained_variance(
                current_returns, current_predictions
            ),
            "train_explained_variance": explained_variance(
                replay_returns[train_indices], train_predictions
            ),
            "holdout_explained_variance": explained_variance(
                replay_returns[holdout_indices], holdout_predictions
            ),
            "replay_size": float(len(replay_returns)),
            "train_size": float(len(train_indices)),
            "holdout_size": float(len(holdout_indices)),
            "train_episode_count": float(len(np.unique(replay_episode_ids[train_indices]))),
            "holdout_episode_count": float(len(np.unique(replay_episode_ids[holdout_indices]))),
        }

    def _build_state(
        self,
        env: EdgeComputingEnv,
        demand: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        service_id: int,
        stage_id: int,
        replica_idx: int,
        planned_deployment: np.ndarray | None = None,
    ) -> np.ndarray:
        assert env.scenario is not None
        nodes = env.scenario.nodes
        max_mem = max(n.memory_gb for n in nodes)
        max_storage = max(n.storage_gb for n in nodes)
        max_compute = max(n.compute_gcycles_per_s for n in nodes)
        total_node_demand = demand.sum(axis=1)
        max_node_demand = max(float(total_node_demand.max()), 1e-9)
        affinity_nodes = np.zeros(self.num_nodes, dtype=np.float32)
        placement = planned_deployment if planned_deployment is not None else env.deployment
        if placement is not None and stage_id > 0:
            affinity_nodes = placement[service_id, stage_id - 1].astype(np.float32)
        node_features = []
        for node in nodes:
            node_features.extend(
                [
                    remaining_memory[node.node_id] / max_mem,
                    remaining_storage[node.node_id] / max_storage,
                    node.compute_gcycles_per_s / max_compute,
                    env.node_compute_load[node.node_id],
                    total_node_demand[node.node_id] / max_node_demand,
                    affinity_nodes[node.node_id],
                ]
            )
            node_features.extend(demand[node.node_id].tolist())
        scalars = [
            service_id / max(self.num_service_types - 1, 1),
            stage_id / max(self.max_service_stages - 1, 1),
            replica_idx / max(self.replicas_per_stage - 1, 1),
            env.current_time_minute / max(env.config.episode_minutes, 1),
            len(env.scenario.services[service_id].stages) / self.max_service_stages,
            np.log1p(env._arrival_rate_per_minute()) / np.log1p(20_000.0),
            np.log1p(env._arrival_rate_per_minute() * env.config.deployment_interval_minutes)
            / np.log1p(5_000_000.0),
            1.0,
        ]
        return np.asarray(
            scalars + self._feedback_features() + node_features + self._build_edge_features(env),
            dtype=np.float32,
        )

    def _build_window_state(
        self,
        env: EdgeComputingEnv,
        demand: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        planned_deployment: np.ndarray | None = None,
    ) -> np.ndarray:
        assert env.scenario is not None
        nodes = env.scenario.nodes
        memory_capacity = env.service_memory_capacities()
        storage_capacity = env.service_storage_capacities()
        max_compute = max(n.compute_gcycles_per_s for n in nodes)
        total_node_demand = demand.sum(axis=1)
        max_node_demand = max(float(total_node_demand.max()), 1e-9)
        active_deployment = (
            planned_deployment if planned_deployment is not None else env.deployment
        )
        deployed_nodes = (
            np.any(active_deployment, axis=(0, 1)).astype(np.float32)
            if active_deployment is not None
            else np.zeros(self.num_nodes, dtype=np.float32)
        )
        node_features: list[float] = []
        for node in nodes:
            node_features.extend(
                [
                    remaining_memory[node.node_id] / max(memory_capacity[node.node_id], 1e-9),
                    remaining_storage[node.node_id] / max(storage_capacity[node.node_id], 1e-9),
                    node.compute_gcycles_per_s / max_compute,
                    env.node_compute_load[node.node_id],
                    total_node_demand[node.node_id] / max_node_demand,
                    deployed_nodes[node.node_id],
                ]
            )
            node_features.extend(demand[node.node_id].tolist())
        current_replica_rate = 0.0
        if active_deployment is not None:
            current_replica_rate = float(active_deployment.mean())
        finite_link_load = env.link_load[np.isfinite(env.link_load)]
        mean_link_load = float(np.mean(finite_link_load)) if finite_link_load.size else 0.0
        max_link_load = float(np.max(finite_link_load)) if finite_link_load.size else 0.0
        scalars = [
            env.current_time_minute / max(env.config.episode_minutes, 1),
            np.log1p(env._arrival_rate_per_minute()) / np.log1p(20_000.0),
            np.log1p(env._arrival_rate_per_minute() * env.config.deployment_interval_minutes)
            / np.log1p(5_000_000.0),
            current_replica_rate,
            float(np.mean(env.node_compute_load)),
            float(np.max(env.node_compute_load)),
            mean_link_load,
            max_link_load,
        ]
        return np.asarray(
            scalars + self._feedback_features() + node_features + self._build_edge_features(env),
            dtype=np.float32,
        )

    def _build_incremental_state(
        self,
        env: EdgeComputingEnv,
        demand: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        deployment: np.ndarray,
    ) -> np.ndarray:
        """Insert per-stage marginal-resource features into the graph state."""

        assert env.scenario is not None
        base = self._build_window_state(
            env,
            demand,
            remaining_memory,
            remaining_storage,
            planned_deployment=deployment,
        )
        node_end = SLOW_GLOBAL_DIM + self.num_nodes * slow_node_feature_dim(
            self.num_service_types
        )
        global_and_nodes = base[:node_end]
        edge_features = base[node_end:]
        stage_features: list[float] = []
        maximum_stage_demand = max(float(demand.sum(axis=0).max()), 1e-9)
        memory_capacity = np.maximum(env.service_memory_capacities(), 1e-9)
        storage_capacity = np.maximum(env.service_storage_capacities(), 1e-9)
        for service_id in range(self.num_service_types):
            service = env.scenario.services[service_id]
            for stage_id in range(self.max_service_stages):
                valid = stage_id < len(service.stages)
                if not valid:
                    stage_features.extend([0.0] * SLOW_STAGE_FEATURE_DIM)
                    continue
                stage = service.stages[stage_id]
                placed = deployment[service_id, stage_id]
                feasible = (
                    (remaining_memory >= stage.memory_gb)
                    & (remaining_storage >= stage.storage_gb)
                    & ~placed
                )
                previous_overlap = (
                    float(np.mean(placed & deployment[service_id, stage_id - 1]))
                    if stage_id > 0
                    else 0.0
                )
                next_overlap = (
                    float(np.mean(placed & deployment[service_id, stage_id + 1]))
                    if stage_id + 1 < len(service.stages)
                    else 0.0
                )
                feasible_remaining = np.minimum(
                    remaining_memory / memory_capacity,
                    remaining_storage / storage_capacity,
                )
                stage_features.extend(
                    [
                        1.0,
                        float(demand[:, service_id].sum()) / maximum_stage_demand,
                        float(stage.compute_gcycles_mean) / 8.0,
                        float(np.mean(stage.memory_gb / memory_capacity)),
                        float(np.mean(stage.storage_gb / storage_capacity)),
                        float(placed.sum()) / max(float(self.replicas_per_stage), 1.0),
                        float(np.mean(feasible)),
                        float(np.mean(feasible_remaining[feasible])) if feasible.any() else 0.0,
                        previous_overlap,
                        next_overlap,
                    ]
                )
        return np.concatenate(
            [
                global_and_nodes,
                np.asarray(stage_features, dtype=np.float32),
                edge_features,
            ]
        ).astype(np.float32, copy=False)

    def _build_incremental_stage_mask(
        self,
        env: EdgeComputingEnv,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        deployment: np.ndarray,
    ) -> np.ndarray:
        assert env.scenario is not None
        stop_action = self.num_service_types * self.max_service_stages
        mask = np.zeros(stop_action + 1, dtype=bool)
        for service in env.scenario.services:
            for stage in service.stages:
                stage_key = service.service_id * self.max_service_stages + stage.stage_id
                below_cap = int(deployment[service.service_id, stage.stage_id].sum()) < self.replicas_per_stage
                feasible = (
                    (remaining_memory >= stage.memory_gb)
                    & (remaining_storage >= stage.storage_gb)
                    & ~deployment[service.service_id, stage.stage_id]
                )
                mask[stage_key] = bool(below_cap and feasible.any())
        mask[stop_action] = True
        return mask

    def _feedback_features(self) -> list[float]:
        feedback = self.last_window_feedback
        return [
            float(np.clip(feedback.get("avg_latency_s", 0.0), 0.0, 10.0)),
            float(np.clip(feedback.get("p95_latency_s", 0.0), 0.0, 10.0)),
            float(np.clip(feedback.get("avg_penalty_latency_s", 0.0) / 10.0, 0.0, 1.0)),
            float(np.clip(feedback.get("deadline_violation_rate", 0.0), 0.0, 1.0)),
            float(np.clip(feedback.get("invalid_action_rate", 0.0), 0.0, 1.0)),
            float(np.clip(feedback.get("max_node_compute_load", 0.0), 0.0, 1.0)),
            float(np.clip(feedback.get("max_link_load", 0.0), 0.0, 1.0)),
            float(np.clip(feedback.get("deployment_memory_fraction", 0.0), 0.0, 1.0)),
            float(np.clip(feedback.get("deployment_storage_fraction", 0.0), 0.0, 1.0)),
        ]

    def _build_edge_features(self, env: EdgeComputingEnv) -> list[float]:
        assert env.scenario is not None
        bandwidth = env.scenario.bandwidth_mb_s
        propagation = env.scenario.propagation_ms
        max_bandwidth = np.nanmax(np.where(np.isfinite(bandwidth), bandwidth, 0.0))
        finite_propagation = np.where(np.isfinite(propagation), propagation, 0.0)
        max_propagation = max(float(finite_propagation.max()), 1e-9)
        features: list[float] = []
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                connected = bool(env.scenario.adjacency[src, dst])
                bw = bandwidth[src, dst] if np.isfinite(bandwidth[src, dst]) else max_bandwidth
                prop = propagation[src, dst] if np.isfinite(propagation[src, dst]) else max_propagation
                load = env.link_load[src, dst] if np.isfinite(env.link_load[src, dst]) else 0.0
                features.extend(
                    [
                        float(connected),
                        float(bw / max(max_bandwidth, 1e-9)),
                        float(prop / max_propagation),
                        float(np.clip(load, 0.0, 1.0)),
                    ]
                )
        return features

    def _node_service_demand(self, env: EdgeComputingEnv) -> np.ndarray:
        assert env.scenario is not None
        demand = np.zeros((self.num_nodes, self.num_service_types), dtype=np.float32)
        for user in env.scenario.users:
            demand[user.home_node] += np.asarray(user.service_weights, dtype=np.float32)
        demand /= max(float(len(env.scenario.users)), 1.0)
        demand *= float(env._arrival_rate_per_minute())
        return np.log1p(demand) / np.log1p(500.0)

    def _build_count_mask(
        self,
        stage,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
        *,
        reserved_memory: float = 0.0,
        reserved_storage: float = 0.0,
    ) -> np.ndarray:
        feasible_slots = int(np.count_nonzero((remaining_memory >= stage.memory_gb) & (remaining_storage >= stage.storage_gb)))
        memory_budget = max(float(remaining_memory.sum()) - reserved_memory, 0.0)
        storage_budget = max(float(remaining_storage.sum()) - reserved_storage, 0.0)
        memory_count = int(np.floor(memory_budget / max(float(stage.memory_gb), 1e-9)))
        storage_count = int(np.floor(storage_budget / max(float(stage.storage_gb), 1e-9)))
        max_count = min(self.replicas_per_stage, feasible_slots, memory_count, storage_count)
        mask = np.zeros(self.replicas_per_stage, dtype=bool)
        if max_count > 0:
            mask[:max_count] = True
        return mask

    def _repair_with_current_or_full(self, env: EdgeComputingEnv, deployment: np.ndarray) -> np.ndarray:
        feasible, _ = env.check_deployment_feasible(deployment)
        if feasible:
            return deployment
        if env.deployment is not None:
            feasible, _ = env.check_deployment_feasible(env.deployment)
            if feasible:
                return env.deployment.copy()
        raise RuntimeError("slow PPO produced infeasible deployment and no feasible fallback exists")

    def _coverage_repair(
        self,
        env: EdgeComputingEnv,
        deployment: np.ndarray,
        remaining_memory: np.ndarray,
        remaining_storage: np.ndarray,
    ) -> None:
        """Repair uncovered service stages without overriding learned replica counts."""

        assert env.scenario is not None
        demand = self._node_service_demand(env)
        for service in env.scenario.services:
            node_order = np.argsort(-demand[:, service.service_id])
            for stage in service.stages:
                if deployment[service.service_id, stage.stage_id].any():
                    continue
                for node_id in node_order:
                    if remaining_memory[node_id] < stage.memory_gb or remaining_storage[node_id] < stage.storage_gb:
                        continue
                    deployment[service.service_id, stage.stage_id, node_id] = True
                    remaining_memory[node_id] -= stage.memory_gb
                    remaining_storage[node_id] -= stage.storage_gb
                    break


@dataclass
class FastSchedulingPPOAgent:
    num_nodes: int
    max_service_stages: int
    policy_kind: str = "gat_node_scorer"
    chain_context: str = "legacy"
    lr: float = 2e-4
    k_epochs: int = 4
    entropy_coef: float = 0.001
    entropy_target: float | None = 0.7
    entropy_max_coef: float = 0.03
    entropy_adaptation_rate: float = 2e-3
    entropy_normalized_control: bool = False
    value_coef: float = 0.5
    target_kl: float | None = 0.015
    minibatch_size: int = 512
    reservation_microbatch_size: int = 16
    path_candidate_limit: int = 32
    load_balanced_updates: bool = True
    load_group_weights: tuple[float, ...] | None = None
    full_batch_kl_stop: bool = True
    device: str = "cpu"
    ppo: PPOAgent = field(init=False)
    entropy_current_coef: float = field(default=0.0, init=False)
    _workload_cache_key: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _workload_cache: tuple[np.ndarray, dict[tuple[int, int], np.ndarray]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _path_selected_proxy_regrets: dict[int, list[float]] = field(default_factory=dict, init=False, repr=False)
    _phase_sqrt_cache_key: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _phase_sqrt_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _phase_sqrt_cache_request_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _structured_route_cache_key: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _structured_route_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )
    _phase_link_sqrt_cache_key: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _phase_link_sqrt_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _phase_link_sqrt_by_request: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _pending_kkt_batch_credits: list[float] = field(default_factory=list, init=False, repr=False)
    _pending_kkt_batch_counts: list[float] = field(default_factory=list, init=False, repr=False)
    _pending_kkt_batch_recorded: bool = field(default=False, init=False, repr=False)
    last_kkt_credit_diagnostics: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.reservation_microbatch_size < 1:
            raise ValueError("reservation_microbatch_size must be >= 1")
        if self.path_candidate_limit < 1:
            raise ValueError("path_candidate_limit must be >= 1")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        if self.entropy_target is not None and self.entropy_target < 0.0:
            raise ValueError("entropy_target must be non-negative")
        if self.entropy_max_coef < self.entropy_coef:
            raise ValueError("entropy_max_coef must be at least entropy_coef")
        if self.entropy_adaptation_rate < 0.0:
            raise ValueError("entropy_adaptation_rate must be non-negative")
        path_mode = self.policy_kind == "path_scorer"
        structured_mode = self.policy_kind == "structured_path"
        kkt_autoregressive_mode = self.policy_kind == "kkt_autoregressive_path"
        if kkt_autoregressive_mode:
            global_dim = FAST_KKT_GLOBAL_DIM
            node_feature_dim = FAST_KKT_NODE_FEATURE_DIM
        elif structured_mode:
            global_dim = FAST_PATH_GLOBAL_DIM
            node_feature_dim = FAST_STRUCT_NODE_FEATURE_DIM
        elif path_mode:
            global_dim = FAST_PATH_GLOBAL_DIM
            node_feature_dim = fast_path_feature_dim(self.max_service_stages)
        else:
            global_dim, node_feature_dim = fast_feature_dims(self.chain_context)
        self.entropy_current_coef = float(self.entropy_coef)
        self.ppo = PPOAgent(
            obs_dim=(
                fast_path_obs_dim(self.max_service_stages, self.path_candidate_limit)
                if path_mode
                else fast_kkt_autoregressive_obs_dim(self.num_nodes, self.max_service_stages)
                if kkt_autoregressive_mode
                else fast_structured_obs_dim(self.num_nodes, self.max_service_stages)
                if structured_mode
                else fast_obs_dim(self.num_nodes, self.policy_kind, self.chain_context)
            ),
            action_dim=(
                self.path_candidate_limit if path_mode else self.num_nodes ** self.max_service_stages
                if structured_mode or kkt_autoregressive_mode else self.num_nodes
            ),
            hidden_dim=128,
            lr=self.lr,
            gamma=1.0 if kkt_autoregressive_mode else 0.99,
            gae_lambda=1.0 if kkt_autoregressive_mode else 0.95,
            k_epochs=self.k_epochs,
            entropy_coef=self.entropy_current_coef,
            value_coef=self.value_coef,
            target_kl=self.target_kl,
            minibatch_size=self.minibatch_size,
            policy_kind="node_scorer" if path_mode else self.policy_kind,
            global_dim=global_dim,
            node_feature_dim=node_feature_dim,
            candidate_cost_feature_index=3 if path_mode else None,
            edge_feature_dim=(
                FAST_KKT_EDGE_FEATURE_DIM
                if kkt_autoregressive_mode
                else FAST_STRUCT_EDGE_FEATURE_DIM
                if structured_mode
                else FAST_EDGE_FEATURE_DIM
            ),
            max_service_stages=self.max_service_stages if structured_mode or kkt_autoregressive_mode else None,
            num_nodes=self.path_candidate_limit if path_mode else self.num_nodes,
            group_balanced_updates=self.load_balanced_updates,
            group_weight_targets=self.load_group_weights,
            full_batch_kl_stop=self.full_batch_kl_stop,
            device=self.device,
        )

    def schedule(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest | None = None,
        deterministic: bool = False,
        record: bool = True,
    ) -> list[int]:
        if request is None:
            assert env.current_request is not None
            request = env.current_request
        return self.schedule_batch(env, [request], deterministic=deterministic, record=record)[0]

    def schedule_batch(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        deterministic: bool = False,
        record: bool = True,
    ) -> list[list[int]]:
        """Schedule independent requests with stage-wise batched inference."""

        env._require_ready()
        if not requests:
            return []
        if self.policy_kind == "kkt_autoregressive_path":
            return self._schedule_kkt_autoregressive_batch(
                env,
                requests,
                deterministic=deterministic,
                record=record,
            )
        if self.policy_kind == "kkt_autoregressive_path":
            _, _, incidence, _ = self._structured_route_metrics(env)
            state, mask = self._kkt_autoregressive_path_state(
                env,
                request,
                remaining_requests=[request],
                batch_size=1,
                decision_index=0,
                compute_sqrt_ledger=np.zeros((self.max_service_stages, self.num_nodes), dtype=np.float64),
                link_sqrt_ledger=np.zeros(
                    (2 * self.max_service_stages - 1, incidence.shape[-1]), dtype=np.float64
                ),
            )
            stats = self.ppo.action_stats(state, mask)
            return self._decode_structured_path(int(stats["action"]), len(request.stage_compute_gcycles)), [stats]

        if self.policy_kind == "structured_path":
            return self._schedule_structured_batch(env, requests, deterministic=deterministic, record=record)
        if self.policy_kind == "path_scorer":
            return self._schedule_paths_batch(env, requests, deterministic=deterministic, record=record)

        schedules: list[list[int]] = [[] for _ in requests]
        transition_records: list[list[tuple[np.ndarray, np.ndarray, int, float, float]]] = [
            [] for _ in requests
        ]
        max_stages = max(len(request.stage_compute_gcycles) for request in requests)
        reservation_node_delta = np.zeros(self.num_nodes, dtype=np.float64)
        reservation_stage_delta: dict[tuple[int, int], np.ndarray] = {}
        for stage_id in range(max_stages):
            active_indices = [
                request_idx
                for request_idx, request in enumerate(requests)
                if stage_id < len(request.stage_compute_gcycles)
            ]
            if not active_indices:
                continue
            for chunk_start in range(0, len(active_indices), self.reservation_microbatch_size):
                chunk = active_indices[chunk_start : chunk_start + self.reservation_microbatch_size]
                states = [
                    self._build_state(
                        env,
                        requests[request_idx],
                        stage_id,
                        schedules[request_idx],
                        reservation_node_delta=reservation_node_delta,
                        reservation_stage_delta=reservation_stage_delta,
                    )
                    for request_idx in chunk
                ]
                masks = [
                    self._build_mask(env, requests[request_idx], stage_id, schedules[request_idx])
                    for request_idx in chunk
                ]
                actions, logprobs, values = self.ppo.act_batch(states, masks, deterministic=deterministic)
                for local_idx, request_idx in enumerate(chunk):
                    action = int(actions[local_idx])
                    if not masks[local_idx][action]:
                        action = int(np.where(masks[local_idx])[0][0])
                    schedules[request_idx].append(action)
                    if record:
                        transition_records[request_idx].append(
                            (
                                states[local_idx],
                                masks[local_idx].astype(bool),
                                action,
                                float(logprobs[local_idx]),
                                float(values[local_idx]),
                            )
                        )
                for local_idx, request_idx in enumerate(chunk):
                    self._reserve_compute_work(
                        env,
                        requests[request_idx],
                        stage_id,
                        int(schedules[request_idx][-1]),
                        reservation_node_delta,
                        reservation_stage_delta,
                    )
        if record:
            # Keep transition order request-major because rollout rewards are
            # assigned after env.step in that same request order.
            for request_records in transition_records:
                for state, mask, action, logprob, value in request_records:
                    self.ppo.buffer.states.append(state)
                    self.ppo.buffer.masks.append(mask)
                    self.ppo.buffer.actions.append(action)
                    self.ppo.buffer.logprobs.append(logprob)
                    self.ppo.buffer.values.append(value)
                    load_group = env.config.demand_load_group
                    self.ppo.buffer.sample_groups.append(
                        float(env.config.demand_load_multiplier if load_group is None else load_group)
                    )
        return schedules

    def _expected_phase_sqrt_load(self, env: EdgeComputingEnv) -> np.ndarray:
        """Expected KKT sqrt-load in each serial compute phase and node."""

        env._require_ready()
        assert env.scenario is not None and env.deployment is not None
        key = (id(env.current_requests), float(env.current_time_minute), float(env.metrics["deployment_updates"]))
        if key == self._phase_sqrt_cache_key and self._phase_sqrt_cache is not None:
            return self._phase_sqrt_cache
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64)
        sqrt_load = np.zeros((self.max_service_stages, self.num_nodes), dtype=np.float64)
        for current in env.current_requests:
            for stage_id, compute in enumerate(current.stage_compute_gcycles):
                nodes = np.flatnonzero(env.deployment[current.service_id, stage_id])
                if len(nodes):
                    shares = capacities[nodes] / max(float(capacities[nodes].sum()), 1e-9)
                    sqrt_load[stage_id, nodes] += float(current.request_count) * np.sqrt(max(float(compute), 0.0)) * shares
        self._phase_sqrt_cache_key = key
        self._phase_sqrt_cache = sqrt_load
        self._phase_sqrt_cache_request_ids = {id(current) for current in env.current_requests}
        return sqrt_load

    def _structured_route_metrics(
        self, env: EdgeComputingEnv
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cache routes, their physical-link incidence, and residual capacities."""

        env._require_ready()
        assert env.scenario is not None
        key = (id(env.current_requests), float(env.current_time_minute), float(env.metrics["deployment_updates"]))
        if key == self._structured_route_cache_key and self._structured_route_cache is not None:
            return self._structured_route_cache
        bandwidth = env.scenario.bandwidth_mb_s
        physical = np.isfinite(bandwidth) & (bandwidth > 0.0)
        np.fill_diagonal(physical, False)
        links = np.argwhere(physical)
        hop_index = np.full((self.num_nodes, self.num_nodes), -1, dtype=np.int32)
        hop_index[links[:, 0], links[:, 1]] = np.arange(len(links))
        effective_capacity = bandwidth[links[:, 0], links[:, 1]] * np.maximum(
            1.0 - 0.75 * env.link_load[links[:, 0], links[:, 1]], 0.10
        )
        incidence = np.zeros((self.num_nodes, self.num_nodes, len(links)), dtype=np.float64)
        transfer_per_mb = np.full((self.num_nodes, self.num_nodes), np.inf, dtype=np.float64)
        propagation = np.full_like(transfer_per_mb, np.inf)
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                path = env.shortest_path(src, dst)
                if path is None:
                    continue
                hop_ids = [hop_index[left, right] for left, right in zip(path, path[1:])]
                if any(hop_id < 0 for hop_id in hop_ids):
                    continue
                incidence[src, dst, hop_ids] = 1.0
                transfer_per_mb[src, dst] = sum(
                    1.0 / max(float(effective_capacity[hop_id]), 1e-9) for hop_id in hop_ids
                )
                propagation[src, dst] = sum(
                    float(env.scenario.propagation_ms[left, right]) / 1000.0
                    for left, right in zip(path, path[1:])
                )
        self._structured_route_cache_key = key
        self._structured_route_cache = (transfer_per_mb, propagation, incidence, effective_capacity)
        return self._structured_route_cache

    def _expected_phase_link_sqrt_load(self, env: EdgeComputingEnv) -> np.ndarray:
        """Expected same-phase KKT link load from current requests' feasible routes.

        The expectation only locates competing traffic.  A candidate route's own
        multiplicity-weighted sqrt demand is added exactly on each of its hops.
        """

        env._require_ready()
        assert env.scenario is not None and env.deployment is not None
        key = (id(env.current_requests), float(env.current_time_minute), float(env.metrics["deployment_updates"]))
        if key == self._phase_link_sqrt_cache_key and self._phase_link_sqrt_cache is not None:
            return self._phase_link_sqrt_cache
        _, _, incidence, _ = self._structured_route_metrics(env)
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64)
        sqrt_load = np.zeros((2 * self.max_service_stages - 1, incidence.shape[-1]), dtype=np.float64)
        by_request: dict[int, np.ndarray] = {}
        for current in env.current_requests:
            expected = np.zeros_like(sqrt_load)
            stage_nodes = [
                np.flatnonzero(env.deployment[current.service_id, stage_id])
                for stage_id in range(len(current.stage_compute_gcycles))
            ]
            shares = [
                capacities[nodes] / max(float(capacities[nodes].sum()), 1e-9)
                for nodes in stage_nodes
            ]
            if len(stage_nodes[0]) and current.input_mb > 0.0:
                expected[0] = (
                    float(current.request_count)
                    * np.sqrt(float(current.input_mb))
                    * (shares[0] @ incidence[current.home_node, stage_nodes[0]])
                )
            for stage_id in range(len(stage_nodes) - 1):
                src_nodes, dst_nodes = stage_nodes[stage_id : stage_id + 2]
                data_mb = float(current.stage_output_mb[stage_id])
                if not len(src_nodes) or not len(dst_nodes) or data_mb <= 0.0:
                    continue
                pair_shares = np.outer(shares[stage_id], shares[stage_id + 1]).reshape(-1)
                route_hops = incidence[np.ix_(src_nodes, dst_nodes)].reshape(-1, incidence.shape[-1])
                expected[2 * stage_id + 2] = (
                    float(current.request_count) * np.sqrt(data_mb) * (pair_shares @ route_hops)
                )
            sqrt_load += expected
            by_request[id(current)] = expected
        self._phase_link_sqrt_cache_key = key
        self._phase_link_sqrt_cache = sqrt_load
        self._phase_link_sqrt_by_request = by_request
        return sqrt_load

    def _structured_link_transfer_costs(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        *,
        phase: int,
        data_mb: float,
        reservation_link_sqrt_delta: np.ndarray | None,
    ) -> np.ndarray:
        """Predict per-request route delay using the environment's phase KKT rule."""

        _, _, incidence, effective_capacity = self._structured_route_metrics(env)
        if data_mb <= 0.0:
            return np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        baseline = self._expected_phase_link_sqrt_load(env)[phase]
        own_expected = self._phase_link_sqrt_by_request.get(id(request))
        competing = baseline.copy()
        if own_expected is not None:
            competing -= own_expected[phase]
        if reservation_link_sqrt_delta is not None:
            competing += reservation_link_sqrt_delta[phase]
        own_sqrt = float(request.request_count) * np.sqrt(float(data_mb))
        per_hop_delay = np.sqrt(float(data_mb)) * (
            np.maximum(competing, 0.0) + own_sqrt
        ) / np.maximum(effective_capacity, 1e-9)
        return (incidence.reshape(-1, incidence.shape[-1]) @ per_hop_delay).reshape(
            self.num_nodes, self.num_nodes
        )

    def _structured_path_state(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        *,
        reservation_sqrt_delta: np.ndarray | None = None,
        reservation_link_sqrt_delta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expose all deployed nodes and routable transitions; never preselect paths."""

        env._require_ready()
        assert env.scenario is not None and env.deployment is not None
        stages = self.max_service_stages
        stage_count = len(request.stage_compute_gcycles)
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64)
        max_capacity = max(float(capacities.max()), 1e-9)
        baseline_sqrt = self._expected_phase_sqrt_load(env)
        transfer_per_mb, propagation, _, _ = self._structured_route_metrics(env)
        ingress_transfer = self._structured_link_transfer_costs(
            env,
            request,
            phase=0,
            data_mb=float(request.input_mb),
            reservation_link_sqrt_delta=reservation_link_sqrt_delta,
        )
        node_features = np.zeros((stages, self.num_nodes, FAST_STRUCT_NODE_FEATURE_DIM), dtype=np.float32)
        edge_features = np.zeros(
            (stages - 1, self.num_nodes, self.num_nodes, FAST_STRUCT_EDGE_FEATURE_DIM), dtype=np.float32
        )
        node_mask = np.zeros((stages, self.num_nodes), dtype=bool)
        for stage_id, compute_raw in enumerate(request.stage_compute_gcycles):
            nodes = np.flatnonzero(env.deployment[request.service_id, stage_id])
            if not len(nodes):
                continue
            compute = max(float(compute_raw), 0.0)
            own_sqrt = float(request.request_count) * np.sqrt(compute)
            shares = capacities[nodes] / max(float(capacities[nodes].sum()), 1e-9)
            baseline_without_own = baseline_sqrt[stage_id, nodes].copy()
            if id(request) in self._phase_sqrt_cache_request_ids:
                baseline_without_own -= own_sqrt * shares
            if reservation_sqrt_delta is not None:
                baseline_without_own += reservation_sqrt_delta[stage_id, nodes]
            competing_sqrt = np.maximum(baseline_without_own, 0.0)
            for offset, node_id in enumerate(nodes):
                effective_capacity = capacities[node_id] * max(
                    1.0 - 0.75 * float(env.node_compute_load[node_id]), 0.10
                )
                predicted_compute = np.sqrt(compute) * (competing_sqrt[offset] + own_sqrt) / max(effective_capacity, 1e-9)
                externality = np.sqrt(compute) * competing_sqrt[offset] / max(effective_capacity, 1e-9)
                ingress = 0.0
                if stage_id == 0:
                    per_mb = transfer_per_mb[request.home_node, node_id]
                    if not np.isfinite(per_mb):
                        continue
                    ingress = ingress_transfer[request.home_node, node_id] + propagation[request.home_node, node_id]
                node_mask[stage_id, node_id] = True
                node_features[stage_id, node_id] = [
                    node_id / max(self.num_nodes - 1, 1),
                    stage_id / max(stages - 1, 1),
                    capacities[node_id] / max_capacity,
                    float(env.node_compute_load[node_id]),
                    float(np.log1p(competing_sqrt[offset] / max(effective_capacity, 1e-9))),
                    float(np.log1p(compute / max(effective_capacity, 1e-9))),
                    float(np.log1p(predicted_compute)),
                    float(np.log1p(externality)),
                    float(np.log1p(ingress)),
                ]
        reachable = np.isfinite(transfer_per_mb)
        finite_propagation = np.where(reachable, propagation, 0.0)
        for stage_id in range(stage_count - 1):
            data_mb = float(request.stage_output_mb[stage_id])
            transfer = self._structured_link_transfer_costs(
                env,
                request,
                phase=2 * stage_id + 2,
                data_mb=data_mb,
                reservation_link_sqrt_delta=reservation_link_sqrt_delta,
            )
            total = transfer + finite_propagation
            edge_features[stage_id, ..., 0] = reachable
            edge_features[stage_id, ..., 1] = np.log1p(total)
            edge_features[stage_id, ..., 2] = np.log1p(transfer)
            edge_features[stage_id, ..., 3] = np.log1p(finite_propagation)
            edge_features[stage_id, ..., 4] = np.eye(self.num_nodes, dtype=np.float32)

        possible = node_mask[0].copy()
        for stage_id in range(1, stage_count):
            possible = node_mask[stage_id] & (
                (possible[:, None] & (edge_features[stage_id - 1, ..., 0] > 0.5)).any(axis=0)
            )
        if not possible.any():
            # Keep a defined action so normal environment penalties report an
            # infeasible deployment/topology, rather than producing NaN logits.
            node_mask[:stage_count] = False
            node_mask[:stage_count, request.home_node] = True
            for stage_id in range(stage_count - 1):
                edge_features[stage_id] = 0.0
                edge_features[stage_id, request.home_node, request.home_node, 0] = 1.0
        for stage_id in range(stage_count, stages):
            node_mask[stage_id, 0] = True
            edge_features[stage_id - 1, :, 0, 0] = 1.0

        global_features = np.asarray(
            [
                request.service_id / max(env.config.num_service_types - 1, 1),
                float(request.input_mb) / 3.0,
                float(sum(request.stage_compute_gcycles)) / 20.0,
                float(request.deadline_s),
                float(request.request_count) / 100.0,
                float(env.current_time_minute) / max(env.config.episode_minutes, 1),
                stage_count / max(stages, 1),
                float(np.log1p(len(env.current_requests))) / np.log1p(5_000.0),
                float(np.mean(env.node_compute_load)),
            ],
            dtype=np.float32,
        )
        state = np.concatenate([global_features, node_features.reshape(-1), edge_features.reshape(-1)])
        return state, node_mask.reshape(-1)

    def _decode_structured_path(self, action: int, stage_count: int) -> list[int]:
        return [int(action // (self.num_nodes ** stage_id) % self.num_nodes) for stage_id in range(stage_count)]

    def _kkt_route_features(
        self,
        env: EdgeComputingEnv,
        *,
        phase: int,
        data_mb: float,
        multiplicity: float,
        link_sqrt_ledger: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return exact causal KKT route costs before the current group is reserved."""

        transfer_per_mb, propagation, incidence, effective_capacity = self._structured_route_metrics(env)
        reachable = np.isfinite(transfer_per_mb)
        if data_mb <= 0.0:
            zeros = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
            return zeros, zeros, zeros, zeros, zeros, zeros
        count = max(float(multiplicity), 1e-9)
        sqrt_data = np.sqrt(float(data_mb))
        own_weight = count * sqrt_data
        ledger = np.maximum(link_sqrt_ledger[phase], 0.0)
        capacity = np.maximum(effective_capacity, 1e-9)
        flat_incidence = incidence.reshape(-1, incidence.shape[-1])
        own_delay = (flat_incidence @ (sqrt_data * (ledger + own_weight) / capacity)).reshape(
            self.num_nodes, self.num_nodes
        )
        marginal_mean = (
            flat_incidence @ ((2.0 * ledger * own_weight + own_weight**2) / (capacity * count))
        ).reshape(self.num_nodes, self.num_nodes)
        ledger_pressure = (flat_incidence @ (ledger / capacity)).reshape(self.num_nodes, self.num_nodes)
        hop_count = flat_incidence.sum(axis=1).reshape(self.num_nodes, self.num_nodes)
        mean_effective_capacity = (
            (flat_incidence @ effective_capacity).reshape(self.num_nodes, self.num_nodes)
            / np.maximum(hop_count, 1.0)
        )
        assert env.scenario is not None
        physical = np.isfinite(env.scenario.bandwidth_mb_s) & (env.scenario.bandwidth_mb_s > 0.0)
        np.fill_diagonal(physical, False)
        physical_links = np.argwhere(physical)
        raw_link_capacity = env.scenario.bandwidth_mb_s[physical_links[:, 0], physical_links[:, 1]]
        link_background = env.link_load[physical_links[:, 0], physical_links[:, 1]]
        mean_raw_capacity = (
            (flat_incidence @ raw_link_capacity).reshape(self.num_nodes, self.num_nodes)
            / np.maximum(hop_count, 1.0)
        )
        mean_background_load = (
            (flat_incidence @ link_background).reshape(self.num_nodes, self.num_nodes)
            / np.maximum(hop_count, 1.0)
        )
        own_delay = np.where(reachable, own_delay, np.inf)
        marginal_mean = np.where(reachable, marginal_mean, np.inf)
        return (
            own_delay,
            marginal_mean,
            ledger_pressure,
            mean_raw_capacity,
            mean_effective_capacity,
            mean_background_load,
        )

    def _kkt_autoregressive_path_state(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        *,
        remaining_requests: list[TaskRequest],
        batch_size: int,
        decision_index: int,
        compute_sqrt_ledger: np.ndarray,
        link_sqrt_ledger: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a full-path state from only already committed batch decisions."""

        env._require_ready()
        assert env.scenario is not None and env.deployment is not None
        stages = self.max_service_stages
        stage_count = len(request.stage_compute_gcycles)
        count = max(float(request.request_count), 1e-9)
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64)
        effective_compute = capacities * np.maximum(1.0 - 0.75 * env.node_compute_load, 0.10)
        max_capacity = max(float(capacities.max()), 1e-9)
        node_features = np.zeros((stages, self.num_nodes, FAST_KKT_NODE_FEATURE_DIM), dtype=np.float32)
        edge_features = np.zeros(
            (stages - 1, self.num_nodes, self.num_nodes, FAST_KKT_EDGE_FEATURE_DIM), dtype=np.float32
        )
        node_mask = np.zeros((stages, self.num_nodes), dtype=bool)
        ingress = self._kkt_route_features(
            env,
            phase=0,
            data_mb=float(request.input_mb),
            multiplicity=count,
            link_sqrt_ledger=link_sqrt_ledger,
        )
        _, ingress_marginal, _, _, _, _ = ingress
        _, propagation, _, _ = self._structured_route_metrics(env)
        for stage_id, compute_raw in enumerate(request.stage_compute_gcycles):
            deployed = np.flatnonzero(env.deployment[request.service_id, stage_id])
            compute = max(float(compute_raw), 0.0)
            sqrt_compute = np.sqrt(compute)
            own_weight = count * sqrt_compute
            for node_id in deployed:
                capacity = max(float(effective_compute[node_id]), 1e-9)
                ledger = max(float(compute_sqrt_ledger[stage_id, node_id]), 0.0)
                own_isolated = compute / capacity
                own_delay = sqrt_compute * (ledger + own_weight) / capacity
                marginal_mean = (2.0 * ledger * own_weight + own_weight**2) / (capacity * count)
                externality = sqrt_compute * ledger / capacity
                ingress_cost = 0.0
                if stage_id == 0:
                    if not np.isfinite(ingress_marginal[request.home_node, node_id]):
                        continue
                    ingress_cost = float(
                        ingress_marginal[request.home_node, node_id]
                        + propagation[request.home_node, node_id]
                    )
                node_mask[stage_id, node_id] = True
                node_features[stage_id, node_id] = [
                    node_id / max(self.num_nodes - 1, 1),
                    stage_id / max(stages - 1, 1),
                    capacities[node_id] / max_capacity,
                    float(env.node_compute_load[node_id]),
                    effective_compute[node_id] / max_capacity,
                    float(np.log1p(ledger / capacity)),
                    float(np.log1p(own_isolated)),
                    float(np.log1p(own_delay)),
                    float(np.log1p(marginal_mean)),
                    float(np.log1p(externality)),
                    float(np.log1p(ingress_cost)),
                ]

        transfer_per_mb, propagation, _, _ = self._structured_route_metrics(env)
        reachable = np.isfinite(transfer_per_mb)
        finite_raw_bandwidth = env.scenario.bandwidth_mb_s[
            np.isfinite(env.scenario.bandwidth_mb_s) & (env.scenario.bandwidth_mb_s > 0.0)
        ]
        max_link_capacity = max(
            float(finite_raw_bandwidth.max()) if finite_raw_bandwidth.size else 1.0,
            1e-9,
        )
        for stage_id in range(stage_count - 1):
            own_delay, marginal, pressure, raw_capacity, effective_capacity, background = self._kkt_route_features(
                env,
                phase=2 * stage_id + 2,
                data_mb=float(request.stage_output_mb[stage_id]),
                multiplicity=count,
                link_sqrt_ledger=link_sqrt_ledger,
            )
            total_own = own_delay + np.where(reachable, propagation, 0.0)
            total_marginal = marginal + np.where(reachable, propagation, 0.0)
            edge_features[stage_id, ..., 0] = reachable
            edge_features[stage_id, ..., 1] = np.log1p(np.where(reachable, total_own, 0.0))
            edge_features[stage_id, ..., 2] = np.log1p(np.where(reachable, total_marginal, 0.0))
            edge_features[stage_id, ..., 3] = np.log1p(np.where(reachable, pressure, 0.0))
            edge_features[stage_id, ..., 4] = np.where(reachable, raw_capacity / max_link_capacity, 0.0)
            edge_features[stage_id, ..., 5] = np.where(reachable, effective_capacity / max_link_capacity, 0.0)
            edge_features[stage_id, ..., 6] = np.where(reachable, background, 0.0)
            edge_features[stage_id, ..., 7] = np.where(reachable, np.log1p(propagation), 0.0)
            edge_features[stage_id, ..., 8] = np.eye(self.num_nodes, dtype=np.float32)

        possible = node_mask[0].copy()
        for stage_id in range(1, stage_count):
            possible = node_mask[stage_id] & (
                (possible[:, None] & (edge_features[stage_id - 1, ..., 0] > 0.5)).any(axis=0)
            )
        if not possible.any():
            node_mask[:stage_count] = False
            node_mask[:stage_count, request.home_node] = True
            for stage_id in range(stage_count - 1):
                edge_features[stage_id] = 0.0
                edge_features[stage_id, request.home_node, request.home_node, 0] = 1.0
                edge_features[stage_id, request.home_node, request.home_node, 8] = 1.0
        for stage_id in range(stage_count, stages):
            node_mask[stage_id, 0] = True
            edge_features[stage_id - 1, :, 0, 0] = 1.0

        remaining_count = float(sum(item.request_count for item in remaining_requests))
        remaining_compute = np.asarray(
            [sum(item.stage_compute_gcycles) for item in remaining_requests], dtype=np.float64
        )
        remaining_input = np.asarray([item.input_mb for item in remaining_requests], dtype=np.float64)
        remaining_output = np.asarray(
            [sum(item.stage_output_mb) for item in remaining_requests], dtype=np.float64
        )
        remaining_deadline = np.asarray([item.deadline_s for item in remaining_requests], dtype=np.float64)
        service_diversity = len({item.service_id for item in remaining_requests})
        global_features = np.asarray(
            [
                request.service_id / max(env.config.num_service_types - 1, 1),
                float(request.input_mb) / 3.0,
                float(sum(request.stage_compute_gcycles)) / 20.0,
                float(request.deadline_s),
                count / 100.0,
                float(env.current_time_minute) / max(env.config.episode_minutes, 1),
                stage_count / max(stages, 1),
                float(np.log1p(batch_size)) / np.log1p(5_000.0),
                float(np.mean(env.node_compute_load)),
                len(remaining_requests) / max(batch_size, 1),
                float(np.log1p(remaining_count)) / np.log1p(5_000.0),
                float(remaining_compute.mean()) / 20.0 if remaining_compute.size else 0.0,
                float(remaining_compute.std()) / 20.0 if remaining_compute.size else 0.0,
                float(remaining_input.mean()) / 3.0 if remaining_input.size else 0.0,
                float(remaining_output.mean()) / 3.0 if remaining_output.size else 0.0,
                float(remaining_deadline.mean()) if remaining_deadline.size else 0.0,
                service_diversity / max(env.config.num_service_types, 1),
                decision_index / max(batch_size - 1, 1),
            ],
            dtype=np.float32,
        )
        state = np.concatenate([global_features, node_features.reshape(-1), edge_features.reshape(-1)])
        return state, node_mask.reshape(-1)

    def _reserve_kkt_path_and_measure_credit(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        path: list[int],
        compute_sqrt_ledger: np.ndarray,
        link_sqrt_ledger: np.ndarray,
    ) -> float:
        """Reserve one selected path and return its exact total weighted KKT increment."""

        assert env.scenario is not None
        count = max(float(request.request_count), 1e-9)
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64)
        effective_compute = capacities * np.maximum(1.0 - 0.75 * env.node_compute_load, 0.10)
        _, propagation, incidence, effective_link = self._structured_route_metrics(env)
        increment = 0.0
        for stage_id, node_id in enumerate(path):
            sqrt_compute = np.sqrt(max(float(request.stage_compute_gcycles[stage_id]), 0.0))
            own_weight = count * sqrt_compute
            ledger = max(float(compute_sqrt_ledger[stage_id, node_id]), 0.0)
            increment += (2.0 * ledger * own_weight + own_weight**2) / max(
                float(effective_compute[node_id]), 1e-9
            )
            compute_sqrt_ledger[stage_id, node_id] += own_weight

        transfers = [(0, request.home_node, path[0], float(request.input_mb))]
        transfers.extend(
            (2 * stage_id + 2, path[stage_id], path[stage_id + 1], float(request.stage_output_mb[stage_id]))
            for stage_id in range(len(path) - 1)
        )
        for phase, src, dst, data_mb in transfers:
            if data_mb <= 0.0 or src == dst:
                continue
            route_incidence = incidence[src, dst]
            if not route_incidence.any():
                increment += count * float(env.config.invalid_action_penalty)
                continue
            sqrt_data = np.sqrt(data_mb)
            own_weight = count * sqrt_data
            ledger = np.maximum(link_sqrt_ledger[phase], 0.0)
            capacity = np.maximum(effective_link, 1e-9)
            increment += float(
                np.sum(route_incidence * (2.0 * ledger * own_weight + own_weight**2) / capacity)
            )
            increment += count * float(propagation[src, dst])
            link_sqrt_ledger[phase] += own_weight * route_incidence
        return float(increment)

    def _kkt_proxy_regret(self, state: np.ndarray, mask: np.ndarray, path: list[int]) -> float:
        stages = self.max_service_stages
        count = len(path)
        node_end = FAST_KKT_GLOBAL_DIM + stages * self.num_nodes * FAST_KKT_NODE_FEATURE_DIM
        node_features = state[FAST_KKT_GLOBAL_DIM:node_end].reshape(
            stages, self.num_nodes, FAST_KKT_NODE_FEATURE_DIM
        )
        edge_features = state[node_end:].reshape(
            stages - 1, self.num_nodes, self.num_nodes, FAST_KKT_EDGE_FEATURE_DIM
        )
        node_mask = mask.reshape(stages, self.num_nodes)
        node_cost = np.expm1(node_features[:count, :, 8]).astype(np.float64)
        node_cost[0] += np.expm1(node_features[0, :, 10])
        best = np.where(node_mask[0], node_cost[0], np.inf)
        selected = float(node_cost[0, path[0]])
        for stage_id in range(1, count):
            route = np.expm1(edge_features[stage_id - 1, ..., 2])
            route = np.where(edge_features[stage_id - 1, ..., 0] > 0.5, route, np.inf)
            best = np.where(node_mask[stage_id], node_cost[stage_id] + np.min(best[:, None] + route, axis=0), np.inf)
            selected += float(node_cost[stage_id, path[stage_id]] + route[path[stage_id - 1], path[stage_id]])
        optimum = float(np.min(best))
        return max(selected - optimum, 0.0) if np.isfinite(selected + optimum) else 0.0

    def _schedule_kkt_autoregressive_batch(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        *,
        deterministic: bool,
        record: bool,
    ) -> list[list[int]]:
        """Select complete paths sequentially and expose only committed KKT load."""

        _, _, incidence, _ = self._structured_route_metrics(env)
        compute_ledger = np.zeros((self.max_service_stages, self.num_nodes), dtype=np.float64)
        link_ledger = np.zeros((2 * self.max_service_stages - 1, incidence.shape[-1]), dtype=np.float64)
        total_count = max(float(sum(request.request_count for request in requests)), 1.0)
        schedules: list[list[int]] = []
        credits: list[float] = []
        counts: list[float] = []
        self._path_selected_proxy_regrets.clear()
        self._pending_kkt_batch_credits.clear()
        self._pending_kkt_batch_counts.clear()
        self._pending_kkt_batch_recorded = bool(record)
        for request_index, request in enumerate(requests):
            state, mask = self._kkt_autoregressive_path_state(
                env,
                request,
                remaining_requests=requests[request_index:],
                batch_size=len(requests),
                decision_index=request_index,
                compute_sqrt_ledger=compute_ledger,
                link_sqrt_ledger=link_ledger,
            )
            action, logprob, value = self.ppo.act(state, mask, deterministic=deterministic)
            path = self._decode_structured_path(action, len(request.stage_compute_gcycles))
            schedules.append(path)
            self._path_selected_proxy_regrets.setdefault(id(request), []).append(
                self._kkt_proxy_regret(state, mask, path)
            )
            if record:
                self.ppo.buffer.states.append(state)
                self.ppo.buffer.masks.append(mask)
                self.ppo.buffer.actions.append(action)
                self.ppo.buffer.logprobs.append(logprob)
                self.ppo.buffer.values.append(value)
                load_group = env.config.demand_load_group
                self.ppo.buffer.sample_groups.append(
                    float(env.config.demand_load_multiplier if load_group is None else load_group)
                )
            increment = self._reserve_kkt_path_and_measure_credit(
                env,
                request,
                path,
                compute_ledger,
                link_ledger,
            )
            credits.append(increment / total_count)
            counts.append(float(request.request_count))
        if record:
            self._pending_kkt_batch_credits.extend(credits)
            self._pending_kkt_batch_counts.extend(counts)
        return schedules

    def finalize_kkt_autoregressive_batch_credit(
        self,
        group_infos: list[dict[str, object]],
        *,
        reward_scale: float,
    ) -> bool:
        """Close one Fast subtrajectory against the environment's authoritative settlement."""

        if self.policy_kind != "kkt_autoregressive_path" or not self._pending_kkt_batch_recorded:
            return False
        if len(group_infos) != len(self._pending_kkt_batch_credits):
            raise ValueError("KKT Fast credits do not match the settled request groups")
        total_count = max(float(sum(self._pending_kkt_batch_counts)), 1.0)
        actual_mean = sum(
            count
            * (
                float(info["compute_delay_s"])
                + float(info["link_delay_s"])
                + float(info["propagation_delay_s"])
                + float(info["penalty_latency_s"])
            )
            for count, info in zip(self._pending_kkt_batch_counts, group_infos)
        ) / total_count
        predicted_mean = float(sum(self._pending_kkt_batch_credits))
        correction = float(actual_mean - predicted_mean)
        corrected = list(self._pending_kkt_batch_credits)
        if corrected:
            corrected[-1] += correction
        for index, credit in enumerate(corrected):
            self.ppo.buffer.rewards.append(-float(credit) * float(reward_scale))
            self.ppo.buffer.dones.append(index == len(corrected) - 1)
            self.ppo.buffer.weights.append(1.0)
        self.last_kkt_credit_diagnostics = {
            "fast_kkt_predicted_mean_cost_s": predicted_mean,
            "fast_kkt_actual_mean_cost_s": float(actual_mean),
            "fast_kkt_terminal_correction_s": correction,
            "fast_kkt_credit_abs_error_s": abs(correction),
            "fast_kkt_batch_groups": float(len(corrected)),
            "fast_kkt_batch_requests": total_count,
        }
        self._pending_kkt_batch_credits.clear()
        self._pending_kkt_batch_counts.clear()
        self._pending_kkt_batch_recorded = False
        return True

    def _structured_proxy_regret(self, state: np.ndarray, mask: np.ndarray, path: list[int]) -> float:
        stages = self.max_service_stages
        count = len(path)
        node_end = FAST_PATH_GLOBAL_DIM + stages * self.num_nodes * FAST_STRUCT_NODE_FEATURE_DIM
        node_features = state[FAST_PATH_GLOBAL_DIM:node_end].reshape(stages, self.num_nodes, FAST_STRUCT_NODE_FEATURE_DIM)
        edge_features = state[node_end:].reshape(
            stages - 1, self.num_nodes, self.num_nodes, FAST_STRUCT_EDGE_FEATURE_DIM
        )
        node_mask = mask.reshape(stages, self.num_nodes)
        node_cost = np.expm1(node_features[:count, :, 6]).astype(np.float64)
        node_cost[0] += np.expm1(node_features[0, :, 8])
        best = np.where(node_mask[0], node_cost[0], np.inf)
        selected = float(node_cost[0, path[0]])
        for stage_id in range(1, count):
            route = np.expm1(edge_features[stage_id - 1, ..., 1])
            route = np.where(edge_features[stage_id - 1, ..., 0] > 0.5, route, np.inf)
            best = np.where(node_mask[stage_id], node_cost[stage_id] + np.min(best[:, None] + route, axis=0), np.inf)
            selected += float(node_cost[stage_id, path[stage_id]] + route[path[stage_id - 1], path[stage_id]])
        optimum = float(np.min(best))
        return max(selected - optimum, 0.0) if np.isfinite(selected + optimum) else 0.0

    def _schedule_structured_batch(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        *,
        deterministic: bool,
        record: bool,
    ) -> list[list[int]]:
        """Sample one masked complete path per request without a path slate."""

        assert env.scenario is not None and env.deployment is not None
        self._path_selected_proxy_regrets.clear()
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64)
        reservation_sqrt_delta = np.zeros((self.max_service_stages, self.num_nodes), dtype=np.float64)
        _, _, route_incidence, _ = self._structured_route_metrics(env)
        reservation_link_sqrt_delta = np.zeros(
            (2 * self.max_service_stages - 1, route_incidence.shape[-1]), dtype=np.float64
        )
        schedules: list[list[int]] = []
        for chunk_start in range(0, len(requests), self.reservation_microbatch_size):
            chunk = requests[chunk_start : chunk_start + self.reservation_microbatch_size]
            observations = [
                self._structured_path_state(
                    env,
                    request,
                    reservation_sqrt_delta=reservation_sqrt_delta,
                    reservation_link_sqrt_delta=reservation_link_sqrt_delta,
                )
                for request in chunk
            ]
            actions, logprobs, values = self.ppo.act_batch(
                [state for state, _ in observations],
                [mask for _, mask in observations],
                deterministic=deterministic,
            )
            for local_idx, request in enumerate(chunk):
                state, mask = observations[local_idx]
                action = int(actions[local_idx])
                path = self._decode_structured_path(action, len(request.stage_compute_gcycles))
                schedules.append(path)
                self._path_selected_proxy_regrets.setdefault(id(request), []).append(
                    self._structured_proxy_regret(state, mask, path)
                )
                if record:
                    self.ppo.buffer.states.append(state)
                    self.ppo.buffer.masks.append(mask)
                    self.ppo.buffer.actions.append(action)
                    self.ppo.buffer.logprobs.append(float(logprobs[local_idx]))
                    self.ppo.buffer.values.append(float(values[local_idx]))
                    load_group = env.config.demand_load_group
                    self.ppo.buffer.sample_groups.append(
                        float(env.config.demand_load_multiplier if load_group is None else load_group)
                    )
            for local_idx, request in enumerate(chunk):
                path = schedules[chunk_start + local_idx]
                own_expected_links = self._phase_link_sqrt_by_request.get(id(request))
                if own_expected_links is not None:
                    reservation_link_sqrt_delta -= own_expected_links
                if request.input_mb > 0.0:
                    reservation_link_sqrt_delta[0] += (
                        float(request.request_count)
                        * np.sqrt(float(request.input_mb))
                        * route_incidence[request.home_node, path[0]]
                    )
                for stage_id in range(len(path) - 1):
                    data_mb = float(request.stage_output_mb[stage_id])
                    if data_mb > 0.0:
                        reservation_link_sqrt_delta[2 * stage_id + 2] += (
                            float(request.request_count)
                            * np.sqrt(data_mb)
                            * route_incidence[path[stage_id], path[stage_id + 1]]
                        )
                for stage_id, node_id in enumerate(path):
                    nodes = np.flatnonzero(env.deployment[request.service_id, stage_id])
                    if not len(nodes):
                        continue
                    own_sqrt = float(request.request_count) * np.sqrt(max(float(request.stage_compute_gcycles[stage_id]), 0.0))
                    shares = capacities[nodes] / max(float(capacities[nodes].sum()), 1e-9)
                    if id(request) in self._phase_sqrt_cache_request_ids:
                        reservation_sqrt_delta[stage_id, nodes] -= own_sqrt * shares
                    reservation_sqrt_delta[stage_id, node_id] += own_sqrt
        return schedules

    def _schedule_paths_batch(
        self,
        env: EdgeComputingEnv,
        requests: list[TaskRequest],
        *,
        deterministic: bool,
        record: bool,
    ) -> list[list[int]]:
        """Choose one complete path per request, then reserve its stage work."""

        schedules: list[list[int]] = []
        self._path_selected_proxy_regrets.clear()
        reservation_node_delta = np.zeros(self.num_nodes, dtype=np.float64)
        reservation_stage_delta: dict[tuple[int, int], np.ndarray] = {}
        for chunk_start in range(0, len(requests), self.reservation_microbatch_size):
            chunk = requests[chunk_start : chunk_start + self.reservation_microbatch_size]
            candidate_sets = [
                self._path_candidates(
                    env,
                    request,
                    reservation_stage_delta=reservation_stage_delta,
                )
                for request in chunk
            ]
            actions, logprobs, values = self.ppo.act_batch(
                [item[1] for item in candidate_sets],
                [item[2] for item in candidate_sets],
                deterministic=deterministic,
            )
            for local_idx, request in enumerate(chunk):
                paths, state, mask = candidate_sets[local_idx]
                action = int(actions[local_idx])
                schedules.append(list(paths[action]))
                feature_dim = fast_path_feature_dim(self.max_service_stages)
                best_cost = float(np.expm1(state[FAST_PATH_GLOBAL_DIM]))
                chosen_cost = float(np.expm1(state[FAST_PATH_GLOBAL_DIM + action * feature_dim]))
                self._path_selected_proxy_regrets.setdefault(id(request), []).append(
                    max(chosen_cost - best_cost, 0.0)
                )
                if record:
                    self.ppo.buffer.states.append(state)
                    self.ppo.buffer.masks.append(mask)
                    self.ppo.buffer.actions.append(action)
                    self.ppo.buffer.logprobs.append(float(logprobs[local_idx]))
                    self.ppo.buffer.values.append(float(values[local_idx]))
                    load_group = env.config.demand_load_group
                    self.ppo.buffer.sample_groups.append(
                        float(env.config.demand_load_multiplier if load_group is None else load_group)
                    )
            for local_idx, request in enumerate(chunk):
                for stage_id, node_id in enumerate(schedules[chunk_start + local_idx]):
                    self._reserve_compute_work(
                        env,
                        request,
                        stage_id,
                        node_id,
                        reservation_node_delta,
                        reservation_stage_delta,
                    )
        return schedules

    def _path_candidates(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        *,
        reservation_stage_delta: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray]:
        """Find the best K deployed, routed paths under a causal cost proxy.

        The proxy is used only to construct a bounded candidate set. The PPO
        actor scores every retained path and may choose any of them. Padding
        slots are masked, so the action is a path slot rather than a node id.
        """

        env._require_ready()
        assert env.scenario is not None and env.deployment is not None
        stage_count = len(request.stage_compute_gcycles)
        stage_masks = env.scheduler_candidate_mask(request)
        stage_nodes = [np.flatnonzero(stage_masks[stage_id]).tolist() for stage_id in range(stage_count)]
        capacities = np.asarray([node.compute_gcycles_per_s for node in env.scenario.nodes], dtype=np.float64)
        stage_costs = np.full((stage_count, self.num_nodes), np.inf, dtype=np.float64)
        stage_pressures = np.zeros((stage_count, self.num_nodes), dtype=np.float64)
        _, expected_stage_work = self._expected_current_compute_workload(env)
        for stage_id, nodes in enumerate(stage_nodes):
            if not nodes:
                continue
            key = (request.service_id, stage_id)
            # KKT settlement separates serial stages, but all services at the
            # same stage phase compete for the same node capacity.
            competing_work = sum(
                (work for (_, phase), work in expected_stage_work.items() if phase == stage_id),
                np.zeros(self.num_nodes, dtype=np.float64),
            )
            if reservation_stage_delta is not None:
                competing_work += sum(
                    (work for (_, phase), work in reservation_stage_delta.items() if phase == stage_id),
                    np.zeros(self.num_nodes, dtype=np.float64),
                )
            own_work = float(request.stage_compute_gcycles[stage_id]) * float(request.request_count)
            shares = capacities[nodes] / max(float(capacities[nodes].sum()), 1e-9)
            competing_work[nodes] = np.maximum(competing_work[nodes] - own_work * shares, 0.0)
            for node_id in nodes:
                effective_capacity = capacities[node_id] * max(
                    1.0 - 0.75 * float(env.node_compute_load[node_id]), 0.10
                )
                pressure = float(competing_work[node_id]) / max(effective_capacity, 1e-9)
                stage_pressures[stage_id, node_id] = pressure
                stage_costs[stage_id, node_id] = (
                    float(request.stage_compute_gcycles[stage_id])
                    / max(effective_capacity, 1e-9)
                    * (1.0 + np.log1p(pressure))
                )

        route_cache: dict[tuple[int, int, int], float] = {}

        def route_cost(stage_id: int, src_node: int, dst_node: int) -> float:
            key = (stage_id, src_node, dst_node)
            if key not in route_cache:
                data_mb = (
                    float(request.input_mb)
                    if stage_id == 0
                    else float(request.stage_output_mb[stage_id - 1])
                )
                transfer_s, propagation_s = self._route_proxy_delay(env, src_node, dst_node, data_mb)
                route_cache[key] = transfer_s + propagation_s
            return route_cache[key]

        # Exact additive lower bounds let the bounded beam retain the K best
        # complete paths under this proxy without enumerating every product.
        future = np.full((stage_count, self.num_nodes), np.inf, dtype=np.float64)
        if stage_count:
            future[stage_count - 1, stage_nodes[-1]] = 0.0
        for stage_id in range(stage_count - 2, -1, -1):
            for node_id in stage_nodes[stage_id]:
                future[stage_id, node_id] = min(
                    (
                        route_cost(stage_id + 1, node_id, next_node)
                        + stage_costs[stage_id + 1, next_node]
                        + future[stage_id + 1, next_node]
                        for next_node in stage_nodes[stage_id + 1]
                    ),
                    default=float("inf"),
                )
        beam: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, ())]
        for stage_id, nodes in enumerate(stage_nodes):
            expanded: list[tuple[float, float, tuple[int, ...]]] = []
            for _, prefix_cost, prefix in beam:
                previous = request.home_node if not prefix else prefix[-1]
                for node_id in nodes:
                    cost = prefix_cost + route_cost(stage_id, previous, node_id) + stage_costs[stage_id, node_id]
                    bound = cost + future[stage_id, node_id]
                    if np.isfinite(bound):
                        expanded.append((bound, cost, prefix + (node_id,)))
            expanded.sort(key=lambda item: (item[0], item[2]))
            beam = expanded[: self.path_candidate_limit]
            if not beam:
                break

        if beam and len(beam[0][2]) == stage_count:
            # Keep some start-node diversity: otherwise a top-K proxy list can
            # contain only variants of one ingress node and permanently hide
            # another deployed replica from PPO exploration.
            anchors: list[tuple[float, tuple[int, ...]]] = []
            for first_node in stage_nodes[0]:
                total = route_cost(0, request.home_node, first_node) + stage_costs[0, first_node]
                if not np.isfinite(total + future[0, first_node]):
                    continue
                path = (first_node,)
                for stage_id in range(1, stage_count):
                    previous = path[-1]
                    next_node = min(
                        stage_nodes[stage_id],
                        key=lambda node_id: (
                            route_cost(stage_id, previous, node_id)
                            + stage_costs[stage_id, node_id]
                            + future[stage_id, node_id],
                            node_id,
                        ),
                    )
                    total += route_cost(stage_id, previous, next_node) + stage_costs[stage_id, next_node]
                    path += (next_node,)
                anchors.append((float(total), path))
            anchors.sort(key=lambda item: (item[0], item[1]))
            diversity_slots = min(len(anchors), self.path_candidate_limit, max(2, self.path_candidate_limit // 4))
            retained = anchors[:diversity_slots]
            seen = {path for _, path in retained}
            for _, cost, path in beam:
                if len(retained) >= self.path_candidate_limit:
                    break
                if path not in seen:
                    retained.append((float(cost), path))
                    seen.add(path)
            retained.sort(key=lambda item: (item[0], item[1]))
            paths = [path for _, path in retained]
            best_cost = retained[0][0]
        else:
            # A broken deployment or disconnected graph has no valid path.
            # Preserve the environment's invalid-action penalty and keep PPO's
            # categorical distribution well-defined with one fallback slot.
            paths = [tuple(nodes[0] if nodes else request.home_node for nodes in stage_nodes)]
            best_cost = 0.0

        global_features = np.asarray(
            [
                request.service_id / max(env.config.num_service_types - 1, 1),
                float(request.input_mb) / 3.0,
                float(sum(request.stage_compute_gcycles)) / 20.0,
                float(request.deadline_s),
                float(request.request_count) / 100.0,
                float(env.current_time_minute) / max(env.config.episode_minutes, 1),
                stage_count / max(self.max_service_stages, 1),
                float(np.log1p(len(env.current_requests))) / np.log1p(5_000.0),
                float(np.mean(env.node_compute_load)),
            ],
            dtype=np.float32,
        )
        feature_dim = fast_path_feature_dim(self.max_service_stages)
        features = np.zeros((self.path_candidate_limit, feature_dim), dtype=np.float32)
        mask = np.zeros(self.path_candidate_limit, dtype=bool)
        for path_idx, path in enumerate(paths):
            mask[path_idx] = True
            compute_total = 0.0
            route_total = 0.0
            max_pressure = 0.0
            per_stage: list[float] = []
            previous = request.home_node
            for stage_id, node_id in enumerate(path):
                compute_s = float(stage_costs[stage_id, node_id])
                route_s = route_cost(stage_id, previous, node_id)
                if not np.isfinite(compute_s + route_s):
                    compute_s, route_s = 10.0, 10.0
                compute_total += compute_s
                route_total += route_s
                max_pressure = max(max_pressure, float(stage_pressures[stage_id, node_id]))
                per_stage.extend(
                    [
                        node_id / max(self.num_nodes - 1, 1),
                        capacities[node_id] / max(float(capacities.max()), 1e-9),
                        float(env.node_compute_load[node_id]),
                        float(np.log1p(compute_s)),
                        float(np.log1p(route_s)),
                    ]
                )
                previous = node_id
            total = compute_total + route_total
            cross_count = sum(left != right for left, right in zip(path, path[1:]))
            summary = [
                float(np.log1p(total)),
                float(np.log1p(compute_total)),
                float(np.log1p(route_total)),
                float(np.log1p(max(total - best_cost, 0.0))),
                cross_count / max(stage_count - 1, 1),
                max_pressure / (1.0 + max_pressure),
            ]
            features[path_idx, : len(summary) + len(per_stage)] = summary + per_stage
        state = np.concatenate([global_features, features.reshape(-1)])
        return paths, state, mask

    def schedule_with_diagnostics(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest | None = None,
    ) -> tuple[list[int], list[dict[str, float | int]]]:
        env._require_ready()
        if request is None:
            assert env.current_request is not None
            request = env.current_request

        if self.policy_kind == "structured_path":
            state, mask = self._structured_path_state(env, request)
            stats = self.ppo.action_stats(state, mask)
            return self._decode_structured_path(int(stats["action"]), len(request.stage_compute_gcycles)), [stats]

        if self.policy_kind == "path_scorer":
            candidates, state, mask = self._path_candidates(env, request)
            stats = self.ppo.action_stats(state, mask)
            return list(candidates[int(stats["action"])]), [stats]

        stage_nodes: list[int] = []
        diagnostics: list[dict[str, float | int]] = []
        for stage_id in range(len(request.stage_compute_gcycles)):
            state = self._build_state(env, request, stage_id, stage_nodes)
            mask = self._build_mask(env, request, stage_id, stage_nodes)
            stats = self.ppo.action_stats(state, mask)
            action = int(stats["action"])
            if not mask[action]:
                action = int(np.where(mask)[0][0])
                stats["action"] = action
            stage_nodes.append(action)
            diagnostics.append(stats)
        return stage_nodes, diagnostics

    def assign_schedule_rewards(self, rewards: list[float], done: bool, weight: float = 1.0) -> None:
        """Attach one additive latency reward to each stage action."""

        if self.policy_kind in {"path_scorer", "structured_path", "kkt_autoregressive_path"}:
            self.ppo.buffer.rewards.append(float(sum(rewards)))
            self.ppo.buffer.dones.append(True)
            self.ppo.buffer.weights.append(float(weight))
            return

        for stage_idx, reward in enumerate(rewards):
            self.ppo.buffer.rewards.append(float(reward))
            self.ppo.buffer.dones.append(bool(done or stage_idx == len(rewards) - 1))
            self.ppo.buffer.weights.append(float(weight))

    def assign_last_schedule_reward(self, reward: float, stage_count: int, done: bool, weight: float = 1.0) -> None:
        """Compatibility wrapper using a terminal-only request reward."""

        if stage_count <= 0:
            return
        rewards = [0.0] * stage_count
        rewards[-1] = float(reward)
        self.assign_schedule_rewards(rewards, done=done, weight=weight)

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        entropy_coef = self.entropy_current_coef
        self.ppo.entropy_coef = entropy_coef
        has_samples = len(self.ppo.buffer) > 0
        metrics = self.ppo.update(
            progress_label=progress_label,
            progress_interval_seconds=progress_interval_seconds,
        )
        observed_entropy = float(metrics.get("entropy", 0.0))
        observed_control_entropy = float(
            metrics.get("normalized_entropy", observed_entropy)
            if self.entropy_normalized_control
            else observed_entropy
        )
        if not has_samples:
            next_coef = entropy_coef
        elif self.entropy_target is None or not np.isfinite(observed_control_entropy):
            next_coef = self.entropy_coef
        else:
            next_coef = entropy_coef + self.entropy_adaptation_rate * (
                float(self.entropy_target) - observed_control_entropy
            )
            next_coef = float(np.clip(next_coef, self.entropy_coef, self.entropy_max_coef))
        self.entropy_current_coef = float(next_coef)
        self.ppo.entropy_coef = self.entropy_current_coef
        metrics["entropy_coef"] = float(entropy_coef)
        metrics["entropy_next_coef"] = self.entropy_current_coef
        metrics["entropy_target"] = float(self.entropy_target) if self.entropy_target is not None else 0.0
        metrics["entropy_control_value"] = observed_control_entropy
        metrics["entropy_normalized_control"] = float(self.entropy_normalized_control)
        metrics["entropy_shortfall"] = (
            max(float(self.entropy_target or 0.0) - observed_control_entropy, 0.0)
            if has_samples
            else 0.0
        )
        metrics["entropy_saturated"] = float(
            np.isclose(self.entropy_current_coef, self.entropy_max_coef)
        )
        return metrics

    def _build_state(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        partial_nodes: list[int],
        *,
        reservation_node_delta: np.ndarray | None = None,
        reservation_stage_delta: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> np.ndarray:
        assert env.scenario is not None
        prev_node = request.home_node if not partial_nodes else partial_nodes[-1]
        nodes = env.scenario.nodes
        max_compute = max(n.compute_gcycles_per_s for n in nodes)
        max_bandwidth = np.nanmax(np.where(np.isfinite(env.scenario.bandwidth_mb_s), env.scenario.bandwidth_mb_s, 0.0))
        tick_request_count = float(sum(item.request_count for item in env.current_requests))
        tick_request_event_count = float(len(env.current_requests))
        tick_service_count = float(
            sum(item.request_count for item in env.current_requests if item.service_id == request.service_id)
        )
        tick_node_counts = np.zeros(self.num_nodes, dtype=np.float64)
        for item in env.current_requests:
            tick_node_counts[item.home_node] += float(item.request_count)
        expected_node_work, expected_stage_work = self._expected_current_compute_workload(env)
        if reservation_node_delta is not None:
            expected_node_work = np.maximum(expected_node_work + reservation_node_delta, 0.0)
        stage_key = (request.service_id, stage_id)
        stage_work = expected_stage_work.get(stage_key, np.zeros(self.num_nodes)).copy()
        if reservation_stage_delta is not None and stage_key in reservation_stage_delta:
            stage_work = np.maximum(stage_work + reservation_stage_delta[stage_key], 0.0)
        candidate_post_work = self._candidate_post_compute_workload(
            env,
            request,
            stage_id,
            expected_node_work,
        )
        node_features = []
        incoming_data_mb = (
            float(request.input_mb)
            if stage_id == 0
            else float(request.stage_output_mb[stage_id - 1])
        )
        current_output_mb = float(request.stage_output_mb[stage_id])
        next_stage_id = stage_id + 1
        has_next_stage = next_stage_id < len(request.stage_compute_gcycles)
        deployed = env.deployment[request.service_id, stage_id] if env.deployment is not None else np.zeros(self.num_nodes, dtype=bool)
        for node in nodes:
            bandwidth = env.scenario.bandwidth_mb_s[prev_node, node.node_id]
            if not np.isfinite(bandwidth):
                bandwidth = 0.0
            reachable = env.shortest_path(prev_node, node.node_id) is not None
            features = [
                    float(deployed[node.node_id]),
                    float(reachable),
                    node.compute_gcycles_per_s / max_compute,
                    env.node_compute_load[node.node_id],
                    bandwidth / max(max_bandwidth, 1e-9),
                    tick_node_counts[node.node_id] / max(tick_request_count, 1.0),
                    self._normalize_compute_pressure(expected_node_work[node.node_id], node.compute_gcycles_per_s),
                    self._normalize_compute_pressure(stage_work[node.node_id], node.compute_gcycles_per_s),
                    self._normalize_compute_pressure(
                        candidate_post_work[node.node_id],
                        node.compute_gcycles_per_s * max(1.0 - 0.75 * env.node_compute_load[node.node_id], 0.10),
                    ),
                ]
            if self.chain_context == "chain-v2":
                transfer_s, propagation_s = self._route_proxy_delay(
                    env,
                    prev_node,
                    node.node_id,
                    incoming_data_mb,
                )
                next_stage_overlap = float(
                    has_next_stage
                    and env.deployment is not None
                    and env.deployment[request.service_id, next_stage_id, node.node_id]
                )
                features.extend(
                    [
                        float(node.node_id == prev_node),
                        float(np.tanh(transfer_s / 0.25)) if np.isfinite(transfer_s) else 1.0,
                        float(np.tanh(propagation_s / 0.05)) if np.isfinite(propagation_s) else 1.0,
                        next_stage_overlap,
                    ]
                )
            node_features.extend(features)
        scalars = [
            request.service_id / max(env.config.num_service_types - 1, 1),
            stage_id / max(self.max_service_stages - 1, 1),
            prev_node / max(self.num_nodes - 1, 1),
            request.input_mb / 2.0,
            request.stage_compute_gcycles[stage_id] / 8.0,
            request.deadline_s / 0.6,
            request.request_count / 100.0,
            env.current_time_minute / max(env.config.episode_minutes, 1),
            len(request.stage_compute_gcycles) / self.max_service_stages,
            np.log1p(tick_request_count) / np.log1p(5_000.0),
            tick_request_event_count / max(env.config.num_edge_nodes * env.config.num_service_types, 1),
            tick_service_count / max(tick_request_count, 1.0),
        ]
        if self.chain_context == "chain-v2":
            total_compute = max(float(sum(request.stage_compute_gcycles)), 1e-9)
            remaining_compute = float(sum(request.stage_compute_gcycles[stage_id:]))
            remaining_output = float(sum(request.stage_output_mb[stage_id:]))
            scalars.extend(
                [
                    incoming_data_mb / 2.0,
                    current_output_mb / 2.0,
                    remaining_compute / total_compute,
                    float(np.tanh(remaining_output / 4.0)),
                ]
            )
        if self.policy_kind == "gat_node_scorer":
            node_features += self._build_edge_features(env)
        return np.asarray(scalars + node_features, dtype=np.float32)

    def stage_counterfactual_regrets(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_nodes: list[int] | tuple[int, ...],
    ) -> list[float]:
        """Approximate action regret against feasible chain-aware alternatives.

        The proxy uses effective compute capacity, routed incoming data, and a
        one-stage downstream lookahead.  It is deliberately independent of the
        actor logits so it remains a useful difference-reward signal.
        """

        if self.policy_kind in {"path_scorer", "structured_path", "kkt_autoregressive_path"}:
            # A stage-local alternative can be worse for the complete chain.
            # Attribute the path-level proxy gap once, not at every stage.
            queued = self._path_selected_proxy_regrets.get(id(request), [])
            regret = queued.pop(0) if queued else 0.0
            return [regret] + [
                0.0
            ] * max(len(stage_nodes) - 1, 0)

        regrets: list[float] = []
        partial: list[int] = []
        for stage_id, selected_node in enumerate(stage_nodes):
            mask = self._build_mask(env, request, stage_id, partial)
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                regrets.append(0.0)
            else:
                costs = [
                    self._stage_candidate_proxy_cost(
                        env,
                        request,
                        stage_id,
                        int(candidate),
                        partial,
                    )
                    for candidate in candidates
                ]
                selected_cost = self._stage_candidate_proxy_cost(
                    env,
                    request,
                    stage_id,
                    int(selected_node),
                    partial,
                )
                best_cost = min(costs)
                regrets.append(max(float(selected_cost - best_cost), 0.0))
            partial.append(int(selected_node))
        return regrets

    def stage_selected_node_marginal_values(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_nodes: list[int] | tuple[int, ...],
    ) -> list[float]:
        """Estimate the value of keeping each selected node on the same trace.

        Positive means that forcing this stage onto its best other deployed
        replica would increase latency; negative means another deployed node
        was preferable.  Unlike the actor-logit baseline, this signal is tied
        to the actual request, chain prefix, topology, and resource state.
        """

        marginal_values: list[float] = []
        partial: list[int] = []
        for stage_id, selected_node_raw in enumerate(stage_nodes):
            selected_node = int(selected_node_raw)
            mask = self._build_mask(env, request, stage_id, partial)
            alternatives = [
                int(node_id)
                for node_id in np.flatnonzero(mask)
                if int(node_id) != selected_node
            ]
            if not alternatives:
                marginal_values.append(0.0)
            else:
                selected_cost = self._stage_candidate_proxy_cost(
                    env,
                    request,
                    stage_id,
                    selected_node,
                    partial,
                )
                alternative_cost = min(
                    self._stage_candidate_proxy_cost(
                        env,
                        request,
                        stage_id,
                        node_id,
                        partial,
                    )
                    for node_id in alternatives
                )
                marginal_values.append(
                    float(np.clip(alternative_cost - selected_cost, -2.0, 2.0))
                )
            partial.append(selected_node)
        return marginal_values

    def chain_oracle_schedule(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        *,
        beam_width: int = 32,
        candidates_per_stage: int = 8,
    ) -> list[int]:
        """Return a bounded beam-search oracle under the active deployment."""

        beam_width = max(int(beam_width), 1)
        candidates_per_stage = max(int(candidates_per_stage), 1)
        paths: list[tuple[list[int], float]] = [([], 0.0)]
        for stage_id in range(len(request.stage_compute_gcycles)):
            expanded: list[tuple[list[int], float]] = []
            for partial, prefix_cost in paths:
                mask = self._build_mask(env, request, stage_id, partial)
                ranked = sorted(
                    np.flatnonzero(mask).tolist(),
                    key=lambda node_id: self._stage_candidate_proxy_cost(
                        env, request, stage_id, int(node_id), partial
                    ),
                )[:candidates_per_stage]
                for node_id in ranked:
                    incremental = self._stage_candidate_proxy_cost(
                        env, request, stage_id, int(node_id), partial
                    )
                    expanded.append((partial + [int(node_id)], prefix_cost + incremental))
            if not expanded:
                return [request.home_node] * len(request.stage_compute_gcycles)
            expanded.sort(key=lambda item: item[1])
            paths = expanded[:beam_width]
        valid = [
            (path, env.evaluate_schedule(request, path))
            for path, _ in paths
        ]
        valid.sort(
            key=lambda item: (
                not bool(item[1]["valid"]),
                float(item[1]["latency_s"]),
            )
        )
        return valid[0][0]

    def _stage_candidate_proxy_cost(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        node_id: int,
        partial_nodes: list[int],
    ) -> float:
        assert env.scenario is not None
        prev_node = request.home_node if not partial_nodes else partial_nodes[-1]
        incoming_data_mb = (
            float(request.input_mb)
            if stage_id == 0
            else float(request.stage_output_mb[stage_id - 1])
        )
        transfer_s, propagation_s = self._route_proxy_delay(
            env,
            prev_node,
            node_id,
            incoming_data_mb,
        )
        if not np.isfinite(transfer_s):
            return float("inf")
        node = env.scenario.nodes[node_id]
        effective_capacity = float(node.compute_gcycles_per_s) * max(
            1.0 - 0.75 * float(env.node_compute_load[node_id]),
            0.10,
        )
        compute_s = float(request.stage_compute_gcycles[stage_id]) / max(effective_capacity, 1e-9)
        downstream_s = 0.0
        next_stage_id = stage_id + 1
        if next_stage_id < len(request.stage_compute_gcycles) and env.deployment is not None:
            next_candidates = np.flatnonzero(env.deployment[request.service_id, next_stage_id])
            lookahead: list[float] = []
            for next_node_id in next_candidates:
                next_transfer, next_propagation = self._route_proxy_delay(
                    env,
                    node_id,
                    int(next_node_id),
                    float(request.stage_output_mb[stage_id]),
                )
                if not np.isfinite(next_transfer):
                    continue
                next_node = env.scenario.nodes[int(next_node_id)]
                next_capacity = float(next_node.compute_gcycles_per_s) * max(
                    1.0 - 0.75 * float(env.node_compute_load[int(next_node_id)]),
                    0.10,
                )
                next_compute = float(request.stage_compute_gcycles[next_stage_id]) / max(
                    next_capacity,
                    1e-9,
                )
                lookahead.append(next_transfer + next_propagation + next_compute)
            if lookahead:
                downstream_s = 0.5 * min(lookahead)
        return compute_s + transfer_s + propagation_s + downstream_s

    @staticmethod
    def _route_proxy_delay(
        env: EdgeComputingEnv,
        src_node: int,
        dst_node: int,
        data_mb: float,
    ) -> tuple[float, float]:
        if int(src_node) == int(dst_node):
            return 0.0, 0.0
        assert env.scenario is not None
        path = env.shortest_path(int(src_node), int(dst_node))
        if path is None:
            return float("inf"), float("inf")
        transfer_s = 0.0
        propagation_s = 0.0
        for left, right in zip(path, path[1:]):
            raw_bandwidth = float(env.scenario.bandwidth_mb_s[left, right])
            residual = max(1.0 - 0.75 * float(env.link_load[left, right]), 0.10)
            transfer_s += float(data_mb) / max(raw_bandwidth * residual, 1e-9)
            propagation_s += float(env.scenario.propagation_ms[left, right]) / 1000.0
        return transfer_s, propagation_s

    def _reserve_compute_work(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        action: int,
        node_delta: np.ndarray,
        stage_deltas: dict[tuple[int, int], np.ndarray],
    ) -> None:
        """Replace the fair-share workload estimate with assignments already made.

        Unscheduled requests remain represented by the capacity-proportional
        baseline.  Once a microbatch is scheduled, its baseline share is
        removed and its work is reserved on the selected node.  Later
        microbatches therefore observe the congestion created by earlier ones.
        """

        assert env.scenario is not None
        assert env.deployment is not None
        candidates = env.deployment[request.service_id, stage_id]
        candidate_ids = np.flatnonzero(candidates)
        if candidate_ids.size == 0:
            return
        capacities = np.asarray(
            [node.compute_gcycles_per_s for node in env.scenario.nodes],
            dtype=np.float64,
        )
        shares = capacities[candidate_ids] / max(float(capacities[candidate_ids].sum()), 1e-9)
        work = float(request.stage_compute_gcycles[stage_id]) * float(request.request_count)
        delta = np.zeros(self.num_nodes, dtype=np.float64)
        delta[candidate_ids] -= work * shares
        delta[int(action)] += work
        node_delta += delta
        stage_key = (request.service_id, stage_id)
        if stage_key not in stage_deltas:
            stage_deltas[stage_key] = np.zeros(self.num_nodes, dtype=np.float64)
        stage_deltas[stage_key] += delta

    @staticmethod
    def _normalize_compute_pressure(work_gcycles: float, capacity_gcycles_per_s: float) -> float:
        """Expose instantaneous batch pressure without confusing it with EWMA load."""

        pressure = max(float(work_gcycles), 0.0) / max(float(capacity_gcycles_per_s), 1e-9)
        return float(np.tanh(pressure))

    def _expected_current_compute_workload(
        self,
        env: EdgeComputingEnv,
    ) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
        """Estimate current-second compute work under the active deployment.

        The environment settles all requests in a second jointly.  The old
        observation exposed only a one-minute EWMA, so the scheduler could not
        distinguish a lightly loaded node from a node facing a large current
        batch.  This estimate distributes each stage's work across its active
        replicas in proportion to their capacities and is cached for the
        current request tick.
        """

        env._require_ready()
        assert env.scenario is not None
        if env.deployment is None:
            return (
                np.zeros(self.num_nodes, dtype=np.float64),
                {},
            )
        cache_key = (
            id(env.current_requests),
            float(env.current_time_minute),
            float(env.metrics.get("deployment_updates", 0.0)),
        )
        if self._workload_cache_key == cache_key and self._workload_cache is not None:
            return self._workload_cache

        node_work = np.zeros(self.num_nodes, dtype=np.float64)
        stage_work: dict[tuple[int, int], np.ndarray] = {}
        capacities = np.asarray(
            [node.compute_gcycles_per_s for node in env.scenario.nodes],
            dtype=np.float64,
        )
        for current_request in env.current_requests:
            request_weight = float(current_request.request_count)
            for current_stage_id, compute_gcycles in enumerate(current_request.stage_compute_gcycles):
                key = (current_request.service_id, current_stage_id)
                candidates = env.deployment[current_request.service_id, current_stage_id]
                candidate_ids = np.flatnonzero(candidates)
                if candidate_ids.size == 0:
                    continue
                candidate_capacities = capacities[candidate_ids]
                shares = candidate_capacities / max(float(candidate_capacities.sum()), 1e-9)
                contribution = float(compute_gcycles) * request_weight * shares
                node_work[candidate_ids] += contribution
                if key not in stage_work:
                    stage_work[key] = np.zeros(self.num_nodes, dtype=np.float64)
                stage_work[key][candidate_ids] += contribution

        self._workload_cache_key = cache_key
        self._workload_cache = (node_work, stage_work)
        return self._workload_cache

    def _candidate_post_compute_workload(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        expected_node_work: np.ndarray,
    ) -> np.ndarray:
        """Estimate node work if this request stage is assigned to each candidate."""

        if env.deployment is None or stage_id >= len(request.stage_compute_gcycles):
            return np.zeros(self.num_nodes, dtype=np.float64)
        candidates = env.deployment[request.service_id, stage_id]
        candidate_ids = np.flatnonzero(candidates)
        if candidate_ids.size == 0:
            return np.zeros(self.num_nodes, dtype=np.float64)

        capacities = np.asarray(
            [node.compute_gcycles_per_s for node in env.scenario.nodes],
            dtype=np.float64,
        )
        candidate_capacities = capacities[candidate_ids]
        shares = candidate_capacities / max(float(candidate_capacities.sum()), 1e-9)
        request_work = float(request.stage_compute_gcycles[stage_id]) * float(request.request_count)
        baseline_without_request = expected_node_work.copy()
        baseline_without_request[candidate_ids] -= request_work * shares
        return np.where(
            candidates,
            baseline_without_request + request_work,
            0.0,
        )

    def _build_edge_features(self, env: EdgeComputingEnv) -> list[float]:
        assert env.scenario is not None
        bandwidth = env.scenario.bandwidth_mb_s
        propagation = env.scenario.propagation_ms
        max_bandwidth = np.nanmax(np.where(np.isfinite(bandwidth), bandwidth, 0.0))
        finite_propagation = np.where(np.isfinite(propagation), propagation, 0.0)
        max_propagation = max(float(finite_propagation.max()), 1e-9)
        features: list[float] = []
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                connected = bool(env.scenario.adjacency[src, dst])
                bw = bandwidth[src, dst] if np.isfinite(bandwidth[src, dst]) else max_bandwidth
                prop = propagation[src, dst] if np.isfinite(propagation[src, dst]) else max_propagation
                features.extend(
                    [
                        float(connected),
                        float(bw / max(max_bandwidth, 1e-9)),
                        float(prop / max_propagation),
                    ]
                )
        return features

    def _build_mask(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest,
        stage_id: int,
        partial_nodes: list[int],
    ) -> np.ndarray:
        assert env.scenario is not None
        mask = env.scheduler_candidate_mask(request)[stage_id].copy()
        prev_node = request.home_node if not partial_nodes else partial_nodes[-1]
        reachable = np.asarray(
            [env.shortest_path(prev_node, node_id) is not None for node_id in range(self.num_nodes)],
            dtype=bool,
        )
        mask &= reachable
        if not mask.any():
            mask = env.scheduler_candidate_mask(request)[stage_id].copy()
        if not mask.any():
            mask[request.home_node] = True
        return mask.astype(bool)


def _weighted_percentile(samples: list[tuple[float, float]], percentile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted((float(value), max(float(weight), 0.0)) for value, weight in samples)
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0.0:
        return float(ordered[-1][0])
    threshold = total_weight * float(percentile) / 100.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return float(ordered[-1][0])


@dataclass
class HierarchicalPPOAgent:
    slow_agent: SlowDeploymentPPOAgent
    fast_agent: FastSchedulingPPOAgent
    window_reward: float = 0.0
    window_steps: float = 0.0
    window_latency_samples: list[tuple[float, float]] = field(default_factory=list)
    window_penalty_latency_sum: float = 0.0
    window_deadline_violations: float = 0.0
    window_invalid_actions: float = 0.0
    window_max_node_compute_load: float = 0.0
    window_max_link_load: float = 0.0
    window_cross_stage_transitions: float = 0.0
    window_stage_transitions: float = 0.0
    window_cross_stage_data_mb: float = 0.0
    window_total_transition_data_mb: float = 0.0
    window_stage_latency_samples: dict[tuple[int, int], list[tuple[float, float]]] = field(default_factory=dict)
    window_stage_deadline_violations: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_weights: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_cross_transitions: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_transition_weights: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_cross_data_mb: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_transition_data_mb: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_used_nodes: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    window_stage_node_weights: dict[tuple[int, int], dict[int, float]] = field(default_factory=dict)
    window_stage_node_latency_samples: dict[
        tuple[int, int], dict[int, list[tuple[float, float]]]
    ] = field(default_factory=dict)
    window_stage_node_cross_transitions: dict[tuple[int, int, int], float] = field(default_factory=dict)
    window_stage_node_transition_weights: dict[tuple[int, int, int], float] = field(default_factory=dict)
    window_stage_node_cross_data_mb: dict[tuple[int, int, int], float] = field(default_factory=dict)
    window_stage_node_transition_data_mb: dict[tuple[int, int, int], float] = field(default_factory=dict)
    window_stage_node_counterfactual_values: dict[tuple[int, int, int], float] = field(
        default_factory=dict
    )
    window_stage_node_counterfactual_weights: dict[tuple[int, int, int], float] = field(
        default_factory=dict
    )
    slow_reward_scale: float = 1.0
    slow_tail_latency_coef: float = 0.35
    slow_colocation_coef: float = 0.05
    slow_deployment_memory_coef: float = 0.03
    slow_deployment_storage_coef: float = 0.01
    slow_migration_coef: float = 0.0
    slow_idle_replica_coef: float = 0.05
    slow_count_unused_replica_coef: float = 0.0
    slow_placement_idle_coef: float = 0.02
    slow_placement_compute_coef: float = 0.20
    slow_count_shortage_coef: float = 0.25
    slow_count_latency_coef: float = 1.0
    slow_deadline_violation_coef: float = 0.10
    slow_shared_return_coef: float = 0.0
    slow_counterfactual_credit_coef: float = 0.0
    window_deployment_memory_fraction: float = 0.0
    window_deployment_storage_fraction: float = 0.0
    window_migration_fraction: float = 0.0
    last_slow_window_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        env: EdgeComputingEnv,
        device: str = "cpu",
        replicas_per_stage: int = 5,
        slow_lr: float = 3e-4,
        slow_count_lr: float = 2e-4,
        slow_placement_lr: float = 1.5e-4,
        fast_lr: float = 2e-4,
        slow_k_epochs: int = 3,
        fast_k_epochs: int = 4,
        slow_entropy_coef: float = 0.001,
        slow_count_entropy_coef: float | None = None,
        slow_placement_entropy_coef: float | None = 0.005,
        slow_placement_entropy_final_coef: float | None = 0.0035,
        slow_placement_entropy_hold_updates: int = 64,
        slow_placement_entropy_decay_updates: int = 64,
        slow_placement_entropy_target: float | None = 1.8,
        slow_placement_entropy_max_coef: float = 0.05,
        slow_placement_entropy_adaptation_rate: float = 2e-3,
        slow_count_global_advantage_coef: float = 0.25,
        slow_placement_global_advantage_coef: float = 0.35,
        slow_placement_global_attribution_coef: float = 0.50,
        slow_global_advantage_ev_full: float = 0.20,
        fast_entropy_coef: float = 0.001,
        fast_entropy_target: float | None = 0.7,
        fast_entropy_max_coef: float = 0.03,
        fast_entropy_adaptation_rate: float = 2e-3,
        fast_entropy_normalized_control: bool = False,
        slow_value_coef: float = 0.5,
        slow_count_value_coef: float = 0.0,
        slow_critic_lr: float | None = 5e-4,
        slow_critic_k_epochs: int = 8,
        slow_critic_replay_windows: int = 96,
        slow_critic_replay_decay: float = 0.90,
        slow_critic_holdout_windows: int = 12,
        slow_critic_holdout_episodes: int = 2,
        slow_critic_gradient_clip: float = 5.0,
        slow_window_gamma: float = 0.95,
        fast_value_coef: float = 0.5,
        slow_target_kl: float | None = 0.03,
        slow_count_target_kl: float | None = 0.015,
        slow_placement_target_kl: float | None = 0.015,
        fast_target_kl: float | None = 0.015,
        slow_minibatch_size: int = 2048,
        fast_minibatch_size: int = 512,
        fast_policy_kind: str = "gat_node_scorer",
        fast_chain_context: str = "legacy",
        fast_reservation_microbatch_size: int = 16,
        fast_path_candidate_limit: int = 32,
        fast_load_balanced_updates: bool = True,
        fast_load_group_weights: tuple[float, ...] | None = None,
        fast_full_batch_kl_stop: bool = True,
        slow_reward_scale: float = 1.0,
        slow_tail_latency_coef: float = 0.35,
        slow_colocation_coef: float = 0.05,
        slow_deployment_memory_coef: float = 0.03,
        slow_deployment_storage_coef: float = 0.01,
        slow_migration_coef: float = 0.0,
        slow_idle_replica_coef: float = 0.05,
        slow_count_unused_replica_coef: float = 0.0,
        slow_placement_idle_coef: float = 0.02,
        slow_placement_compute_coef: float = 0.20,
        slow_count_shortage_coef: float = 0.25,
        slow_count_latency_coef: float = 1.0,
        slow_deadline_violation_coef: float = 0.10,
        slow_shared_return_coef: float = 0.0,
        slow_deterministic_count_mode: str = "expected",
        slow_allocation_mode: str = "legacy-factorized",
        slow_counterfactual_credit_coef: float = 0.0,
    ) -> "HierarchicalPPOAgent":
        return cls(
            slow_agent=SlowDeploymentPPOAgent(
                num_nodes=env.config.num_edge_nodes,
                num_service_types=env.config.num_service_types,
                max_service_stages=env.config.max_service_stages,
                replicas_per_stage=replicas_per_stage,
                lr=slow_lr,
                count_lr=slow_count_lr,
                placement_lr=slow_placement_lr,
                k_epochs=slow_k_epochs,
                entropy_coef=slow_entropy_coef,
                count_entropy_coef=slow_count_entropy_coef,
                placement_entropy_coef=slow_placement_entropy_coef,
                placement_entropy_final_coef=slow_placement_entropy_final_coef,
                placement_entropy_hold_updates=slow_placement_entropy_hold_updates,
                placement_entropy_decay_updates=slow_placement_entropy_decay_updates,
                placement_entropy_target=slow_placement_entropy_target,
                placement_entropy_max_coef=slow_placement_entropy_max_coef,
                placement_entropy_adaptation_rate=slow_placement_entropy_adaptation_rate,
                count_global_advantage_coef=slow_count_global_advantage_coef,
                placement_global_advantage_coef=slow_placement_global_advantage_coef,
                placement_global_attribution_coef=slow_placement_global_attribution_coef,
                global_advantage_ev_full=slow_global_advantage_ev_full,
                value_coef=slow_value_coef,
                count_value_coef=slow_count_value_coef,
                critic_lr=slow_critic_lr,
                critic_k_epochs=slow_critic_k_epochs,
                critic_replay_windows=slow_critic_replay_windows,
                critic_replay_decay=slow_critic_replay_decay,
                critic_holdout_windows=slow_critic_holdout_windows,
                critic_holdout_episodes=slow_critic_holdout_episodes,
                critic_gradient_clip=slow_critic_gradient_clip,
                window_gamma=slow_window_gamma,
                target_kl=slow_target_kl,
                count_target_kl=slow_count_target_kl,
                placement_target_kl=slow_placement_target_kl,
                minibatch_size=slow_minibatch_size,
                tail_latency_coef=slow_tail_latency_coef,
                deterministic_count_mode=slow_deterministic_count_mode,
                allocation_mode=slow_allocation_mode,
                counterfactual_credit_coef=slow_counterfactual_credit_coef,
                device=device,
            ),
            fast_agent=FastSchedulingPPOAgent(
                num_nodes=env.config.num_edge_nodes,
                max_service_stages=env.config.max_service_stages,
                policy_kind=fast_policy_kind,
                chain_context=fast_chain_context,
                lr=fast_lr,
                k_epochs=fast_k_epochs,
                entropy_coef=fast_entropy_coef,
                entropy_target=fast_entropy_target,
                entropy_max_coef=fast_entropy_max_coef,
                entropy_adaptation_rate=fast_entropy_adaptation_rate,
                entropy_normalized_control=fast_entropy_normalized_control,
                value_coef=fast_value_coef,
                target_kl=fast_target_kl,
                minibatch_size=fast_minibatch_size,
                reservation_microbatch_size=fast_reservation_microbatch_size,
                path_candidate_limit=fast_path_candidate_limit,
                load_balanced_updates=fast_load_balanced_updates,
                load_group_weights=fast_load_group_weights,
                full_batch_kl_stop=fast_full_batch_kl_stop,
                device=device,
            ),
            slow_reward_scale=slow_reward_scale,
            slow_tail_latency_coef=slow_tail_latency_coef,
            slow_colocation_coef=slow_colocation_coef,
            slow_deployment_memory_coef=slow_deployment_memory_coef,
            slow_deployment_storage_coef=slow_deployment_storage_coef,
            slow_migration_coef=slow_migration_coef,
            slow_idle_replica_coef=slow_idle_replica_coef,
            slow_count_unused_replica_coef=slow_count_unused_replica_coef,
            slow_placement_idle_coef=slow_placement_idle_coef,
            slow_placement_compute_coef=slow_placement_compute_coef,
            slow_count_shortage_coef=slow_count_shortage_coef,
            slow_count_latency_coef=slow_count_latency_coef,
            slow_deadline_violation_coef=slow_deadline_violation_coef,
            slow_shared_return_coef=slow_shared_return_coef,
            slow_counterfactual_credit_coef=slow_counterfactual_credit_coef,
        )

    def maybe_update_deployment(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> None:
        if not env.needs_deployment_update:
            return
        if record:
            self.flush_slow_window_reward(done=False, env=env)
        deployment = self.slow_agent.plan_deployment(env, deterministic=deterministic, record=record)
        migration_count = env.apply_deployment(deployment)
        if record:
            memory_fraction, storage_fraction = self._deployment_resource_fractions(env)
            assert env.scenario is not None
            stage_count = sum(len(service.stages) for service in env.scenario.services)
            possible_placements = max(stage_count * env.config.num_edge_nodes, 1)
            self.window_deployment_memory_fraction = memory_fraction
            self.window_deployment_storage_fraction = storage_fraction
            self.window_migration_fraction = float(migration_count / possible_placements)

    def reset_episode_context(self) -> None:
        """Drop feedback that belongs to a completed independent episode.

        PPO buffers intentionally remain intact because one optimizer update
        batches several ten-minute episodes.
        """

        self.slow_agent.last_window_feedback = {}
        self.last_slow_window_metrics = {}

    def act(self, env: EdgeComputingEnv, deterministic: bool = False, record: bool = True) -> list[int]:
        self.maybe_update_deployment(env, deterministic=deterministic, record=record)
        return self.fast_agent.schedule(env, deterministic=deterministic, record=record)

    def act_batch(
        self,
        env: EdgeComputingEnv,
        deterministic: bool = False,
        record: bool = True,
    ) -> list[list[int]]:
        self.maybe_update_deployment(env, deterministic=deterministic, record=record)
        return self.fast_agent.schedule_batch(
            env,
            list(env.current_requests),
            deterministic=deterministic,
            record=record,
        )

    def observe_step_reward(
        self,
        reward: float,
        stage_count: int,
        done: bool,
        weight: float = 1.0,
        fast_stage_rewards: list[float] | None = None,
        slow_reward: float | None = None,
        latency_s: float | None = None,
        penalty_latency_s: float | None = None,
        deadline_s: float | None = None,
        invalid: bool = False,
        max_node_compute_load: float | None = None,
        max_link_load: float | None = None,
        cross_stage_transitions: float = 0.0,
        stage_transitions: float = 0.0,
        stage_transition_data_mb: list[float] | tuple[float, ...] | None = None,
        service_id: int | None = None,
        stage_nodes: list[int] | tuple[int, ...] | None = None,
        slow_stage_costs: list[float] | None = None,
        slow_stage_marginal_values: list[float] | None = None,
        env: EdgeComputingEnv | None = None,
        record_fast: bool = True,
        record_slow: bool = True,
    ) -> None:
        if record_fast:
            if fast_stage_rewards is None:
                self.fast_agent.assign_last_schedule_reward(reward, stage_count, done, weight=weight)
            else:
                if len(fast_stage_rewards) != stage_count:
                    raise ValueError("fast_stage_rewards must match stage_count")
                self.fast_agent.assign_schedule_rewards(fast_stage_rewards, done=done, weight=weight)
        if not record_slow:
            return
        self.window_reward += (reward if slow_reward is None else slow_reward) * weight
        self.window_steps += weight
        if latency_s is not None:
            self.window_latency_samples.append((float(latency_s), float(weight)))
            self.window_penalty_latency_sum += float(penalty_latency_s or 0.0) * weight
            self.window_deadline_violations += float(
                bool(deadline_s is not None and float(latency_s) > float(deadline_s))
            ) * weight
            self.window_invalid_actions += float(bool(invalid)) * weight
            if max_node_compute_load is not None:
                self.window_max_node_compute_load = max(self.window_max_node_compute_load, float(max_node_compute_load))
            if max_link_load is not None:
                self.window_max_link_load = max(self.window_max_link_load, float(max_link_load))
        self.window_cross_stage_transitions += max(float(cross_stage_transitions), 0.0)
        self.window_stage_transitions += max(float(stage_transitions), 0.0)
        if service_id is not None and stage_nodes is not None and slow_stage_costs is not None:
            if len(stage_nodes) != len(slow_stage_costs):
                raise ValueError("slow_stage_costs must match stage_nodes")
            if slow_stage_marginal_values is not None and len(stage_nodes) != len(
                slow_stage_marginal_values
            ):
                raise ValueError("slow_stage_marginal_values must match stage_nodes")
            violated = float(bool(deadline_s is not None and latency_s is not None and latency_s > deadline_s))
            for stage_id, (node_id, stage_cost) in enumerate(zip(stage_nodes, slow_stage_costs)):
                stage_key = (int(service_id), int(stage_id))
                self.window_stage_latency_samples.setdefault(stage_key, []).append((float(stage_cost), float(weight)))
                self.window_stage_deadline_violations[stage_key] = (
                    self.window_stage_deadline_violations.get(stage_key, 0.0) + violated * weight
                )
                self.window_stage_weights[stage_key] = self.window_stage_weights.get(stage_key, 0.0) + weight
                self.window_stage_used_nodes.setdefault(stage_key, set()).add(int(node_id))
                node_weights = self.window_stage_node_weights.setdefault(stage_key, {})
                node_weights[int(node_id)] = node_weights.get(int(node_id), 0.0) + float(weight)
                node_samples = self.window_stage_node_latency_samples.setdefault(stage_key, {})
                node_samples.setdefault(int(node_id), []).append((float(stage_cost), float(weight)))
                if slow_stage_marginal_values is not None:
                    node_key = (int(service_id), int(stage_id), int(node_id))
                    self.window_stage_node_counterfactual_values[node_key] = (
                        self.window_stage_node_counterfactual_values.get(node_key, 0.0)
                        + float(slow_stage_marginal_values[stage_id]) * float(weight)
                    )
                    self.window_stage_node_counterfactual_weights[node_key] = (
                        self.window_stage_node_counterfactual_weights.get(node_key, 0.0)
                        + float(weight)
                    )
            # Give the placement actions that own each adjacent pair a local
            # cross-node credit.  The previous global window penalty never
            # reached factorized Placement returns, so the actor could not
            # learn colocation even though the diagnostic reported it.
            transition_data = list(stage_transition_data_mb or [])
            if transition_data and len(transition_data) != max(len(stage_nodes) - 1, 0):
                raise ValueError("stage_transition_data_mb must match adjacent stage transitions")
            for transition_id in range(1, len(stage_nodes)):
                crossed = float(int(stage_nodes[transition_id - 1]) != int(stage_nodes[transition_id]))
                data_mb = float(transition_data[transition_id - 1]) if transition_data else 1.0
                weighted_data = max(data_mb, 0.0) * float(weight)
                self.window_cross_stage_data_mb += crossed * weighted_data
                self.window_total_transition_data_mb += weighted_data
                for endpoint_stage_id in (transition_id - 1, transition_id):
                    stage_key = (int(service_id), int(endpoint_stage_id))
                    endpoint_node_id = int(stage_nodes[endpoint_stage_id])
                    self.window_stage_cross_transitions[stage_key] = (
                        self.window_stage_cross_transitions.get(stage_key, 0.0) + crossed * float(weight)
                    )
                    self.window_stage_transition_weights[stage_key] = (
                        self.window_stage_transition_weights.get(stage_key, 0.0) + float(weight)
                    )
                    self.window_stage_cross_data_mb[stage_key] = (
                        self.window_stage_cross_data_mb.get(stage_key, 0.0)
                        + crossed * weighted_data
                    )
                    self.window_stage_transition_data_mb[stage_key] = (
                        self.window_stage_transition_data_mb.get(stage_key, 0.0)
                        + weighted_data
                    )
                    node_key = (stage_key[0], stage_key[1], endpoint_node_id)
                    self.window_stage_node_cross_transitions[node_key] = (
                        self.window_stage_node_cross_transitions.get(node_key, 0.0)
                        + crossed * float(weight)
                    )
                    self.window_stage_node_transition_weights[node_key] = (
                        self.window_stage_node_transition_weights.get(node_key, 0.0)
                        + float(weight)
                    )
                    self.window_stage_node_cross_data_mb[node_key] = (
                        self.window_stage_node_cross_data_mb.get(node_key, 0.0)
                        + crossed * weighted_data
                    )
                    self.window_stage_node_transition_data_mb[node_key] = (
                        self.window_stage_node_transition_data_mb.get(node_key, 0.0)
                        + weighted_data
                    )
        if done:
            self.flush_slow_window_reward(done=True, env=env)

    def flush_slow_window_reward(self, done: bool, env: EdgeComputingEnv | None = None) -> None:
        total_weight = float(sum(weight for _, weight in self.window_latency_samples))
        if total_weight > 0.0:
            mean_latency = float(
                sum(latency * weight for latency, weight in self.window_latency_samples) / total_weight
            )
            p95_latency = _weighted_percentile(self.window_latency_samples, 95.0)
            tail_latency = (
                (1.0 - self.slow_tail_latency_coef) * mean_latency
                + self.slow_tail_latency_coef * p95_latency
            )
            latency_return = -self.slow_reward_scale * tail_latency
            feedback = {
                "avg_latency_s": mean_latency,
                "p95_latency_s": p95_latency,
                "avg_penalty_latency_s": self.window_penalty_latency_sum / total_weight,
                "deadline_violation_rate": self.window_deadline_violations / total_weight,
                "invalid_action_rate": self.window_invalid_actions / total_weight,
                "max_node_compute_load": self.window_max_node_compute_load,
                "max_link_load": self.window_max_link_load,
                "deployment_memory_fraction": self.window_deployment_memory_fraction,
                "deployment_storage_fraction": self.window_deployment_storage_fraction,
            }
        else:
            latency_return = self.window_reward / float(self.window_steps) if self.window_steps > 0 else 0.0
            feedback = {}
        cross_stage_rate = self.window_cross_stage_transitions / max(self.window_stage_transitions, 1e-9)
        data_weighted_cross_stage_rate = self.window_cross_stage_data_mb / max(
            self.window_total_transition_data_mb,
            1e-9,
        )
        colocation_rate_for_credit = (
            data_weighted_cross_stage_rate
            if self.window_total_transition_data_mb > 0.0
            else cross_stage_rate
        )
        colocation_cost = self.slow_colocation_coef * colocation_rate_for_credit
        if total_weight > 0.0:
            feedback.update(
                {
                    "cross_stage_transition_rate": cross_stage_rate,
                    "colocation_rate": 1.0 - cross_stage_rate,
                    "data_weighted_cross_stage_rate": data_weighted_cross_stage_rate,
                }
            )
        deployment_memory_cost = self.slow_deployment_memory_coef * self.window_deployment_memory_fraction
        deployment_storage_cost = self.slow_deployment_storage_coef * self.window_deployment_storage_fraction
        migration_cost = self.slow_migration_coef * self.window_migration_fraction
        deadline_cost = self.slow_deadline_violation_coef * float(feedback.get("deadline_violation_rate", 0.0))
        operating_cost = self.slow_reward_scale * (
            deployment_memory_cost + deployment_storage_cost + migration_cost + colocation_cost + deadline_cost
        )
        window_return = latency_return - operating_cost
        count_stage_returns, placement_stage_returns, count_credit_metrics = self._factorized_stage_returns(env)
        shared_return_credit = self.slow_shared_return_coef * window_return
        if shared_return_credit != 0.0:
            count_stage_returns = {
                key: value + shared_return_credit
                for key, value in count_stage_returns.items()
            }
            placement_stage_returns = {
                key: value + shared_return_credit
                for key, value in placement_stage_returns.items()
            }
        incremental_add_actions = sum(
            node_id is not None for node_id in self.slow_agent.pending_count_node_ids
        )
        incremental_stop_actions = sum(
            stage_key == (-1, -1)
            for stage_key in self.slow_agent.pending_count_stage_keys
        )
        self.slow_agent.assign_pending_reward(
            window_return,
            done=done,
            count_stage_returns=count_stage_returns,
            placement_stage_returns=placement_stage_returns,
        )
        self.slow_agent.last_window_feedback = feedback
        self.last_slow_window_metrics = {
            "slow_window_return": float(window_return),
            "slow_window_latency_return": float(latency_return),
            "slow_window_avg_latency": float(feedback.get("avg_latency_s", 0.0)),
            "slow_window_p95_latency": float(feedback.get("p95_latency_s", 0.0)),
            "slow_tail_latency_cost": float(
                -latency_return / max(self.slow_reward_scale, 1e-9)
                if total_weight > 0.0
                else 0.0
            ),
            "slow_colocation_credit_coef": float(self.slow_colocation_coef),
            "slow_colocation_cost": float(self.slow_reward_scale * colocation_cost),
            "slow_cross_stage_transition_rate": float(cross_stage_rate),
            "slow_colocation_rate": float(1.0 - cross_stage_rate),
            "slow_data_weighted_cross_stage_rate": float(data_weighted_cross_stage_rate),
            "slow_shared_return_credit": float(shared_return_credit),
            "slow_incremental_add_actions": float(incremental_add_actions),
            "slow_incremental_stop_actions": float(incremental_stop_actions),
            "slow_deployment_memory_cost": float(self.slow_reward_scale * deployment_memory_cost),
            "slow_deployment_storage_cost": float(self.slow_reward_scale * deployment_storage_cost),
            "slow_migration_cost": float(self.slow_reward_scale * migration_cost),
            "slow_deadline_violation_cost": float(self.slow_reward_scale * deadline_cost),
            "slow_factorized_stage_return_mean": (
                float(np.mean([value for key, value in placement_stage_returns.items() if len(key) == 2]))
                if placement_stage_returns
                else 0.0
            ),
            "slow_factorized_stage_return_std": (
                float(np.std([value for key, value in placement_stage_returns.items() if len(key) == 2]))
                if placement_stage_returns
                else 0.0
            ),
            "slow_factorized_count_return_mean": (
                float(np.mean(list(count_stage_returns.values()))) if count_stage_returns else 0.0
            ),
            "slow_factorized_count_return_std": (
                float(np.std(list(count_stage_returns.values()))) if count_stage_returns else 0.0
            ),
            **count_credit_metrics,
            "slow_deployment_memory_fraction": float(self.window_deployment_memory_fraction),
            "slow_deployment_storage_fraction": float(self.window_deployment_storage_fraction),
            "slow_migration_fraction": float(self.window_migration_fraction),
        }
        self.window_reward = 0.0
        self.window_steps = 0
        self.window_latency_samples.clear()
        self.window_penalty_latency_sum = 0.0
        self.window_deadline_violations = 0.0
        self.window_invalid_actions = 0.0
        self.window_max_node_compute_load = 0.0
        self.window_max_link_load = 0.0
        self.window_cross_stage_transitions = 0.0
        self.window_stage_transitions = 0.0
        self.window_cross_stage_data_mb = 0.0
        self.window_total_transition_data_mb = 0.0
        self.window_stage_latency_samples.clear()
        self.window_stage_deadline_violations.clear()
        self.window_stage_weights.clear()
        self.window_stage_cross_transitions.clear()
        self.window_stage_transition_weights.clear()
        self.window_stage_cross_data_mb.clear()
        self.window_stage_transition_data_mb.clear()
        self.window_stage_used_nodes.clear()
        self.window_stage_node_weights.clear()
        self.window_stage_node_latency_samples.clear()
        self.window_stage_node_cross_transitions.clear()
        self.window_stage_node_transition_weights.clear()
        self.window_stage_node_cross_data_mb.clear()
        self.window_stage_node_transition_data_mb.clear()
        self.window_stage_node_counterfactual_values.clear()
        self.window_stage_node_counterfactual_weights.clear()
        self.window_deployment_memory_fraction = 0.0
        self.window_deployment_storage_fraction = 0.0
        self.window_migration_fraction = 0.0

    def _factorized_stage_returns(
        self,
        env: EdgeComputingEnv | None,
    ) -> tuple[dict[tuple[int, ...], float], dict[tuple[int, ...], float], dict[str, float]]:
        """Build distinct Count efficiency and Placement latency returns."""

        if env is None or env.scenario is None or env.deployment is None:
            return {}, {}, {
                "slow_count_effective_replicas_per_stage": 0.0,
                "slow_count_redundant_replica_fraction": 0.0,
                "slow_count_unused_replica_fraction": 0.0,
                "slow_count_unused_replica_cost": 0.0,
                "slow_count_replica_imbalance_fraction": 0.0,
                "slow_count_replica_efficiency_cost": 0.0,
                "slow_placement_node_compute_load": 0.0,
                "slow_placement_node_compute_cost": 0.0,
                "slow_counterfactual_marginal_value": 0.0,
                "slow_counterfactual_samples": 0.0,
            }
        memory_capacity = max(float(env.service_memory_capacities().sum()), 1e-9)
        storage_capacity = max(float(env.service_storage_capacities().sum()), 1e-9)
        count_returns: dict[tuple[int, ...], float] = {}
        placement_returns: dict[tuple[int, ...], float] = {}
        effective_replica_counts: list[float] = []
        redundant_replica_fractions: list[float] = []
        unused_replica_fractions: list[float] = []
        replica_imbalance_fractions: list[float] = []
        placement_node_compute_loads: list[float] = []
        counterfactual_values: list[float] = []
        for service in env.scenario.services:
            for stage in service.stages:
                stage_key = (service.service_id, stage.stage_id)
                samples = self.window_stage_latency_samples.get(stage_key, [])
                if not samples:
                    continue
                total_weight = max(float(sum(weight for _, weight in samples)), 1e-9)
                mean_cost = float(sum(cost * weight for cost, weight in samples) / total_weight)
                p95_cost = _weighted_percentile(samples, 95.0)
                tail_cost = (1.0 - self.slow_tail_latency_coef) * mean_cost + self.slow_tail_latency_coef * p95_cost
                placed = env.deployment[service.service_id, stage.stage_id]
                replica_count = int(placed.sum())
                placed_nodes = set(np.flatnonzero(placed).tolist())
                node_weights = self.window_stage_node_weights.get(stage_key, {})
                placed_weights = np.asarray(
                    [max(float(node_weights.get(node_id, 0.0)), 0.0) for node_id in placed_nodes],
                    dtype=np.float64,
                )
                assigned_weight = float(placed_weights.sum())
                if assigned_weight > 0.0:
                    shares = placed_weights / assigned_weight
                    effective_replicas = min(
                        1.0 / max(float(np.square(shares).sum()), 1e-9),
                        float(replica_count),
                    )
                else:
                    effective_replicas = 0.0
                redundant_replica_fraction = 1.0 - effective_replicas / max(float(replica_count), 1.0)
                used_replica_count = sum(
                    float(node_weights.get(node_id, 0.0)) > 0.0
                    for node_id in placed_nodes
                )
                unused_replica_fraction = 1.0 - used_replica_count / max(
                    float(replica_count),
                    1.0,
                )
                # Total redundancy contains two disjoint causes:
                #   unused: deployed nodes that served no sampled traffic;
                #   imbalance: concentration among nodes that did serve it.
                # When the explicit unused coefficient is enabled, charge the
                # disjoint components instead of adding unused on top of total
                # redundancy.  The zero-coefficient branch preserves legacy
                # checkpoint/configuration semantics.
                replica_imbalance_fraction = max(
                    redundant_replica_fraction - unused_replica_fraction,
                    0.0,
                )
                if self.slow_count_unused_replica_coef > 0.0:
                    replica_efficiency_cost = (
                        self.slow_idle_replica_coef * replica_imbalance_fraction
                        + self.slow_count_unused_replica_coef * unused_replica_fraction
                    )
                else:
                    replica_efficiency_cost = (
                        self.slow_idle_replica_coef * redundant_replica_fraction
                    )
                effective_replica_counts.append(effective_replicas)
                redundant_replica_fractions.append(redundant_replica_fraction)
                unused_replica_fractions.append(unused_replica_fraction)
                replica_imbalance_fractions.append(replica_imbalance_fraction)
                memory_fraction = replica_count * float(stage.memory_gb) / memory_capacity
                storage_fraction = replica_count * float(stage.storage_gb) / storage_capacity
                violation_rate = self.window_stage_deadline_violations.get(stage_key, 0.0) / max(
                    self.window_stage_weights.get(stage_key, 0.0), 1e-9
                )
                stage_cross_transition_rate = self.window_stage_cross_transitions.get(stage_key, 0.0) / max(
                    self.window_stage_transition_weights.get(stage_key, 0.0), 1e-9
                )
                stage_transition_data = self.window_stage_transition_data_mb.get(stage_key, 0.0)
                if stage_transition_data > 0.0:
                    stage_cross_transition_rate = self.window_stage_cross_data_mb.get(
                        stage_key,
                        0.0,
                    ) / stage_transition_data
                count_cost = (
                    self.slow_count_latency_coef * tail_cost
                    + self.slow_deployment_memory_coef * memory_fraction
                    + self.slow_deployment_storage_coef * storage_fraction
                    + replica_efficiency_cost
                    + self.slow_count_shortage_coef * violation_rate
                )
                placement_cost = (
                    tail_cost
                    + self.slow_deadline_violation_coef * violation_rate
                    + self.slow_colocation_coef * stage_cross_transition_rate
                )
                count_returns[stage_key] = -self.slow_reward_scale * float(count_cost)
                node_returns: list[float] = []
                node_sample_groups = self.window_stage_node_latency_samples.get(stage_key, {})
                for node_id in placed_nodes:
                    node_key = (service.service_id, stage.stage_id, int(node_id))
                    node_samples = node_sample_groups.get(int(node_id), [])
                    node_weight = float(sum(weight for _, weight in node_samples))
                    if node_weight > 0.0:
                        node_mean_cost = float(
                            sum(cost * weight for cost, weight in node_samples) / node_weight
                        )
                        node_p95_cost = _weighted_percentile(node_samples, 95.0)
                        node_tail_cost = (
                            (1.0 - self.slow_tail_latency_coef) * node_mean_cost
                            + self.slow_tail_latency_coef * node_p95_cost
                        )
                        idle_cost = 0.0
                    else:
                        node_tail_cost = tail_cost
                        idle_cost = self.slow_placement_idle_coef
                    node_transition_weight = self.window_stage_node_transition_weights.get(node_key, 0.0)
                    if node_transition_weight > 0.0:
                        node_cross_rate = self.window_stage_node_cross_transitions.get(node_key, 0.0) / max(
                            node_transition_weight,
                            1e-9,
                        )
                    else:
                        adjacent_stage_ids = [
                            adjacent_stage_id
                            for adjacent_stage_id in (stage.stage_id - 1, stage.stage_id + 1)
                            if 0 <= adjacent_stage_id < len(service.stages)
                        ]
                        node_cross_rate = (
                            float(
                                np.mean(
                                    [
                                        not bool(
                                            env.deployment[
                                                service.service_id,
                                                adjacent_stage_id,
                                                int(node_id),
                                            ]
                                        )
                                        for adjacent_stage_id in adjacent_stage_ids
                                    ]
                                )
                            )
                            if adjacent_stage_ids
                            else 0.0
                        )
                    node_transition_data = self.window_stage_node_transition_data_mb.get(node_key, 0.0)
                    if node_transition_data > 0.0:
                        node_cross_rate = self.window_stage_node_cross_data_mb.get(
                            node_key,
                            0.0,
                        ) / node_transition_data
                    counterfactual_weight = self.window_stage_node_counterfactual_weights.get(
                        node_key,
                        0.0,
                    )
                    marginal_value = (
                        self.window_stage_node_counterfactual_values.get(node_key, 0.0)
                        / counterfactual_weight
                        if counterfactual_weight > 0.0
                        else 0.0
                    )
                    if counterfactual_weight > 0.0:
                        counterfactual_values.append(float(marginal_value))
                    node_cost = (
                        node_tail_cost
                        + self.slow_deadline_violation_coef * violation_rate
                        + self.slow_colocation_coef * node_cross_rate
                        + self.slow_placement_compute_coef * float(env.node_compute_load[int(node_id)])
                        + idle_cost
                    )
                    placement_node_compute_loads.append(float(env.node_compute_load[int(node_id)]))
                    node_return = self.slow_reward_scale * (
                        -float(node_cost)
                        + self.slow_counterfactual_credit_coef * float(marginal_value)
                    )
                    placement_returns[node_key] = node_return
                    node_returns.append(node_return)
                    if self.slow_agent.allocation_mode == "global-incremental":
                        per_replica_resource_cost = (
                            self.slow_deployment_memory_coef
                            * float(stage.memory_gb)
                            / memory_capacity
                            + self.slow_deployment_storage_coef
                            * float(stage.storage_gb)
                            / storage_capacity
                        )
                        unused_cost = (
                            self.slow_count_unused_replica_coef
                            if node_weight <= 0.0
                            else 0.0
                        )
                        count_returns[node_key] = self.slow_reward_scale * (
                            self.slow_counterfactual_credit_coef * float(marginal_value)
                            - per_replica_resource_cost
                            - unused_cost
                        )
                placement_returns[stage_key] = (
                    float(np.mean(node_returns))
                    if node_returns
                    else -self.slow_reward_scale * float(placement_cost)
                )
        if self.slow_agent.allocation_mode == "global-incremental":
            # STOP has zero local marginal utility.  The shared window return
            # is added below just like it is for every add-replica action.
            count_returns[(-1, -1)] = 0.0
        return count_returns, placement_returns, {
            "slow_count_effective_replicas_per_stage": (
                float(np.mean(effective_replica_counts)) if effective_replica_counts else 0.0
            ),
            "slow_count_redundant_replica_fraction": (
                float(np.mean(redundant_replica_fractions)) if redundant_replica_fractions else 0.0
            ),
            "slow_count_unused_replica_fraction": (
                float(np.mean(unused_replica_fractions)) if unused_replica_fractions else 0.0
            ),
            "slow_count_unused_replica_cost": (
                self.slow_reward_scale
                * self.slow_count_unused_replica_coef
                * (float(np.mean(unused_replica_fractions)) if unused_replica_fractions else 0.0)
            ),
            "slow_count_replica_imbalance_fraction": (
                float(np.mean(replica_imbalance_fractions))
                if replica_imbalance_fractions
                else 0.0
            ),
            "slow_count_replica_efficiency_cost": (
                self.slow_reward_scale
                * (
                    self.slow_idle_replica_coef
                    * (
                        float(np.mean(replica_imbalance_fractions))
                        if replica_imbalance_fractions
                        else 0.0
                    )
                    + self.slow_count_unused_replica_coef
                    * (
                        float(np.mean(unused_replica_fractions))
                        if unused_replica_fractions
                        else 0.0
                    )
                )
                if self.slow_count_unused_replica_coef > 0.0
                else self.slow_reward_scale
                * self.slow_idle_replica_coef
                * (
                    float(np.mean(redundant_replica_fractions))
                    if redundant_replica_fractions
                    else 0.0
                )
            ),
            "slow_placement_node_compute_load": (
                float(np.mean(placement_node_compute_loads)) if placement_node_compute_loads else 0.0
            ),
            "slow_placement_node_compute_cost": (
                self.slow_reward_scale
                * self.slow_placement_compute_coef
                * (float(np.mean(placement_node_compute_loads)) if placement_node_compute_loads else 0.0)
            ),
            "slow_counterfactual_marginal_value": (
                float(np.mean(counterfactual_values)) if counterfactual_values else 0.0
            ),
            "slow_counterfactual_samples": float(len(counterfactual_values)),
        }

    def _deployment_resource_fractions(self, env: EdgeComputingEnv) -> tuple[float, float]:
        assert env.scenario is not None
        assert env.deployment is not None
        memory_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        storage_used = np.zeros(env.config.num_edge_nodes, dtype=np.float64)
        for service in env.scenario.services:
            for stage in service.stages:
                placed = env.deployment[service.service_id, stage.stage_id]
                memory_used += placed * stage.memory_gb
                storage_used += placed * stage.storage_gb
        memory_fraction = float(memory_used.sum() / max(float(env.service_memory_capacities().sum()), 1e-9))
        storage_fraction = float(storage_used.sum() / max(float(env.service_storage_capacities().sum()), 1e-9))
        return memory_fraction, storage_fraction

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, dict[str, float]]:
        return {
            "slow": self.update_slow(
                progress_label=f"{progress_label} slow PPO",
                progress_interval_seconds=progress_interval_seconds,
            ),
            "fast": self.update_fast(
                progress_label=f"{progress_label} fast PPO",
                progress_interval_seconds=progress_interval_seconds,
            ),
        }

    @property
    def completed_slow_windows(self) -> int:
        return len(self.slow_agent.window_returns)

    def update_slow(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        return self.slow_agent.update(
            progress_label=progress_label,
            progress_interval_seconds=progress_interval_seconds,
        )

    def update_fast(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        return self.fast_agent.update(
            progress_label=progress_label,
            progress_interval_seconds=progress_interval_seconds,
        )
