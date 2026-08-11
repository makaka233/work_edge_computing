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


def slow_node_feature_dim(num_service_types: int) -> int:
    return SLOW_NODE_BASE_FEATURE_DIM + num_service_types


def slow_obs_dim(num_nodes: int, num_service_types: int) -> int:
    return (
        SLOW_GLOBAL_DIM
        + num_nodes * slow_node_feature_dim(num_service_types)
        + num_nodes * num_nodes * SLOW_EDGE_FEATURE_DIM
    )


def fast_obs_dim(num_nodes: int, policy_kind: str = "gat_node_scorer") -> int:
    dim = FAST_GLOBAL_DIM + num_nodes * FAST_NODE_FEATURE_DIM
    if policy_kind == "gat_node_scorer":
        dim += num_nodes * num_nodes * FAST_EDGE_FEATURE_DIM
    return dim


FAST_GLOBAL_DIM = 12
FAST_NODE_FEATURE_DIM = 9
FAST_EDGE_FEATURE_DIM = 3


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
    placement_entropy_max_coef: float = 0.015
    placement_entropy_adaptation_rate: float = 5e-4
    count_global_advantage_coef: float = 0.25
    placement_global_advantage_coef: float = 0.35
    value_coef: float = 0.5
    count_value_coef: float = 0.0
    critic_lr: float | None = None
    critic_k_epochs: int = 4
    target_kl: float | None = 0.03
    count_target_kl: float | None = 0.015
    placement_target_kl: float | None = None
    minibatch_size: int = 2048
    tail_latency_coef: float = 0.35
    deterministic_count_mode: str = "expected"
    device: str = "cpu"
    count_ppo: PPOAgent = field(init=False)
    placement_ppo: PPOAgent = field(init=False)
    ppo: PPOAgent = field(init=False)
    window_critic: SlowWindowCritic = field(init=False)
    critic_optimizer: torch.optim.Optimizer = field(init=False)
    pending_count_indices: list[int] = field(default_factory=list)
    pending_placement_indices: list[int] = field(default_factory=list)
    pending_count_stage_keys: list[tuple[int, int]] = field(default_factory=list)
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
    pending_window_id: int | None = None
    last_window_feedback: dict[str, float] = field(default_factory=dict)
    placement_updates_completed: int = field(default=0, init=False)
    placement_entropy_current_coef: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.tail_latency_coef <= 1.0:
            raise ValueError("tail_latency_coef must be in [0, 1]")
        if self.deterministic_count_mode not in {"expected", "mode"}:
            raise ValueError("deterministic_count_mode must be 'expected' or 'mode'")
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
        self.placement_entropy_current_coef = placement_entropy_start
        self.count_ppo = PPOAgent(
            obs_dim=slow_obs_dim(self.num_nodes, self.num_service_types),
            action_dim=self.replicas_per_stage,
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
            policy_kind="slow_gat_count",
            global_dim=SLOW_GLOBAL_DIM,
            node_feature_dim=slow_node_feature_dim(self.num_service_types),
            edge_feature_dim=SLOW_EDGE_FEATURE_DIM,
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
        self.pending_placement_stage_keys.clear()
        window_id: int | None = None
        if record:
            if self.pending_window_id is not None:
                raise RuntimeError("previous slow deployment window has no completed return")
            window_state = self._build_window_state(env, demand, remaining_memory, remaining_storage)
            state_t = torch.as_tensor(window_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                old_value = float(self.window_critic(state_t).item())
            window_id = len(self.window_states)
            self.window_states.append(window_state)
            self.window_old_values.append(old_value)
            self.pending_window_id = window_id

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

    def assign_pending_reward(
        self,
        reward: float,
        done: bool,
        *,
        stage_returns: dict[tuple[int, int], float] | None = None,
        count_stage_returns: dict[tuple[int, int], float] | None = None,
        placement_stage_returns: dict[tuple[int, ...], float] | None = None,
    ) -> None:
        del done
        if self.pending_window_id is None:
            return
        if self.pending_window_id != len(self.window_returns):
            raise RuntimeError("slow deployment windows must be completed in collection order")
        self.window_returns.append(float(reward))
        stage_returns = {} if stage_returns is None else stage_returns
        count_stage_returns = stage_returns if count_stage_returns is None else count_stage_returns
        placement_stage_returns = stage_returns if placement_stage_returns is None else placement_stage_returns
        self.count_action_returns.extend(
            float(count_stage_returns.get(stage_key, reward)) for stage_key in self.pending_count_stage_keys
        )
        self.count_action_stage_keys.extend(self.pending_count_stage_keys)
        if len(self.pending_placement_indices) != len(self.pending_placement_stage_keys):
            raise ValueError("slow placement action indices and stage keys are misaligned")
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
        self.pending_placement_stage_keys.clear()

    def update(self, *, progress_label: str = "", progress_interval_seconds: float = 0.0) -> dict[str, float]:
        if self.pending_window_id is not None:
            raise ValueError("slow deployment buffer contains a window without a return")
        if not self.window_returns:
            return self.empty_update_metrics()
        if len(self.window_states) != len(self.window_returns):
            raise ValueError("slow deployment window states and returns are misaligned")

        returns = np.asarray(self.window_returns, dtype=np.float32)
        old_values = np.asarray(self.window_old_values, dtype=np.float32)
        window_advantages = returns - old_values
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
        count_metrics = self.count_ppo.update_from_returns(
            count_returns,
            sample_weights=self._equal_window_action_weights(count_ids),
            advantage_group_ids=np.asarray(
                [
                    service_id * self.max_service_stages + stage_id
                    for service_id, stage_id in self.count_action_stage_keys
                ],
                dtype=np.int64,
            ),
            actor_use_value_baseline=False,
            auxiliary_advantages=window_advantages[count_ids],
            auxiliary_advantage_coef=self.count_global_advantage_coef,
            progress_label=f"{progress_label} count",
            progress_interval_seconds=progress_interval_seconds,
        )
        placement_entropy_coef = self.placement_entropy_coefficient()
        self.placement_ppo.entropy_coef = placement_entropy_coef
        placement_metrics = self.placement_ppo.update_from_returns(
            placement_returns,
            sample_weights=self._equal_window_action_weights(placement_ids),
            advantage_group_ids=np.asarray(
                [
                    service_id * self.max_service_stages + stage_id
                    for service_id, stage_id in self.placement_action_stage_keys
                ],
                dtype=np.int64,
            ),
            actor_use_value_baseline=False,
            auxiliary_advantages=window_advantages[placement_ids],
            auxiliary_advantage_coef=self.placement_global_advantage_coef,
            progress_label=f"{progress_label} placement",
            progress_interval_seconds=progress_interval_seconds,
        )
        critic_metrics = self._update_window_critic(returns)
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
        return {
            "loss": count_metrics["loss"] + placement_metrics["loss"] + self.value_coef * critic_metrics["value_loss"],
            "policy_loss": count_metrics["policy_loss"] + placement_metrics["policy_loss"],
            "value_loss": count_metrics["value_loss"] + placement_metrics["value_loss"],
            "entropy": count_metrics["entropy"] + placement_metrics["entropy"],
            "approx_kl": max(count_metrics.get("approx_kl", 0.0), placement_metrics.get("approx_kl", 0.0)),
            "window_count": float(window_count),
            "window_return_mean": float(returns.mean()),
            "window_return_std": float(returns.std()),
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
            "count_loss": count_metrics["loss"],
            "count_advantage_mean": count_metrics.get("advantage_mean", 0.0),
            "count_advantage_std": count_metrics.get("advantage_std", 0.0),
            "count_global_advantage_mean": count_metrics.get("auxiliary_advantage_mean", 0.0),
            "count_global_advantage_std": count_metrics.get("auxiliary_advantage_std", 0.0),
            "count_global_advantage_coef": self.count_global_advantage_coef,
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
            "placement_global_advantage_coef": self.placement_global_advantage_coef,
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
            "count_loss": 0.0,
            "count_advantage_mean": 0.0,
            "count_advantage_std": 0.0,
            "count_global_advantage_mean": 0.0,
            "count_global_advantage_std": 0.0,
            "count_global_advantage_coef": self.count_global_advantage_coef,
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
            "placement_global_advantage_coef": self.placement_global_advantage_coef,
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

    def _update_window_critic(self, returns: np.ndarray) -> dict[str, float]:
        states = torch.as_tensor(np.stack(self.window_states), dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        value_loss = 0.0
        for _ in range(max(self.critic_k_epochs, 1)):
            values = self.window_critic(states)
            loss = nn.functional.mse_loss(values, targets)
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.window_critic.parameters(), 0.5)
            self.critic_optimizer.step()
            value_loss = float(loss.item())
        with torch.no_grad():
            predictions = self.window_critic(states).detach().cpu().numpy()
        target_variance = float(np.var(returns))
        explained_variance = 0.0 if target_variance <= 1e-8 else 1.0 - float(np.var(returns - predictions)) / target_variance
        return {"value_loss": value_loss, "explained_variance": explained_variance}

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
    ) -> np.ndarray:
        assert env.scenario is not None
        nodes = env.scenario.nodes
        memory_capacity = env.service_memory_capacities()
        storage_capacity = env.service_storage_capacities()
        max_compute = max(n.compute_gcycles_per_s for n in nodes)
        total_node_demand = demand.sum(axis=1)
        max_node_demand = max(float(total_node_demand.max()), 1e-9)
        deployed_nodes = (
            np.any(env.deployment, axis=(0, 1)).astype(np.float32)
            if env.deployment is not None
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
        if env.deployment is not None:
            current_replica_rate = float(env.deployment.mean())
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
    lr: float = 2e-4
    k_epochs: int = 4
    entropy_coef: float = 0.001
    entropy_target: float | None = 0.7
    entropy_max_coef: float = 0.01
    entropy_adaptation_rate: float = 5e-4
    value_coef: float = 0.5
    target_kl: float | None = 0.015
    minibatch_size: int = 512
    reservation_microbatch_size: int = 16
    load_balanced_updates: bool = True
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

    def __post_init__(self) -> None:
        if self.reservation_microbatch_size < 1:
            raise ValueError("reservation_microbatch_size must be >= 1")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        if self.entropy_target is not None and self.entropy_target < 0.0:
            raise ValueError("entropy_target must be non-negative")
        if self.entropy_max_coef < self.entropy_coef:
            raise ValueError("entropy_max_coef must be at least entropy_coef")
        if self.entropy_adaptation_rate < 0.0:
            raise ValueError("entropy_adaptation_rate must be non-negative")
        self.entropy_current_coef = float(self.entropy_coef)
        self.ppo = PPOAgent(
            obs_dim=fast_obs_dim(self.num_nodes, self.policy_kind),
            action_dim=self.num_nodes,
            hidden_dim=128,
            lr=self.lr,
            gamma=0.99,
            k_epochs=self.k_epochs,
            entropy_coef=self.entropy_current_coef,
            value_coef=self.value_coef,
            target_kl=self.target_kl,
            minibatch_size=self.minibatch_size,
            policy_kind=self.policy_kind,
            global_dim=FAST_GLOBAL_DIM,
            node_feature_dim=FAST_NODE_FEATURE_DIM,
            edge_feature_dim=FAST_EDGE_FEATURE_DIM,
            num_nodes=self.num_nodes,
            group_balanced_updates=self.load_balanced_updates,
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
                    self.ppo.buffer.sample_groups.append(float(env.config.demand_load_multiplier))
        return schedules

    def schedule_with_diagnostics(
        self,
        env: EdgeComputingEnv,
        request: TaskRequest | None = None,
    ) -> tuple[list[int], list[dict[str, float | int]]]:
        env._require_ready()
        if request is None:
            assert env.current_request is not None
            request = env.current_request

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
        if not has_samples:
            next_coef = entropy_coef
        elif self.entropy_target is None or not np.isfinite(observed_entropy):
            next_coef = self.entropy_coef
        else:
            next_coef = entropy_coef + self.entropy_adaptation_rate * (
                float(self.entropy_target) - observed_entropy
            )
            next_coef = float(np.clip(next_coef, self.entropy_coef, self.entropy_max_coef))
        self.entropy_current_coef = float(next_coef)
        self.ppo.entropy_coef = self.entropy_current_coef
        metrics["entropy_coef"] = float(entropy_coef)
        metrics["entropy_next_coef"] = self.entropy_current_coef
        metrics["entropy_target"] = float(self.entropy_target) if self.entropy_target is not None else 0.0
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
        deployed = env.deployment[request.service_id, stage_id] if env.deployment is not None else np.zeros(self.num_nodes, dtype=bool)
        for node in nodes:
            bandwidth = env.scenario.bandwidth_mb_s[prev_node, node.node_id]
            if not np.isfinite(bandwidth):
                bandwidth = 0.0
            reachable = env.shortest_path(prev_node, node.node_id) is not None
            node_features.extend(
                [
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
            )
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
        if self.policy_kind == "gat_node_scorer":
            node_features += self._build_edge_features(env)
        return np.asarray(scalars + node_features, dtype=np.float32)

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
    window_stage_latency_samples: dict[tuple[int, int], list[tuple[float, float]]] = field(default_factory=dict)
    window_stage_deadline_violations: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_weights: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_cross_transitions: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_transition_weights: dict[tuple[int, int], float] = field(default_factory=dict)
    window_stage_used_nodes: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    window_stage_node_weights: dict[tuple[int, int], dict[int, float]] = field(default_factory=dict)
    window_stage_node_latency_samples: dict[
        tuple[int, int], dict[int, list[tuple[float, float]]]
    ] = field(default_factory=dict)
    window_stage_node_cross_transitions: dict[tuple[int, int, int], float] = field(default_factory=dict)
    window_stage_node_transition_weights: dict[tuple[int, int, int], float] = field(default_factory=dict)
    slow_reward_scale: float = 1.0
    slow_tail_latency_coef: float = 0.35
    slow_colocation_coef: float = 0.05
    slow_deployment_memory_coef: float = 0.03
    slow_deployment_storage_coef: float = 0.01
    slow_migration_coef: float = 0.0
    slow_idle_replica_coef: float = 0.05
    slow_placement_idle_coef: float = 0.02
    slow_placement_compute_coef: float = 0.20
    slow_count_shortage_coef: float = 0.25
    slow_count_latency_coef: float = 1.0
    slow_deadline_violation_coef: float = 0.10
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
        slow_placement_entropy_max_coef: float = 0.015,
        slow_placement_entropy_adaptation_rate: float = 5e-4,
        slow_count_global_advantage_coef: float = 0.25,
        slow_placement_global_advantage_coef: float = 0.35,
        fast_entropy_coef: float = 0.001,
        fast_entropy_target: float | None = 0.7,
        fast_entropy_max_coef: float = 0.01,
        fast_entropy_adaptation_rate: float = 5e-4,
        slow_value_coef: float = 0.5,
        slow_count_value_coef: float = 0.0,
        slow_critic_lr: float | None = None,
        slow_critic_k_epochs: int = 4,
        fast_value_coef: float = 0.5,
        slow_target_kl: float | None = 0.03,
        slow_count_target_kl: float | None = 0.015,
        slow_placement_target_kl: float | None = 0.015,
        fast_target_kl: float | None = 0.015,
        slow_minibatch_size: int = 2048,
        fast_minibatch_size: int = 512,
        fast_policy_kind: str = "gat_node_scorer",
        fast_reservation_microbatch_size: int = 16,
        fast_load_balanced_updates: bool = True,
        fast_full_batch_kl_stop: bool = True,
        slow_reward_scale: float = 1.0,
        slow_tail_latency_coef: float = 0.35,
        slow_colocation_coef: float = 0.05,
        slow_deployment_memory_coef: float = 0.03,
        slow_deployment_storage_coef: float = 0.01,
        slow_migration_coef: float = 0.0,
        slow_idle_replica_coef: float = 0.05,
        slow_placement_idle_coef: float = 0.02,
        slow_placement_compute_coef: float = 0.20,
        slow_count_shortage_coef: float = 0.25,
        slow_count_latency_coef: float = 1.0,
        slow_deadline_violation_coef: float = 0.10,
        slow_deterministic_count_mode: str = "expected",
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
                value_coef=slow_value_coef,
                count_value_coef=slow_count_value_coef,
                critic_lr=slow_critic_lr,
                critic_k_epochs=slow_critic_k_epochs,
                target_kl=slow_target_kl,
                count_target_kl=slow_count_target_kl,
                placement_target_kl=slow_placement_target_kl,
                minibatch_size=slow_minibatch_size,
                tail_latency_coef=slow_tail_latency_coef,
                deterministic_count_mode=slow_deterministic_count_mode,
                device=device,
            ),
            fast_agent=FastSchedulingPPOAgent(
                num_nodes=env.config.num_edge_nodes,
                max_service_stages=env.config.max_service_stages,
                policy_kind=fast_policy_kind,
                lr=fast_lr,
                k_epochs=fast_k_epochs,
                entropy_coef=fast_entropy_coef,
                entropy_target=fast_entropy_target,
                entropy_max_coef=fast_entropy_max_coef,
                entropy_adaptation_rate=fast_entropy_adaptation_rate,
                value_coef=fast_value_coef,
                target_kl=fast_target_kl,
                minibatch_size=fast_minibatch_size,
                reservation_microbatch_size=fast_reservation_microbatch_size,
                load_balanced_updates=fast_load_balanced_updates,
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
            slow_placement_idle_coef=slow_placement_idle_coef,
            slow_placement_compute_coef=slow_placement_compute_coef,
            slow_count_shortage_coef=slow_count_shortage_coef,
            slow_count_latency_coef=slow_count_latency_coef,
            slow_deadline_violation_coef=slow_deadline_violation_coef,
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
        service_id: int | None = None,
        stage_nodes: list[int] | tuple[int, ...] | None = None,
        slow_stage_costs: list[float] | None = None,
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
            # Give the placement actions that own each adjacent pair a local
            # cross-node credit.  The previous global window penalty never
            # reached factorized Placement returns, so the actor could not
            # learn colocation even though the diagnostic reported it.
            for transition_id in range(1, len(stage_nodes)):
                crossed = float(int(stage_nodes[transition_id - 1]) != int(stage_nodes[transition_id]))
                for endpoint_stage_id in (transition_id - 1, transition_id):
                    stage_key = (int(service_id), int(endpoint_stage_id))
                    endpoint_node_id = int(stage_nodes[endpoint_stage_id])
                    self.window_stage_cross_transitions[stage_key] = (
                        self.window_stage_cross_transitions.get(stage_key, 0.0) + crossed * float(weight)
                    )
                    self.window_stage_transition_weights[stage_key] = (
                        self.window_stage_transition_weights.get(stage_key, 0.0) + float(weight)
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
        colocation_cost = self.slow_colocation_coef * cross_stage_rate
        if total_weight > 0.0:
            feedback.update(
                {
                    "cross_stage_transition_rate": cross_stage_rate,
                    "colocation_rate": 1.0 - cross_stage_rate,
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
            "slow_colocation_cost": float(self.slow_reward_scale * colocation_cost),
            "slow_cross_stage_transition_rate": float(cross_stage_rate),
            "slow_colocation_rate": float(1.0 - cross_stage_rate),
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
        self.window_stage_latency_samples.clear()
        self.window_stage_deadline_violations.clear()
        self.window_stage_weights.clear()
        self.window_stage_cross_transitions.clear()
        self.window_stage_transition_weights.clear()
        self.window_stage_used_nodes.clear()
        self.window_stage_node_weights.clear()
        self.window_stage_node_latency_samples.clear()
        self.window_stage_node_cross_transitions.clear()
        self.window_stage_node_transition_weights.clear()
        self.window_deployment_memory_fraction = 0.0
        self.window_deployment_storage_fraction = 0.0
        self.window_migration_fraction = 0.0

    def _factorized_stage_returns(
        self,
        env: EdgeComputingEnv | None,
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, ...], float], dict[str, float]]:
        """Build distinct Count efficiency and Placement latency returns."""

        if env is None or env.scenario is None or env.deployment is None:
            return {}, {}, {
                "slow_count_effective_replicas_per_stage": 0.0,
                "slow_count_redundant_replica_fraction": 0.0,
                "slow_placement_node_compute_load": 0.0,
                "slow_placement_node_compute_cost": 0.0,
            }
        memory_capacity = max(float(env.service_memory_capacities().sum()), 1e-9)
        storage_capacity = max(float(env.service_storage_capacities().sum()), 1e-9)
        count_returns: dict[tuple[int, int], float] = {}
        placement_returns: dict[tuple[int, ...], float] = {}
        effective_replica_counts: list[float] = []
        redundant_replica_fractions: list[float] = []
        placement_node_compute_loads: list[float] = []
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
                effective_replica_counts.append(effective_replicas)
                redundant_replica_fractions.append(redundant_replica_fraction)
                memory_fraction = replica_count * float(stage.memory_gb) / memory_capacity
                storage_fraction = replica_count * float(stage.storage_gb) / storage_capacity
                violation_rate = self.window_stage_deadline_violations.get(stage_key, 0.0) / max(
                    self.window_stage_weights.get(stage_key, 0.0), 1e-9
                )
                stage_cross_transition_rate = self.window_stage_cross_transitions.get(stage_key, 0.0) / max(
                    self.window_stage_transition_weights.get(stage_key, 0.0), 1e-9
                )
                count_cost = (
                    self.slow_count_latency_coef * tail_cost
                    + self.slow_deployment_memory_coef * memory_fraction
                    + self.slow_deployment_storage_coef * storage_fraction
                    + self.slow_idle_replica_coef * redundant_replica_fraction
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
                    node_cost = (
                        node_tail_cost
                        + self.slow_deadline_violation_coef * violation_rate
                        + self.slow_colocation_coef * node_cross_rate
                        + self.slow_placement_compute_coef * float(env.node_compute_load[int(node_id)])
                        + idle_cost
                    )
                    placement_node_compute_loads.append(float(env.node_compute_load[int(node_id)]))
                    node_return = -self.slow_reward_scale * float(node_cost)
                    placement_returns[node_key] = node_return
                    node_returns.append(node_return)
                placement_returns[stage_key] = (
                    float(np.mean(node_returns))
                    if node_returns
                    else -self.slow_reward_scale * float(placement_cost)
                )
        return count_returns, placement_returns, {
            "slow_count_effective_replicas_per_stage": (
                float(np.mean(effective_replica_counts)) if effective_replica_counts else 0.0
            ),
            "slow_count_redundant_replica_fraction": (
                float(np.mean(redundant_replica_fractions)) if redundant_replica_fractions else 0.0
            ),
            "slow_placement_node_compute_load": (
                float(np.mean(placement_node_compute_loads)) if placement_node_compute_loads else 0.0
            ),
            "slow_placement_node_compute_cost": (
                self.slow_reward_scale
                * self.slow_placement_compute_coef
                * (float(np.mean(placement_node_compute_loads)) if placement_node_compute_loads else 0.0)
            ),
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
