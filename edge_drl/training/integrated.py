from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from edge_drl.agents.deployment import DeploymentActorCriticUpdater, DeploymentDecision, TorchAgentDDeployer
from edge_drl.agents.mappo import PPOUpdater, RolloutBuffer
from edge_drl.agents.networks import AgentDActorCritic, AgentSActorCritic
from edge_drl.agents.torch_schedulers import TorchAgentSScheduler
from edge_drl.env.environment import EdgeComputingEnv
from .behavior_cloning import BCSamples, collect_bc_samples, pretrain_agent_s_bc
from edge_drl.world_model.model import WorldModel


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action_features: np.ndarray
    next_state: np.ndarray
    reward: float


class IntegratedTrainer:
    """Integrated dual-agent trainer with Agent-D, Agent-S, and world model."""

    def __init__(
        self,
        config: dict,
        device: str = "cpu",
        rollout_tasks: int = 512,
        world_batch_size: int = 64,
        ppo_lr: float = 3e-4,
        ppo_gae_lambda: float = 0.95,
        ppo_entropy_coef: float = 0.001,
    ):
        self.env = EdgeComputingEnv(config)
        self.device = torch.device(device)
        obs = self.env.reset()
        self.state_dim = int(obs.shape[0])
        self.action_feature_dim = 6
        self.agent_s = AgentSActorCritic(
            obs_dim=self.env.task_obs_dim,
            num_actions=self.env.path_manager.num_actions,
        ).to(self.device)
        self.scheduler = TorchAgentSScheduler(self.agent_s, device=device)
        self.ppo = PPOUpdater(self.agent_s, lr=ppo_lr, gae_lambda=ppo_gae_lambda, entropy_coef=ppo_entropy_coef)
        self.agent_d = AgentDActorCritic(
            obs_dim=self.state_dim,
            num_services=self.env.num_services,
            max_stages=self.env.max_stages,
            num_nodes=self.env.num_nodes,
        ).to(self.device)
        self.deployer = TorchAgentDDeployer(
            self.agent_d,
            service_stage_mask=self.env.service_stage_mask,
            stage_memory=self.env.stage_memory,
            stage_storage=self.env.stage_storage,
            memory_capacity=self.env.memory_capacity,
            storage_capacity=self.env.storage_capacity,
            max_replicas=int(config["simulation"]["max_service_replicas"]),
            device=device,
        )
        self.agent_d_updater = DeploymentActorCriticUpdater(self.agent_d)
        self.world_model = WorldModel(self.state_dim, self.action_feature_dim).to(self.device)
        self.world_opt = torch.optim.Adam(self.world_model.parameters(), lr=3e-4)
        self.rollout = RolloutBuffer()
        self.replay: deque[Transition] = deque(maxlen=10000)
        self.rollout_tasks = rollout_tasks
        self.world_batch_size = world_batch_size
        self.bc_samples: BCSamples | None = None

    def train(
        self,
        episodes: int,
        seconds: int | None = None,
        agent_d_warmup_episodes: int = 0,
    ) -> list[dict[str, float]]:
        history = []
        for ep in range(episodes):
            state = self.env.reset()
            self.rollout.clear()
            ep_reward = 0.0
            ep_tasks = 0
            ep_invalid = 0
            ep_delay_sum = 0.0
            ep_steps = 0
            max_seconds = seconds or int(self.env.config["simulation"]["seconds_per_episode"])
            deployment_interval = int(self.env.config["simulation"]["deployment_interval_seconds"])
            macro_decisions: list[DeploymentDecision] = []
            macro_returns: list[float] = []
            current_macro_decision: DeploymentDecision | None = None
            current_macro_reward = 0.0
            ppo_stats: list[dict[str, float]] = []
            world_losses: list[float] = []

            use_agent_d = ep >= agent_d_warmup_episodes
            for _ in range(max_seconds):
                deployment_action = None
                if use_agent_d and self.env.time_s % deployment_interval == 0:
                    if current_macro_decision is not None:
                        macro_decisions.append(current_macro_decision)
                        macro_returns.append(current_macro_reward)
                    deployment_action = self.deployer.decide(state)
                    current_macro_decision = self.deployer.last_decision
                    current_macro_reward = 0.0

                self.scheduler.reset_records()
                next_state, reward, done, info = self.env.step(self.scheduler, deployment_action=deployment_action)
                ep_reward += float(reward)
                current_macro_reward += float(reward)
                ep_tasks += int(info["num_tasks"])
                ep_invalid += int(info["invalid_schedule"])
                ep_delay_sum += float(info["average_delay"])
                ep_steps += 1

                action_features = self.scheduler.action_features(
                    self.env.path_manager.num_actions,
                    int(info["num_tasks"]),
                    int(info["invalid_schedule"]),
                )
                deployment_features = self._deployment_action_features(deployment_action)
                action_features = np.concatenate([action_features, deployment_features]).astype(np.float32)
                self.replay.append(Transition(state, action_features, next_state, float(reward)))
                self._append_scheduler_records(float(reward), done)

                if len(self.rollout.obs) >= self.rollout_tasks:
                    ppo_stats.append(self.ppo.update(self.rollout))
                    self.rollout.clear()
                world_loss = self._train_world_model_step()
                if world_loss > 0.0:
                    world_losses.append(world_loss)

                state = next_state
                if done:
                    break

            if current_macro_decision is not None:
                macro_decisions.append(current_macro_decision)
                macro_returns.append(current_macro_reward)

            if self.rollout.obs:
                ppo_stats.append(self.ppo.update(self.rollout))
                self.rollout.clear()
            agent_d_stats = self.agent_d_updater.update(macro_decisions, macro_returns)

            episode_stats = {
                    "episode": float(ep),
                    "reward": ep_reward,
                    "reward_mean": ep_reward / max(ep_steps, 1),
                    "tasks": float(ep_tasks),
                    "tasks_per_second": float(ep_tasks) / max(ep_steps, 1),
                    "avg_delay": ep_delay_sum / max(ep_steps, 1),
                    "invalid": float(ep_invalid),
                    "world_replay": float(len(self.replay)),
                    "agent_d_loss": float(agent_d_stats["agent_d_loss"]),
                    "deployments": float(len(macro_decisions)),
                    "agent_d_enabled": float(use_agent_d),
                    "seconds": float(ep_steps),
                    "world_loss": float(np.mean(world_losses)) if world_losses else 0.0,
                    "ppo_updates": float(len(ppo_stats)),
                }
            for name in (
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "explained_variance",
            ):
                episode_stats[f"ppo_{name}"] = (
                    float(np.mean([stats[name] for stats in ppo_stats])) if ppo_stats else 0.0
                )
            history.append(episode_stats)
        return history

    def pretrain_agent_s(
        self,
        seconds: int,
        epochs: int = 3,
        batch_size: int = 256,
        max_samples: int | None = None,
        lr: float = 1e-3,
    ) -> dict[str, float]:
        samples = collect_bc_samples(self.env.config, seconds=seconds, max_samples=max_samples)
        self.bc_samples = samples
        return pretrain_agent_s_bc(
            self.agent_s,
            samples,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=str(self.device),
        )

    def save_checkpoint(self, path: str | Path, episode: int | None = None, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episode": episode,
            "config": self.env.config,
            "state_dim": self.state_dim,
            "task_obs_dim": self.env.task_obs_dim,
            "action_feature_dim": self.action_feature_dim,
            "agent_s": self.agent_s.state_dict(),
            "agent_d": self.agent_d.state_dict(),
            "world_model": self.world_model.state_dict(),
            "ppo_optimizer": self.ppo.opt.state_dict(),
            "agent_d_optimizer": self.agent_d_updater.opt.state_dict(),
            "world_optimizer": self.world_opt.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict:
        payload = torch.load(Path(path), map_location=self.device)
        self.agent_s.load_state_dict(payload["agent_s"], strict=strict)
        self.agent_d.load_state_dict(payload["agent_d"], strict=strict)
        self.world_model.load_state_dict(payload["world_model"], strict=strict)
        if "ppo_optimizer" in payload:
            self.ppo.opt.load_state_dict(payload["ppo_optimizer"])
        if "agent_d_optimizer" in payload:
            self.agent_d_updater.opt.load_state_dict(payload["agent_d_optimizer"])
        if "world_optimizer" in payload:
            self.world_opt.load_state_dict(payload["world_optimizer"])
        return payload

    def scale_ppo_lr(self, factor: float) -> float:
        current = float(self.ppo.opt.param_groups[0]["lr"])
        return self.set_ppo_lr(current * float(factor))

    def set_ppo_lr(self, lr: float) -> float:
        for group in self.ppo.opt.param_groups:
            group["lr"] = float(lr)
        return float(lr)

    def _deployment_action_features(self, deployment_action: np.ndarray | None) -> np.ndarray:
        if deployment_action is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        density = float(deployment_action.sum()) / float(deployment_action.size)
        violation = self.env.deployment_violation(deployment_action)
        violation_score = violation["missing"] + violation["memory"] + 0.1 * violation["storage"]
        return np.array([density, float(violation_score)], dtype=np.float32)

    def _append_scheduler_records(self, reward: float, done: bool) -> None:
        decision_count = len(self.scheduler.decisions)
        if decision_count <= 0:
            return
        per_decision_reward = float(reward) / float(decision_count)
        for record in self.scheduler.decisions:
            self.rollout.obs.append(record.obs)
            self.rollout.masks.append(record.mask)
            self.rollout.actions.append(record.action)
            self.rollout.log_probs.append(record.log_prob)
            self.rollout.values.append(record.value)
            self.rollout.rewards.append(per_decision_reward)
            self.rollout.dones.append(done)

    def _train_world_model_step(self) -> float:
        if len(self.replay) < self.world_batch_size:
            return 0.0
        idx = np.random.choice(len(self.replay), size=self.world_batch_size, replace=False)
        batch = [self.replay[int(i)] for i in idx]
        state = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.stack([b.action_features for b in batch]), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        loss = self.world_model.loss(state, action, next_state, reward)
        self.world_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_opt.step()
        return float(loss.detach().cpu().item())
