"""On-policy Slow actors with bounded, value-only conditional replay.

No actions, advantages or policy labels are replayed. The baseline multiplier
for a rollout is fixed from PREVIOUS rollouts, never fitted to its own returns.
"""
from __future__ import annotations

import copy
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SlowConditionalLearning:
    def __init__(self, windows=96, samples_per_window=8, decay=.9, prefix_credit=False):
        self.windows = int(windows)
        self.samples_per_window = int(samples_per_window)
        self.decay = float(decay)
        if self.windows < 1 or self.samples_per_window < 1 or not 0 < self.decay <= 1:
            raise ValueError("invalid conditional critic replay settings")
        self.replay = []
        self.update_index = 0
        self.baseline_coef = 0.0
        self.prefix_credit = bool(prefix_credit)
        self.trace_reliability = 0.0

    def state_dict(self):
        return {"update_index": self.update_index, "baseline_coef": self.baseline_coef,
                "trace_reliability": self.trace_reliability, "prefix_credit": self.prefix_credit,
                "replay": [{"states": torch.from_numpy(x["states"].copy()),
                            "returns": torch.from_numpy(x["returns"].copy()),
                            "update": x["update"], "behavior_update":x.get("behavior_update", x["update"]-1),
                            "window_id":x.get("window_id",-1)} for x in self.replay]}

    def load_state_dict(self, state):
        self.update_index = int(state["update_index"])
        self.baseline_coef = float(state["baseline_coef"])
        if bool(state.get("prefix_credit", False)) != self.prefix_credit:
            raise ValueError("Slow prefix credit mode differs from checkpoint")
        self.trace_reliability = float(state.get("trace_reliability", 0.0))
        if not 0 <= self.baseline_coef <= 1:
            raise ValueError("invalid conditional baseline multiplier")
        self.replay = [{"states": x["states"].detach().cpu().numpy().copy(),
                        "returns": x["returns"].detach().cpu().numpy().copy(),
                        "update": int(x["update"]), "behavior_update":int(x.get("behavior_update",x["update"]-1)),
                        "window_id":int(x.get("window_id",-1))} for x in state["replay"]][-self.windows:]

    @staticmethod
    def prediction_metrics(returns, values, weights):
        mean = np.average(returns, weights=weights)
        var = np.average((returns - mean)**2, weights=weights)
        residual = returns - values
        residual_var = np.average((residual - np.average(residual, weights=weights))**2, weights=weights)
        return {"ev": float(1 - residual_var / var) if var > 1e-8 else 0.0,
                "mse": float(np.average(residual**2, weights=weights))}

    def update(self, ppo, actions, logprobs, advantages, returns, weights,
               critic_weights, window_ids):
        """Guard whole minibatch epochs; fit values only AFTER the actor."""
        n = len(actions)
        size = max(1, min(ppo.minibatch_size, n))
        states_np = np.stack(ppo.buffer.states)
        masks_np = np.stack(ppo.buffer.masks)
        used_coef = self.baseline_coef
        old_values = np.asarray(ppo.buffer.values, dtype=np.float32)
        before = self.prediction_metrics(returns, old_values, critic_weights)
        extra_diagnostics = {}
        structured = bool(getattr(ppo.policy,"structured_critic",False))
        if structured:
            from edge_drl.models.slow_diagnostics import baseline_diagnostics, score_gradient_diagnostics
            extra_diagnostics.update(baseline_diagnostics(returns,old_values,critic_weights,window_ids,used_coef))
            if self.update_index % 10 == 0:
                extra_diagnostics.update(score_gradient_diagnostics(ppo,states_np,masks_np,actions,returns,
                                                              old_values,window_ids,used_coef))
        actor_started = perf_counter()
        accepted = backtracks = actor_steps = consumed = checks = 0
        stopped = False
        metrics = {"loss": 0., "policy_loss": 0., "value_loss": 0.}
        gradient_metrics = {"actor_gradient_norm": float("nan"), "entropy_gradient_norm": float("nan")}
        rates = [float(g["lr"]) for g in ppo.optimizer.param_groups]
        for _ in range(ppo.k_epochs):
            order = np.random.permutation(n)
            policy_before = copy.deepcopy(ppo.policy.state_dict())
            optimizer_before = copy.deepcopy(ppo.optimizer.state_dict())
            epoch_accepted = False
            for attempt in range(7):
                if attempt:
                    ppo.policy.load_state_dict(policy_before)
                    # Adam's loader may retain references to state tensors.
                    # Never allow a retried step to mutate the saved snapshot.
                    ppo.optimizer.load_state_dict(copy.deepcopy(optimizer_before))
                for g, lr in zip(ppo.optimizer.param_groups, rates):
                    g["lr"] = lr * .5**attempt
                totals = dict.fromkeys(metrics, 0.)
                steps = 0
                for start in range(0, n, size):
                    idx = order[start:start+size]
                    states = torch.as_tensor(states_np[idx], device=ppo.device)
                    masks = torch.as_tensor(masks_np[idx], dtype=torch.bool, device=ppo.device)
                    acts = torch.as_tensor(actions[idx], device=ppo.device)
                    old = torch.as_tensor(logprobs[idx], device=ppo.device)
                    adv = torch.as_tensor(advantages[idx], device=ppo.device)
                    w = torch.as_tensor(weights[idx], device=ppo.device)
                    dist, _ = ppo.policy(states, masks)
                    ratio = (dist.log_prob(acts) - old).exp()
                    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-ppo.eps_clip, 1+ppo.eps_clip) * adv)
                    policy_loss = -(surrogate*w).sum() / w.sum().clamp_min(1e-8)
                    entropy_loss = -ppo.entropy_coef * (dist.entropy()*w).sum() / w.sum().clamp_min(1e-8)
                    loss = policy_loss + entropy_loss
                    if self.update_index % 10 == 0 and accepted == 0 and attempt == 0 and start == 0:
                        actor_parameters = [p for name,p in ppo.policy.named_parameters()
                                            if p.requires_grad and not name.startswith("critic.")]
                        for label, objective in (("actor_gradient_norm",policy_loss),
                                                 ("entropy_gradient_norm",entropy_loss)):
                            grads = torch.autograd.grad(objective, actor_parameters, retain_graph=True, allow_unused=True)
                            gradient_metrics[label] = float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
                    ppo.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    ppo._clip_policy_gradients()
                    ppo.optimizer.step()
                    steps += 1
                    consumed += 1
                    mass = float(w.sum()) / max(float(weights.sum()), 1e-8)
                    totals["loss"] += float(loss.detach()) * mass
                    totals["policy_loss"] += float(policy_loss.detach()) * mass
                diagnostics = ppo._buffer_policy_diagnostics(actions, logprobs, weights)
                checks += 1
                kl = diagnostics["approx_kl"]
                if (all(bool(torch.isfinite(p).all()) for p in ppo.policy.parameters())
                        and np.isfinite(kl) and (ppo.target_kl is None or kl <= ppo.target_kl)):
                    accepted += 1
                    actor_steps += steps
                    metrics = totals
                    epoch_accepted = True
                    break
                backtracks += 1
            if not epoch_accepted:
                ppo.policy.load_state_dict(policy_before)
                ppo.optimizer.load_state_dict(copy.deepcopy(optimizer_before))
                stopped = True
            for g, lr in zip(ppo.optimizer.param_groups, rates):
                g["lr"] = lr
            if stopped:
                break
        metrics.update(ppo._buffer_policy_diagnostics(actions, logprobs, weights))
        if structured:
            from edge_drl.models.slow_diagnostics import window_policy_change
            extra_diagnostics.update(window_policy_change(ppo,states_np,masks_np,actions,logprobs,window_ids))
            extra_diagnostics['actor_update_time_s'] = perf_counter()-actor_started
        self.update_index += 1
        for window in np.unique(window_ids):
            idx = np.flatnonzero(window_ids == window)
            rng = np.random.default_rng(self.update_index * 100003 + int(window))
            idx = rng.choice(idx, min(len(idx), self.samples_per_window), replace=False)
            self.replay.append({"states": states_np[idx].copy(), "returns": returns[idx].copy(),
                                "update": self.update_index, "behavior_update":self.update_index-1,
                                "window_id":int(window)})
        self.replay = self.replay[-self.windows:]
        critic_metrics = self.fit_critic(ppo)
        with torch.no_grad():
            post = []
            for start in range(0, n, size):
                states = torch.as_tensor(states_np[start:start+size], device=ppo.device)
                features = ppo.policy.conditional_critic_features(states)
                post.extend(ppo._raw_values(ppo.policy.critic(features).squeeze(-1)).cpu().tolist())
        after = self.prediction_metrics(returns, np.asarray(post), critic_weights)
        # The covariance-optimal state baseline scale is estimated only for
        # FUTURE rollouts. Never use these fitted returns for this actor update.
        centered_values = old_values - np.average(old_values, weights=critic_weights)
        centered_returns = returns - np.average(returns, weights=critic_weights)
        variance = np.average(centered_values**2, weights=critic_weights)
        covariance = np.average(centered_returns*centered_values, weights=critic_weights)
        optimal = float(np.clip(covariance / max(variance, 1e-8), 0., 1.))
        self.baseline_coef = .8 * self.baseline_coef + .2 * optimal
        self.trace_reliability = .8 * self.trace_reliability + .2 * float(np.clip(before["ev"], 0, 1))
        metrics.update(critic_metrics)
        metrics.update(gradient_metrics)
        metrics.update(extra_diagnostics)
        metrics["value_loss"] = critic_metrics["critic_extra_value_loss"]
        metrics.update({"epochs_completed": float(accepted), "optimizer_steps": float(actor_steps),
                        "minibatches_completed": float(consumed), "minibatches_planned": float(ppo.k_epochs * int(np.ceil(n/size))),
                        "samples_seen_fraction": 1., "min_group_seen_fraction": 1.,
                        "full_batch_kl_checks": float(checks), "kl_backtracks": float(backtracks),
                        "kl_early_stop": float(stopped), "value_return_mean": ppo.value_return_mean,
                        "value_return_std": ppo.value_return_std,
                        "accepted_step_lr": rates[0] * .5**attempt if epoch_accepted else 0.,
                        "critic_prequential_ev": before["ev"], "critic_prequential_mse": before["mse"],
                        "post_explained_variance": after["ev"], "critic_post_mse": after["mse"],
                        "baseline_coef_used": used_coef, "baseline_coef_next": self.baseline_coef,
                        "critic_replay_windows": float(len(self.replay))})
        ppo.buffer.clear()
        return metrics

    def fit_critic(self, ppo):
        started = perf_counter()
        states_np = np.concatenate([x["states"] for x in self.replay])
        returns = np.concatenate([x["returns"] for x in self.replay])
        weights_np = np.concatenate([np.full(len(x["returns"]), self.decay**(self.update_index-x["update"]) / len(x["returns"]), dtype=np.float32) for x in self.replay])
        targets = (returns - ppo.value_return_mean) / ppo.value_return_std
        features = []
        # Re-encode historical states with the CURRENT actor, then detach.
        # Graph/context inference is paid once, not once per value epoch.
        with torch.no_grad():
            for start in range(0, len(states_np), ppo.minibatch_size):
                states = torch.as_tensor(states_np[start:start+ppo.minibatch_size], device=ppo.device)
                features.append(ppo.policy.conditional_critic_features(states).detach())
        features = torch.cat(features)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=ppo.device)
        weights = torch.as_tensor(weights_np, dtype=torch.float32, device=ppo.device)
        epochs = max(1, ppo.critic_extra_epochs + ppo.k_epochs)
        for _ in range(epochs):
            ppo.optimizer.zero_grad(set_to_none=True)
            if getattr(ppo.policy,"structured_critic",False):
                loss = torch.zeros((),device=ppo.device)
                size = min(64,ppo.minibatch_size)
                # Accumulate the same weighted full-batch gradient while
                # bounding graph activations; do not cache learned critic encodings.
                for start in range(0,len(features),size):
                    selected = slice(start,start+size)
                    values = ppo.policy.critic(features[selected]).squeeze(-1)
                    local = (F.smooth_l1_loss(values,targets[selected],reduction="none")*weights[selected]).sum()/weights.sum()
                    (ppo.value_coef*local).backward()
                    loss += local.detach()
            else:
                values = ppo.policy.critic(features).squeeze(-1)
                loss = (F.smooth_l1_loss(values, targets, reduction="none") * weights).sum() / weights.sum()
                (ppo.value_coef * loss).backward()
            nn.utils.clip_grad_norm_(ppo.policy.critic.parameters(), 5.)
            ppo.optimizer.step()
        return {"critic_extra_optimizer_steps": float(epochs), "critic_optimizer_steps": float(epochs),
                "critic_extra_value_loss": float(loss.detach()), "critic_replay_samples": float(len(targets)),
                "critic_update_time_s":perf_counter()-started,
                "critic_replay_max_age":float(max(self.update_index-x['update'] for x in self.replay))}
