"""Mixed-action PPO, deliberately separate from the Proposed PPO implementation.

One shared encoder, three conditional heads and one centralized value function
are optimized by a single Adam instance. Resource actions are Beta coordinates
transformed to positive priorities; they are not an analytical allocation rule.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Categorical

from edge_drl.models.ppo import RolloutBuffer


class UnifiedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, nodes: int, hidden: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        self.deployment = nn.Linear(hidden, nodes + 1)
        self.routing = nn.Linear(hidden, nodes + 1)
        self.resources = nn.Sequential(nn.Linear(hidden + nodes, hidden), nn.Tanh(),
                                       nn.Linear(hidden, 4))
        self.critic = nn.Linear(hidden, 1)
        self.nodes = nodes
        # Small initial categorical logits avoid accidental initial preference.
        for head in (self.deployment, self.routing, self.resources[-1]):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def distributions(self, states, masks, actions=None):
        encoded = self.shared(states)
        schedule = states[:, 2] > 0.5
        logits = torch.where(schedule[:, None], self.routing(encoded), self.deployment(encoded))
        categorical = Categorical(logits=logits.masked_fill(~masks, -1e9))
        if actions is None:
            return categorical, self.critic(encoded).squeeze(-1)
        selected = nn.functional.one_hot(actions.clamp(max=self.nodes - 1), self.nodes).float()
        params = nn.functional.softplus(self.resources(torch.cat((encoded, selected), -1))) + 1.0
        beta = Beta(params[:, :2], params[:, 2:])
        return categorical, beta, self.critic(encoded).squeeze(-1), schedule

    def evaluate(self, states, masks, actions, units, active):
        categorical, beta, values, schedule = self.distributions(states, masks, actions)
        enabled = active * schedule[:, None]
        logprobs = categorical.log_prob(actions) + (beta.log_prob(units) * enabled).sum(-1)
        # Latent Beta entropy is a separately scaled exploration regularizer.
        resource_entropy = (beta.entropy() * enabled).sum(-1) / enabled.sum(-1).clamp(min=1)
        entropy = categorical.entropy() + resource_entropy
        normalized = categorical.entropy() / masks.sum(-1).float().clamp(min=2).log()
        return logprobs, values, entropy, normalized, resource_entropy


def normalize_groups(values, weights, groups):
    result = np.zeros(len(values), dtype=np.float32)
    for group in set(groups):
        indices = np.asarray([value == group for value in groups])
        mass = weights[indices]
        mean = np.average(values[indices], weights=mass)
        std = np.sqrt(np.average((values[indices] - mean) ** 2, weights=mass))
        # A group with no variation contains no identifiable action signal.
        if std > 1e-6:
            result[indices] = (values[indices] - mean) / max(std, 0.1)
    return result


class MixedActionPPO:
    def __init__(self, *, obs_dim, num_nodes, hidden_dim, lr, k_epochs,
                 entropy_coef, value_coef, target_kl, minibatch_size, device):
        self.device = torch.device(device)
        self.policy = UnifiedActorCritic(obs_dim, num_nodes, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.units: list[np.ndarray] = []
        self.active: list[np.ndarray] = []
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.target_kl = target_kl
        self.minibatch_size = minibatch_size

    @torch.no_grad()
    def act(self, state, mask, deterministic=False, resource_active=(False, False), source_node=None):
        states = torch.as_tensor(state, device=self.device).float()[None]
        masks = torch.as_tensor(mask, device=self.device).bool()[None]
        categorical, _ = self.policy.distributions(states, masks)
        actions = categorical.probs.argmax(-1) if deterministic else categorical.sample()
        _, beta, _, _ = self.policy.distributions(states, masks, actions)
        units = (beta.mean if deterministic else beta.sample()).clamp(1e-6, 1 - 1e-6)
        resource_active = (resource_active[0], resource_active[1] and int(actions.item()) != source_node)
        self.last_active = np.asarray(resource_active, dtype=np.float32)
        active = torch.as_tensor(resource_active, device=self.device).float()[None]
        logprob, value, _, _, _ = self.policy.evaluate(states, masks, actions, units, active)
        return int(actions.item()), float(logprob.item()), float(value.item()), units[0].cpu().numpy()

    def update(self, returns, advantages, sample_weights, *, progress_label="",
               progress_interval_seconds=0.0):
        n = len(self.buffer.actions)
        if not n:
            raise RuntimeError("UnifiedPPO cannot optimize an empty rollout")
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)
        states = tensor(self.buffer.states)
        masks = tensor(self.buffer.masks, torch.bool)
        actions = tensor(self.buffer.actions, torch.long)
        units, active = tensor(self.units), tensor(self.active)
        oldlogs, oldvalues = tensor(self.buffer.logprobs), tensor(self.buffer.values)
        returns, advantages, weights = tensor(returns), tensor(advantages), tensor(sample_weights)
        seen = np.zeros(n, dtype=bool)
        optimizer_steps, epochs = 0, 0
        stopped = False
        last_print = time.perf_counter()
        sums = dict(loss=0., policy_loss=0., value_loss=0., entropy=0., normalized_entropy=0.,
                    resource_entropy=0., clip_fraction=0.)
        mass_total = 0.
        kl = 0.
        for epoch in range(self.k_epochs):
            permutation = np.random.permutation(n)
            for offset in range(0, n, self.minibatch_size):
                ids_np = permutation[offset:offset + self.minibatch_size]
                ids = torch.as_tensor(ids_np, device=self.device)
                logprobs, values, entropy, normalized, resource_entropy = self.policy.evaluate(
                    states[ids], masks[ids], actions[ids], units[ids], active[ids])
                ratio = (logprobs - oldlogs[ids]).exp()
                # Fixed global denominator: random minibatch mode composition
                # must not change deployment/scheduling's intended weights.
                batch_weights = weights[ids] * n / len(ids_np) / weights.sum()
                policy_loss = -(torch.minimum(ratio * advantages[ids],
                    ratio.clamp(0.8, 1.2) * advantages[ids]) * batch_weights).sum()
                value_error = (values - returns[ids]).square()
                clipped_values = oldvalues[ids] + (values - oldvalues[ids]).clamp(-0.2, 0.2)
                value_loss = (torch.maximum(value_error, (clipped_values - returns[ids]).square())
                              * batch_weights).sum()
                entropy_loss = (entropy * batch_weights).sum()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite UnifiedPPO loss; update aborted")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                optimizer_steps += 1
                seen[ids_np] = True
                mass = float(weights[ids].sum())
                denominator = weights[ids].sum().clamp(min=1e-9)
                means = dict(loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
                             value_loss=float(value_loss.detach()), entropy=float(entropy_loss.detach()),
                             normalized_entropy=float((normalized * weights[ids]).sum().detach() / denominator),
                             resource_entropy=float((resource_entropy * weights[ids]).sum().detach() / denominator),
                             clip_fraction=float(((ratio - 1).abs() > .2).float().mul(weights[ids]).sum().detach() / denominator))
                for key in sums:
                    sums[key] += means[key] * mass
                mass_total += mass
                if progress_label and progress_interval_seconds > 0 and time.perf_counter() - last_print >= progress_interval_seconds:
                    print(f"{progress_label} epoch={epoch+1}/{self.k_epochs} samples={offset+len(ids_np)}/{n}", flush=True)
                    last_print = time.perf_counter()
            epochs = epoch + 1
            # Check the whole rollout after a complete epoch, not only its last
            # minibatch. Every sampled load is visited before any KL early-stop.
            with torch.no_grad():
                kl_sum = 0.
                for offset in range(0, n, self.minibatch_size):
                    ids = slice(offset, offset + self.minibatch_size)
                    logs, _, _, _, _ = self.policy.evaluate(states[ids], masks[ids], actions[ids], units[ids], active[ids])
                    delta = logs - oldlogs[ids]
                    kl_sum += float(((delta.exp() - 1 - delta) * weights[ids]).sum())
                kl = kl_sum / float(weights.sum())
            if self.target_kl is not None and kl > self.target_kl:
                stopped = True
                break
        metrics = {key: value / max(mass_total, 1e-9) for key, value in sums.items()}
        metrics.update(approx_kl=kl, epochs_completed=float(epochs), kl_early_stop=float(stopped),
                       optimizer_steps=float(optimizer_steps), samples_seen_fraction=float(seen.mean()))
        self.buffer.clear()
        self.units.clear()
        self.active.clear()
        return metrics


class UnifiedHybridPPO(MixedActionPPO):
    """Compatibility façade for the mixed-action agent; no second optimizer."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.priorities = self.units
        self.last_priorities = np.asarray([.5, .5], dtype=np.float32)
        self.last_active = np.zeros(2, dtype=np.float32)

    def act(self, state, mask, deterministic=False):
        schedule = bool(state[2] > .5)
        source = int(round(float(state[12]) * max(self.policy.nodes - 1, 1)))
        action, logprob, value, units = super().act(
            state, mask, deterministic=deterministic,
            resource_active=(schedule and state[9] > 0, schedule and state[10] > 0),
            source_node=source)
        self.last_priorities = units
        return action, logprob, value

    def update_from_returns(self, returns, *, sample_weights, **kwargs):
        states = np.asarray(self.buffer.states)
        schedule = states[:, 2] > .5
        weights = np.asarray(sample_weights, dtype=np.float32).copy()
        for mode in (False, True):
            selected = schedule == mode
            if selected.any():
                weights[selected] /= weights[selected].sum()
        # Condition on action mode and sampled load, not across heterogeneous
        # deployment and scheduling return scales. Value targets stay physical.
        groups = [(bool(mode), float(load)) for mode, load in
                  zip(schedule, self.buffer.sample_groups)]
        advantages = normalize_groups(
            np.asarray(returns) - np.asarray(self.buffer.values), weights, groups)
        metrics = super().update(returns, advantages, weights, **kwargs)
        metrics["advantage_std"] = float(np.std(advantages))
        return metrics
