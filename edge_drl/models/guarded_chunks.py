"""Bounded multiple PPO steps per complete, group-covered epoch."""
import copy
import numpy as np
import torch


def grouped_chunks(n, groups, count):
    groups = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
    _, frequencies = np.unique(groups, return_counts=True)
    chunks = [[] for _ in range(min(count, n, int(frequencies.min())))]
    for group in np.unique(groups):
        indices = np.random.permutation(np.flatnonzero(groups == group))
        for chunk, part in zip(chunks, np.array_split(indices, len(chunks))):
            chunk.extend(part.tolist())
    return [np.random.permutation(chunk) for chunk in chunks if chunk]


def update_guarded_chunks(ppo, actions, old_logprobs, advantages, targets, weights,
                          groups, advantage_mean, advantage_std, critic_weights):
    n = len(actions)
    size = min(n, ppo.minibatch_size)
    critic_weights = weights if critic_weights is None else critic_weights
    states = np.stack(ppo.buffer.states)
    masks = np.stack(ppo.buffer.masks)
    rates = [float(g["lr"]) for g in ppo.optimizer.param_groups]
    accepted = steps = backtracks = checks = consumed = 0
    stopped = False
    metrics = {"loss": 0., "policy_loss": 0., "value_loss": 0.}
    accepted_lr = 0.
    for _ in range(ppo.k_epochs):
        chunks = grouped_chunks(n, groups, ppo.guarded_update_chunks)
        policy_before = copy.deepcopy(ppo.policy.state_dict())
        optimizer_before = copy.deepcopy(ppo.optimizer.state_dict())
        success = False
        for attempt in range(7):
            if attempt:
                ppo.policy.load_state_dict(policy_before)
                ppo.optimizer.load_state_dict(copy.deepcopy(optimizer_before))
            for g, lr in zip(ppo.optimizer.param_groups, rates):
                g["lr"] = lr * .5**attempt
            totals = dict.fromkeys(metrics, 0.)
            for chunk in chunks:
                ppo.optimizer.zero_grad(set_to_none=True)
                # Fixed global normalization preserves target group weights
                # even when chunk masses differ slightly.
                denom = max(float(weights.sum()) / len(chunks), 1e-8)
                value_denom = max(float(critic_weights.sum()) / len(chunks), 1e-8)
                for start in range(0, len(chunk), size):
                    ix = chunk[start:start+size]
                    state = torch.as_tensor(states[ix], device=ppo.device)
                    mask = torch.as_tensor(masks[ix], dtype=torch.bool, device=ppo.device)
                    action = torch.as_tensor(actions[ix], device=ppo.device)
                    old = torch.as_tensor(old_logprobs[ix], device=ppo.device)
                    adv = torch.as_tensor(advantages[ix], device=ppo.device)
                    target = torch.as_tensor(targets[ix], device=ppo.device)
                    weight = torch.as_tensor(weights[ix], device=ppo.device)
                    value_weight = torch.as_tensor(critic_weights[ix], device=ppo.device)
                    dist, value = ppo.policy(state, mask)
                    ratio = (dist.log_prob(action) - old).exp()
                    surrogate = torch.minimum(ratio*adv, ratio.clamp(1-ppo.eps_clip, 1+ppo.eps_clip)*adv)
                    actor = -(surrogate * weight).sum() / denom
                    critic = ((value-target).square()*value_weight).sum() / value_denom
                    entropy = (dist.entropy()*weight).sum() / denom
                    loss = actor + ppo.value_coef*critic - ppo.entropy_coef*entropy
                    loss.backward()
                    for key, val in (("loss",loss),("policy_loss",actor),("value_loss",critic)):
                        totals[key] += float(val.detach()) / len(chunks)
                    consumed += 1
                ppo._clip_policy_gradients()
                ppo.optimizer.step()
            diagnostics = ppo._buffer_policy_diagnostics(actions, old_logprobs, weights, groups)
            checks += 1
            kl = max(diagnostics["approx_kl"], diagnostics["max_group_approx_kl"])
            if (all(bool(torch.isfinite(p).all()) for p in ppo.policy.parameters())
                    and np.isfinite(kl) and (ppo.target_kl is None or kl <= ppo.target_kl)):
                accepted += 1
                steps += len(chunks)
                accepted_lr = rates[0] * .5**attempt
                metrics = totals
                success = True
                break
            backtracks += 1
        if not success:
            ppo.policy.load_state_dict(policy_before)
            ppo.optimizer.load_state_dict(copy.deepcopy(optimizer_before))
            stopped = True
        for g, lr in zip(ppo.optimizer.param_groups, rates):
            g["lr"] = lr
        if stopped:
            break
    metrics.update(ppo._buffer_policy_diagnostics(actions, old_logprobs, weights, groups))
    metrics.update({"epochs_completed":float(accepted), "optimizer_steps":float(steps),
                    "kl_backtracks":float(backtracks), "kl_early_stop":float(stopped),
                    "accepted_step_lr":accepted_lr, "samples_seen_fraction":1.,
                    "min_group_seen_fraction":1., "full_batch_kl_checks":float(checks),
                    "minibatches_completed":float(consumed),
                    "minibatches_planned":float(ppo.k_epochs * sum(int(np.ceil(len(c)/size)) for c in chunks)),
                    "advantage_mean":advantage_mean, "advantage_std":advantage_std,
                    "value_return_mean":ppo.value_return_mean, "value_return_std":ppo.value_return_std})
    ppo.buffer.clear()
    return metrics
