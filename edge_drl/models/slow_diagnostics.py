"""Pre-update baseline and per-window policy-change measurements.

These are descriptive diagnostics on correlated rollout windows, never a
claim of independent samples or a substitute reward/baseline estimator.
"""
from __future__ import annotations

import numpy as np
import torch


def weighted_variance(values, weights):
    mean = np.average(values, weights=weights)
    return float(np.average((values-mean)**2, weights=weights))


def baseline_diagnostics(returns, old_values, weights, window_ids, coefficient):
    residual = returns - coefficient * old_values
    variance = weighted_variance(returns, weights)
    adjusted = weighted_variance(residual, weights)
    by_window = np.asarray([np.average(residual[window_ids == w], weights=weights[window_ids == w])
                            for w in np.unique(window_ids)])
    return {'baseline_return_variance':variance, 'baseline_residual_variance':adjusted,
        'baseline_residual_variance_ratio':adjusted/variance if variance > 1e-12 else float('nan'),
        'baseline_window_residual_variance':float(np.var(by_window)),
        'diagnostic_windows':float(len(by_window))}


def window_policy_change(ppo, states, masks, actions, old_logprobs, window_ids):
    log_ratios = []
    with torch.no_grad():
        for start in range(0, len(actions), ppo.minibatch_size):
            selected = slice(start,start+ppo.minibatch_size)
            dist,_ = ppo.policy(torch.as_tensor(states[selected],device=ppo.device),
                                torch.as_tensor(masks[selected],dtype=torch.bool,device=ppo.device))
            new = dist.log_prob(torch.as_tensor(actions[selected],device=ppo.device))
            log_ratios.extend((new-torch.as_tensor(old_logprobs[selected],device=ppo.device)).cpu().tolist())
    log_ratios = np.asarray(log_ratios,dtype=np.float64)
    sums = np.asarray([log_ratios[window_ids == w].sum() for w in np.unique(window_ids)])
    kl = np.expm1(log_ratios)-log_ratios
    kl_sums = np.asarray([kl[window_ids == w].sum() for w in np.unique(window_ids)])
    return {'window_log_ratio_mean':float(sums.mean()), 'window_log_ratio_abs_max':float(np.abs(sums).max()),
        'window_kl_proxy_mean':float(kl_sums.mean()), 'window_kl_proxy_max':float(kl_sums.max())}


def score_gradient_diagnostics(ppo, states, masks, actions, returns, values, window_ids, coefficient):
    """Window score gradients at behavior policy, before any parameter step.

    Uses unnormalized MC returns versus the causal scaled value baseline;
    measures the unclipped estimator, not variance of the final normalized,
    clipped multi-epoch PPO update. No gradient is applied or kept in .grad.
    """
    parameters = [p for name,p in ppo.policy.named_parameters() if not name.startswith('critic.') and p.requires_grad]
    vectors = [[], []]
    for window in np.unique(window_ids):
        indices = np.flatnonzero(window_ids == window)
        accum = [torch.zeros(sum(p.numel() for p in parameters),device=ppo.device) for _ in range(2)]
        for start in range(0,len(indices),ppo.minibatch_size):
            idx = indices[start:start+ppo.minibatch_size]
            dist,_ = ppo.policy(torch.as_tensor(states[idx],device=ppo.device),torch.as_tensor(masks[idx],dtype=torch.bool,device=ppo.device))
            logprob = dist.log_prob(torch.as_tensor(actions[idx],device=ppo.device))
            for arm,target in enumerate((returns[idx],returns[idx]-coefficient*values[idx])):
                objective = -(logprob*torch.as_tensor(target,device=ppo.device)).sum()/len(indices)
                grads = torch.autograd.grad(objective,parameters,retain_graph=(arm==0),allow_unused=True)
                accum[arm] += torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(parameters,grads)])
        for arm in range(2):
            vectors[arm].append(accum[arm].cpu().numpy())
    arrays = [np.asarray(v,dtype=np.float64) for v in vectors]
    means = [v.mean(axis=0) for v in arrays]
    variances = [float(np.mean(np.sum((v-mean)**2,axis=1))) for v,mean in zip(arrays,means)]
    norm = np.linalg.norm(means[0])*np.linalg.norm(means[1])
    return {'score_gradient_mc_variance':variances[0], 'score_gradient_baseline_variance':variances[1],
        'score_gradient_variance_ratio':variances[1]/variances[0] if variances[0]>1e-20 else float('nan'),
        'score_gradient_mean_cosine':float(np.dot(means[0],means[1])/norm) if norm>1e-20 else float('nan')}
