from copy import deepcopy

import numpy as np
import pytest
import torch

from edge_drl.agents.slow_transfer import copy_slow_actor
from edge_drl.models.quota import beta_quota_probabilities
from tests.comparison_helpers import tiny_replay_env
from tests.test_integer_slow_quota import make_agent
from tests.test_joint_credit_v2 import make_agent as beta_agent


def state_and_policy():
    env = tiny_replay_env(); env.reset()
    slow = make_agent(env, slow_allocation_mode='stage-budget-v5').slow_agent
    slow.plan_deployment(env)
    return slow, slow.count_ppo.buffer.states[0]


@pytest.mark.parametrize('alpha,beta', [(2.,2.), (5.,2.), (1.01,1.01), (1.001,1000.), (1000.,1.001), (1000.,1000.)])
@pytest.mark.parametrize('maximum', [0,1,3,5,31])
def test_discrete_mass_agrees_with_independent_beta_cdf(alpha, beta, maximum):
    from edge_drl.models.ppo import GraphAttentionActorCritic
    nodes = max(6,maximum+1)
    policy = GraphAttentionActorCritic(global_dim=3,node_feature_dim=2,edge_feature_dim=1,
        num_nodes=nodes,action_dim=nodes+1,action_mode='quota-beta',hidden_dim=8)
    state = np.zeros(3+2*nodes+nodes*nodes,dtype=np.float32)
    raw = np.array([alpha-1, beta-1])
    inverse = raw + np.log(-np.expm1(-raw))
    with torch.no_grad():
        policy.actor[-1].weight.zero_()
        policy.actor[-1].bias.copy_(torch.tensor(inverse, dtype=torch.float32))
        mask = torch.arange(nodes+1)[None, :] <= maximum
        distribution, _ = policy(torch.from_numpy(state)[None, :], mask)
    probabilities = distribution.probs[0].numpy()
    np.testing.assert_allclose(probabilities[:maximum+1], beta_quota_probabilities(alpha,beta,maximum), atol=2e-5, rtol=2e-4)
    assert np.all(probabilities[:maximum+1] > 0)
    assert np.all(probabilities[maximum+1:] == 0)
    assert np.isfinite(probabilities).all()


def test_actor_transfer_preserves_trained_beta_parameters_and_class_probabilities():
    env = tiny_replay_env(); env.reset()
    source = beta_agent(env).slow_agent
    target = make_agent(env, slow_allocation_mode='stage-budget-v5').slow_agent
    source.plan_deployment(env)
    critics = {name: deepcopy(getattr(target,name).policy.critic.state_dict())
               for name in ('count_ppo','stage_ppo','placement_ppo')}
    copy_slow_actor(source,target)
    # New global quota context is appended before the node/edge blocks.
    old_state = source.count_ppo.buffer.states[0]
    width = source._global_dim()
    new_state = np.r_[old_state[:width], np.zeros(5,dtype=np.float32), old_state[width:]]
    with torch.no_grad():
        original,_ = source.count_ppo.policy(torch.from_numpy(old_state)[None,:], torch.ones((1,1),dtype=torch.bool))
        mask = torch.arange(7)[None,:] <= 5
        new,_ = target.count_ppo.policy(torch.from_numpy(new_state)[None,:],mask)
    expected = beta_quota_probabilities(float(original.concentration1),float(original.concentration0),5)
    np.testing.assert_allclose(new.probs[0,:6].numpy(), expected, atol=2e-5, rtol=2e-4)
    for name in critics:
        head = getattr(target,name)
        assert not head.optimizer.state
        for key,value in head.policy.critic.state_dict().items():
            torch.testing.assert_close(value,critics[name][key],rtol=0,atol=0)
    fresh_count = make_agent(env).slow_agent
    before = deepcopy(fresh_count.count_ppo.policy.state_dict())
    copy_slow_actor(source,fresh_count,scope='stage-placement')
    for key,value in fresh_count.count_ppo.policy.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_integer_log_mass_has_correct_differentiable_actor_gradient():
    slow,state = state_and_policy()
    policy = slow.count_ppo.policy
    mask = torch.arange(7)[None,:] <= 5
    state = torch.from_numpy(state)[None,:]
    loss = -policy(state,mask)[0].log_prob(torch.tensor([4])).sum()
    gradient = torch.autograd.grad(loss,policy.actor[-1].bias)[0].detach().numpy()
    numeric = []
    parameter = policy.actor[-1].bias
    for i in range(2):
        with torch.no_grad():
            original = float(parameter[i]); parameter[i] = original + .002
            plus = -float(policy(state,mask)[0].log_prob(torch.tensor([4])))
            parameter[i] = original - .002
            minus = -float(policy(state,mask)[0].log_prob(torch.tensor([4])))
            parameter[i] = original
        numeric.append((plus-minus)/.004)
    np.testing.assert_allclose(gradient,numeric,atol=2e-4,rtol=5e-3)
    assert np.linalg.norm(gradient) > 0


def test_mean_decoding_records_probability_of_executed_integer():
    slow,state = state_and_policy()
    ppo = slow.count_ppo
    mask = np.arange(7) <= 5
    with torch.no_grad():
        distribution,_ = ppo.policy(torch.from_numpy(state)[None,:],torch.from_numpy(mask)[None,:])
    expected = int(np.floor(float((distribution.probs[0]*torch.arange(7)).sum())+.5))
    action,log_probability,_ = ppo.act(state,mask,deterministic=True,categorical_quota_mean=True)
    assert action == expected
    assert log_probability == pytest.approx(float(distribution.log_prob(torch.tensor([action]))),abs=1e-6)
