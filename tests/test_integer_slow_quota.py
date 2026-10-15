from copy import deepcopy

import numpy as np
import pytest
import torch

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.models.quota import beta_modal_fraction, beta_quota_probabilities
from tests.comparison_helpers import tiny_replay_env
from train_dual_ppo import load_checkpoint, save_checkpoint


def make_agent(env, **kwargs):
    kwargs.setdefault('slow_deployment_context', 'deployment-aware-v2')
    kwargs.setdefault('slow_allocation_mode', 'stage-budget-v4')
    return HierarchicalPPOAgent.from_env(env, replicas_per_stage=6,
        training_repair='batch-credit-v1', fast_chain_context='chain-kkt-v3',
        slow_learning_repair='prefix-credit-v2', slow_minibatch_size=16,
        slow_critic_replay_windows=12, slow_critic_holdout_windows=1,
        slow_critic_k_epochs=2, slow_k_epochs=2, slow_stage_k_epochs=2,
        slow_placement_k_epochs=2, fast_independent_critic=True, **kwargs)


@pytest.mark.parametrize('quota', [0, 5])
@pytest.mark.parametrize('allocation', ['stage-budget-v4','stage-budget-v5'])
def test_integer_endpoint_quotas_are_executed_and_recorded_exactly(monkeypatch, quota, allocation):
    env=tiny_replay_env();env.reset();agent=make_agent(env,slow_allocation_mode=allocation)
    def act(state, mask, **kwargs):
        assert mask[0] and mask[5] and not mask[6]
        return quota, -np.log(mask.sum()), 0.
    monkeypatch.setattr(agent.slow_agent.count_ppo,'act',act)
    dep=agent.slow_agent.plan_deployment(env)
    assert env.check_deployment_feasible(dep)[0]
    assert dep.sum()==5*(1+quota)
    buffer=agent.slow_agent.count_ppo.buffer
    assert all(isinstance(q,int) and q==quota for q in buffer.actions)
    assert not agent.slow_agent.count_ppo.continuous_actions


@pytest.mark.parametrize('allocation', ['stage-budget-v4','stage-budget-v5'])
def test_integer_probability_support_logprob_and_pre_action_observations(allocation):
    torch.manual_seed(42)
    env=tiny_replay_env();env.reset();agent=make_agent(env,slow_allocation_mode=allocation);slow=agent.slow_agent
    slow.plan_deployment(env)
    count=slow.count_ppo
    for state,mask,action,old in zip(count.buffer.states,count.buffer.masks,count.buffer.actions,count.buffer.logprobs):
        with torch.no_grad():
            dist,_=count.policy(torch.from_numpy(state).unsqueeze(0),torch.from_numpy(mask).unsqueeze(0))
        assert np.all(dist.probs[0].numpy()[mask]>0)
        assert np.all(dist.probs[0].numpy()[~mask]==0)
        assert float(dist.log_prob(torch.tensor([action])))==pytest.approx(old,abs=1e-6)
        context=state[slow._global_dim()-5:slow._global_dim()]
        assert context[0]==pytest.approx((mask.sum()-1)/6)
        np.testing.assert_array_equal(context[1:],[0.,0.,1.,0.])
    # Placement sees its selected quota and remaining quota before each action.
    for state in slow.placement_ppo.buffer.states:
        maximum,selected,remaining,is_count,is_placement=state[slow._global_dim()-5:slow._global_dim()]
        if is_placement:
            assert not is_count and 0 < remaining <= selected <= maximum <= 1.


@pytest.mark.parametrize('allocation', ['stage-budget-v4','stage-budget-v5'])
def test_true_window_returns_update_discrete_actor_and_resume(tmp_path,allocation):
    torch.manual_seed(17)
    env=tiny_replay_env();env.reset();agent=make_agent(env,slow_allocation_mode=allocation);slow=agent.slow_agent
    before=deepcopy(slow.count_ppo.policy.state_dict())
    for index in range(3):
        slow.plan_deployment(env)
        slow.assign_pending_reward(-1.-index,done=True)
    metrics=slow.update()
    assert metrics['count_optimizer_steps']>0
    assert any(not torch.equal(before[k],v) for k,v in slow.count_ppo.policy.state_dict().items()
               if not k.startswith('critic.'))
    assert {float(y) for row in slow.count_ppo.slow_conditional_learning.replay for y in row['returns']}=={-1.,-2.,-3.}
    path=tmp_path/'integer.pt';save_checkpoint(agent,path,{'update':1})
    restored=make_agent(env,slow_allocation_mode=allocation);load_checkpoint(restored,path)
    a=slow.plan_deployment(env,deterministic=True,record=False)
    b=restored.slow_agent.plan_deployment(env,deterministic=True,record=False)
    np.testing.assert_array_equal(a,b)
    assert restored.slow_agent.quota_decode==('categorical-mean' if allocation=='stage-budget-v5' else 'categorical-mode')


def test_beta_integer_decoder_uses_bin_mass_and_handles_extremes():
    for alpha,beta in [(2.,2.),(5.,2.),(1.001,1000.),(1000.,1.001)]:
        p=beta_quota_probabilities(alpha,beta,31)
        a=beta_modal_fraction(alpha,beta,31)
        assert p.sum()==pytest.approx(1.) and np.all(p>=0)
        assert int(np.floor(a*32))==int(np.argmax(p))
    assert beta_quota_probabilities(2.,2.,0).tolist()==[1.]
    with pytest.raises(ValueError):
        beta_modal_fraction(-1.,2.,31)


def test_decoder_version_mismatch_is_explicit(tmp_path):
    from tests.test_joint_credit_v2 import make_agent as beta_agent
    env=tiny_replay_env();env.reset();agent=beta_agent(env)
    path=tmp_path/'beta.pt';save_checkpoint(agent,path,{})
    agent.slow_agent.quota_decode='integer-mode'
    with pytest.raises(ValueError,match='quota decoder'):
        load_checkpoint(agent,path)
    with pytest.raises(ValueError,match='incompatible'):
        make_agent(env,slow_quota_decode='mean')
