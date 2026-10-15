"""Regression for prequential baselines, critic replay and epoch KL rollback."""
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pytest
import torch

from edge_drl.agents.drl import HierarchicalPPOAgent
from train_dual_ppo import save_checkpoint, load_checkpoint, resolve_slow_learning_repair
from tests.comparison_helpers import tiny_replay_env


def collect(windows=3):
    torch.manual_seed(84)
    np.random.seed(84)
    env = tiny_replay_env(); env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env, training_repair="batch-credit-v1", fast_chain_context="chain-kkt-v3",
        slow_allocation_mode="budget-continuous-v2", slow_deployment_context="deployment-aware-v2",
        slow_learning_repair="conditional-replay-v1", slow_critic_replay_windows=6,
        slow_critic_holdout_windows=1, slow_minibatch_size=8,
        slow_k_epochs=2, slow_stage_k_epochs=2, slow_placement_k_epochs=2, slow_critic_k_epochs=4,
    )
    slow = agent.slow_agent
    for i in range(windows):
        env.apply_deployment(slow.plan_deployment(env))
        slow.assign_pending_reward(float(-1-i), done=True)
    return env, agent


def test_critic_feature_path_matches_forward():
    _, agent = collect()
    for ppo in [agent.slow_agent.count_ppo, agent.slow_agent.stage_ppo, agent.slow_agent.placement_ppo]:
        states = torch.as_tensor(np.stack(ppo.buffer.states))
        masks = torch.as_tensor(np.stack(ppo.buffer.masks))
        with torch.no_grad():
            expected = ppo.policy(states, masks)[1]
            actual = ppo.policy.critic(ppo.policy.conditional_critic_features(states)).squeeze(-1)
        torch.testing.assert_close(actual, expected)


def test_all_heads_coverage_replay_and_lagged_baseline(monkeypatch):
    env, agent = collect()
    slow = agent.slow_agent
    slow.update()
    for ppo in [slow.count_ppo, slow.stage_ppo, slow.placement_ppo]:
        learner = ppo.slow_conditional_learning
        assert len(learner.replay) == 3
        assert 0 <= learner.baseline_coef <= 1
    diag = slow.last_update_diagnostics
    for head in ["count", "stage", "placement"]:
        assert diag[head+"_samples_seen_fraction"] == 1
        assert diag[head+"_approx_kl"] <= .03
        assert diag[head+"_baseline_coef_used"] == 0
        assert diag[head+"_critic_optimizer_steps"] == 4
    assert diag['placement_optimizer_steps'] > 2  # genuine minibatch steps
    for i in range(5):
        env.apply_deployment(slow.plan_deployment(env))
        slow.assign_pending_reward(-1., done=True)
    before = slow.count_ppo.slow_conditional_learning.baseline_coef
    slow.update()
    assert slow.last_update_diagnostics['count_baseline_coef_used'] == before
    for ppo in [slow.count_ppo, slow.stage_ppo, slow.placement_ppo]:
        learner = ppo.slow_conditional_learning
        assert len(learner.replay) == 6
        assert all(len(x['returns']) <= 8 for x in learner.replay)
        assert all('actions' not in x and 'advantages' not in x for x in learner.replay)


def test_value_replay_never_updates_actor_or_actor_adam():
    _, agent = collect()
    slow = agent.slow_agent
    slow.update()
    ppo = slow.placement_ppo
    actor = {n:p.detach().clone() for n,p in ppo.policy.named_parameters() if not n.startswith('critic.')}
    moments = {n:deepcopy(ppo.optimizer.state.get(p, {})) for n,p in ppo.policy.named_parameters() if n in actor}
    ppo.slow_conditional_learning.fit_critic(ppo)
    for n,p in ppo.policy.named_parameters():
        if n in actor:
            assert torch.equal(actor[n], p)
            for key,value in moments[n].items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, ppo.optimizer.state[p][key])


def test_rejected_epoch_restores_actor_and_adam(monkeypatch):
    _, agent = collect()
    ppo = agent.slow_agent.placement_ppo
    actor = {n:p.detach().clone() for n,p in ppo.policy.named_parameters() if not n.startswith('critic.')}
    monkeypatch.setattr(ppo, '_buffer_policy_diagnostics', lambda *a,**k: {'approx_kl': 1., 'max_group_approx_kl': 1.})
    ids=np.asarray(agent.slow_agent.placement_window_ids)
    metrics=ppo.update_from_returns(np.array([-1.,-2.,-3.],dtype=np.float32)[ids], critic_window_ids=ids)
    assert metrics['optimizer_steps'] == 0
    assert metrics['kl_early_stop'] == 1
    for n,p in ppo.policy.named_parameters():
        if n in actor:
            assert torch.equal(actor[n], p)
            assert p not in ppo.optimizer.state


def test_repair_checkpoint_roundtrip_and_old_resume_default(tmp_path):
    env, agent = collect()
    agent.slow_agent.update()
    path=tmp_path/'new.pt'; save_checkpoint(agent,path,{})
    restored=HierarchicalPPOAgent.from_env(
        env, training_repair='batch-credit-v1', fast_chain_context='chain-kkt-v3',
        slow_allocation_mode='budget-continuous-v2', slow_deployment_context='deployment-aware-v2',
        slow_learning_repair='conditional-replay-v1')
    load_checkpoint(restored,path)
    for head in ['count','stage','placement']:
        a=getattr(agent.slow_agent,head+'_ppo').slow_conditional_learning
        b=getattr(restored.slow_agent,head+'_ppo').slow_conditional_learning
        assert a.baseline_coef == b.baseline_coef
        assert a.update_index == b.update_index
        assert len(a.replay) == len(b.replay)
        for x,y in zip(a.replay,b.replay):
            np.testing.assert_array_equal(x['states'],y['states'])
            np.testing.assert_array_equal(x['returns'],y['returns'])
    args=Namespace(load_checkpoint=str(path),slow_learning_repair='auto',slow_deployment_context='deployment-aware-v2')
    assert resolve_slow_learning_repair(args)=='conditional-replay-v1'
    old=tmp_path/'old.pt'; torch.save({'slow_deployment_context':'deployment-aware-v2'},old)
    args.load_checkpoint=str(old)
    assert resolve_slow_learning_repair(args)=='none'
    args.load_checkpoint=''
    assert resolve_slow_learning_repair(args)=='conditional-replay-v1'


def test_legacy_rejects_repair():
    env=tiny_replay_env(); env.reset()
    with pytest.raises(ValueError,match='requires deployment-aware'):
        HierarchicalPPOAgent.from_env(env,slow_learning_repair='conditional-replay-v1')


def test_bad_critic_does_not_contaminate_initial_actor_advantages():
    _, agent=collect()
    ppo=agent.slow_agent.count_ppo
    ppo.buffer.values[:]=[1000.,2000.,3000.]
    returns=np.array([-1.,-2.,-3.],dtype=np.float32)
    metrics=ppo.update_from_returns(returns,critic_window_ids=np.arange(3))
    assert metrics['baseline_coef_used']==0
    assert metrics['advantage_std']==pytest.approx(float(returns.std()))
    assert metrics['baseline_coef_next']==0  # historical covariance was negative


def test_rollback_preserves_existing_actor_adam_moments(monkeypatch):
    env, agent=collect()
    slow=agent.slow_agent
    slow.update()
    for i in range(2):
        env.apply_deployment(slow.plan_deployment(env))
        slow.assign_pending_reward(-2.,done=True)
    ppo=slow.placement_ppo
    actor={n:p.detach().clone() for n,p in ppo.policy.named_parameters() if not n.startswith('critic.')}
    moments={n:deepcopy(ppo.optimizer.state[p]) for n,p in ppo.policy.named_parameters() if n in actor}
    monkeypatch.setattr(ppo,'_buffer_policy_diagnostics',lambda *a,**k: {'approx_kl':1.,'max_group_approx_kl':1.})
    ids=np.asarray(slow.placement_window_ids)
    ppo.update_from_returns(np.full(len(ids),-2.,dtype=np.float32),critic_window_ids=ids)
    for n,p in ppo.policy.named_parameters():
        if n in actor:
            assert torch.equal(p,actor[n])
            for k,v in moments[n].items():
                if isinstance(v,torch.Tensor):
                    assert torch.equal(v,ppo.optimizer.state[p][k])


def test_upgrade_clears_only_obsolete_critic_adam(tmp_path):
    _, source=collect()
    source.slow_agent.learning_repair='none'
    for h in ['count','stage','placement']:
        getattr(source.slow_agent,h+'_ppo').slow_conditional_learning=None
    source.slow_agent.update()
    for p in source.fast_agent.ppo.policy.parameters():
        p.grad=torch.full_like(p,.01)
    source.fast_agent.ppo.optimizer.step()
    path=tmp_path/'old.pt'; save_checkpoint(source,path,{})
    _, upgraded=collect()
    load_checkpoint(upgraded,path)
    for h in ['count','stage','placement']:
        a=getattr(source.slow_agent,h+'_ppo')
        b=getattr(upgraded.slow_agent,h+'_ppo')
        for (na,pa),(nb,pb) in zip(a.policy.named_parameters(),b.policy.named_parameters()):
            assert na==nb and torch.equal(pa,pb)
            if na.startswith('critic.'):
                assert pa in a.optimizer.state
                assert pb not in b.optimizer.state
            else:
                for k,v in a.optimizer.state.get(pa,{}).items():
                    if isinstance(v,torch.Tensor):
                        assert torch.equal(v,b.optimizer.state[pb][k])
    for pa,pb in zip(source.fast_agent.ppo.policy.parameters(),upgraded.fast_agent.ppo.policy.parameters()):
        assert torch.equal(pa,pb)
        for k,v in source.fast_agent.ppo.optimizer.state[pa].items():
            if isinstance(v,torch.Tensor):
                assert torch.equal(v,upgraded.fast_agent.ppo.optimizer.state[pb][k])
