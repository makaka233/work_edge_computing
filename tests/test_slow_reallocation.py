from copy import deepcopy

import numpy as np
import pytest
import torch

from edge_drl.agents.drl import HierarchicalPPOAgent
from tests.comparison_helpers import tiny_replay_env
from train_dual_ppo import save_checkpoint, load_checkpoint


def make_agent(env, **overrides):
    kwargs = dict(replicas_per_stage=6, training_repair='batch-credit-v1',
        fast_chain_context='chain-kkt-v3', slow_deployment_context='deployment-chain-v3',
        slow_allocation_mode='stage-budget-v6', slow_learning_repair='prefix-structured-v3',
        slow_minibatch_size=64, slow_critic_replay_windows=12, slow_critic_holdout_windows=1,
        slow_critic_k_epochs=2, slow_k_epochs=2, slow_stage_k_epochs=2,
        slow_placement_k_epochs=2, fast_independent_critic=True)
    kwargs.update(overrides)
    return HierarchicalPPOAgent.from_env(env, **kwargs)


def select_first(state, mask, **kwargs):
    return int(np.flatnonzero(mask)[0]), -np.log(mask.sum()), 0.


@pytest.mark.parametrize('quota', [0, 1, 5])
def test_reallocation_terminates_and_respects_full_physical_support(monkeypatch, quota):
    env = tiny_replay_env(); env.reset(); slow = make_agent(env).slow_agent
    monkeypatch.setattr(slow.stage_ppo, 'act', select_first)
    def count(state, mask, **kwargs):
        maximum = int(np.flatnonzero(mask)[-1])
        np.testing.assert_array_equal(mask, np.arange(7) <= maximum)
        action = min(quota, maximum)
        return action, -np.log(mask.sum()), 0.
    monkeypatch.setattr(slow.count_ppo, 'act', count)
    deployment = slow.plan_deployment(env)
    assert env.check_deployment_feasible(deployment)[0]
    stages = sum(len(s.stages) for s in env.scenario.services)
    assert deployment.sum() == stages * (1 if quota == 0 else 6)
    actions = len(slow.stage_ppo.buffer.actions)
    assert actions <= stages * 6 + stages + 1
    if quota == 0:
        assert actions == 2 * stages
        assert slow._reallocation_closed.sum() == stages
    # Tail phase and closed-state flags are observable BEFORE each action.
    tail = slow._global_dim() - slow.num_service_types * slow.max_service_stages - 1
    tail_states = [state for state in slow.stage_ppo.buffer.states if state[tail] == 1]
    assert all(np.all(state[tail+1:slow._global_dim()] >= 0) for state in tail_states)
    if quota == 0:
        assert sum(tail_states[-1][tail+1:slow._global_dim()]) == stages-1


def test_learned_stop_is_recorded_and_receives_real_window_return(monkeypatch):
    env = tiny_replay_env(); env.reset(); slow = make_agent(env).slow_agent
    def stage(state, mask, **kwargs):
        return (len(mask)-1 if mask[-1] else int(np.flatnonzero(mask)[0])), -np.log(mask.sum()), 0.
    monkeypatch.setattr(slow.stage_ppo, 'act', stage)
    monkeypatch.setattr(slow.count_ppo, 'act', lambda state, mask, **kw: (0, -np.log(mask.sum()), 0.))
    deployment = slow.plan_deployment(env)
    assert env.check_deployment_feasible(deployment)[0]
    assert slow.stage_ppo.buffer.actions[-1] == slow.stage_ppo.policy.action_dim-1
    assert slow.pending_stage_keys[-1] == (-1, -1)
    assert slow.stage_ppo.buffer.masks[-1][-1]
    slow.assign_pending_reward(-3.125, done=True)
    assert slow.stage_action_returns[-1] == -3.125
    assert len(slow.stage_action_returns) == len(slow.stage_ppo.buffer.actions)


def test_structured_critic_has_independent_graph_gradients_and_popart_preserves_values():
    env = tiny_replay_env(); env.reset(); slow = make_agent(env).slow_agent
    slow.plan_deployment(env)
    for head in (slow.count_ppo, slow.stage_ppo, slow.placement_ppo):
        states = torch.from_numpy(np.stack(head.buffer.states[:3]))
        masks = torch.from_numpy(np.stack(head.buffer.masks[:3]))
        head.optimizer.zero_grad(set_to_none=True)
        before = head.policy(states, masks)[1]
        before.sum().backward()
        assert all(p.grad is None for name,p in head.policy.named_parameters() if not name.startswith('critic.'))
        assert head.policy.critic.node_encoder[0].weight.grad.abs().sum() > 0
        assert head.policy.critic.edge_bias[-1].weight.grad.abs().sum() > 0
        raw_before = head._raw_values(before.detach())
        head._normalized_value_targets(np.array([-1.,-2.,-4.], dtype=np.float32), np.ones(3,dtype=np.float32))
        raw_after = head._raw_values(head.policy(states, masks)[1]).detach()
        torch.testing.assert_close(raw_before,raw_after,atol=1e-6,rtol=1e-5)


def test_structured_replay_updates_and_checkpoint_resume(tmp_path):
    torch.manual_seed(72)
    env = tiny_replay_env(); env.reset(); agent = make_agent(env); slow = agent.slow_agent
    before = deepcopy(slow.placement_ppo.policy.state_dict())
    for index in range(3):
        slow.plan_deployment(env)
        slow.assign_pending_reward(-1.-index, done=True)
    metrics = slow.update()
    assert metrics['placement_optimizer_steps'] > 0
    assert any(not torch.equal(before[k],v) for k,v in slow.placement_ppo.policy.state_dict().items() if not k.startswith('critic.'))
    assert any(not torch.equal(before[k],v) for k,v in slow.placement_ppo.policy.state_dict().items() if k.startswith('critic.'))
    path = tmp_path/'reallocation.pt'; save_checkpoint(agent,path,{'update':1})
    restored = make_agent(env); load_checkpoint(restored,path)
    np.testing.assert_array_equal(slow.plan_deployment(env,deterministic=True,record=False),
        restored.slow_agent.plan_deployment(env,deterministic=True,record=False))
    assert not metrics['window_critic_active']
    diagnostics = slow.last_update_diagnostics
    assert diagnostics['placement_diagnostic_windows'] == 3
    assert diagnostics['placement_score_gradient_variance_ratio'] == pytest.approx(1.)  # First batch uses MC.
    assert diagnostics['placement_window_kl_proxy_mean'] >= 0
    assert diagnostics['placement_critic_update_time_s'] > 0
    with pytest.raises(ValueError,match='allocation'):
        load_checkpoint(make_agent(env,slow_allocation_mode='stage-budget-v5'),path)


def test_preset_resolves_new_schema_without_starting_training(monkeypatch):
    from train_dual_ppo import parse_args,apply_proposed_design,resolve_slow_quota_decode
    monkeypatch.setattr('sys.argv',['train_dual_ppo.py','--pressure-profile','mec-latency200','--proposed-design','joint-credit-v5'])
    args = apply_proposed_design(parse_args())
    assert args.slow_allocation_mode == 'stage-budget-v6'
    assert args.slow_learning_repair == 'prefix-structured-v3'
    assert args.slow_deployment_context == 'deployment-chain-v3'
    assert args.slow_critic_replay_windows == 24
    assert resolve_slow_quota_decode(args) == 'categorical-mean'


@pytest.mark.parametrize('source_allocation',['stage-budget-v3','stage-budget-v5'])
def test_full_actor_migration_to_reallocation_preserves_parameters(source_allocation):
    from edge_drl.agents.slow_transfer import copy_slow_actor
    from tests.test_joint_credit_v2 import make_agent as original_agent
    from tests.test_integer_slow_quota import make_agent as integer_agent
    env=tiny_replay_env();env.reset()
    source = original_agent(env).slow_agent if source_allocation=='stage-budget-v3' else integer_agent(env,slow_allocation_mode=source_allocation,slow_deployment_context='deployment-chain-v3').slow_agent
    target = make_agent(env).slow_agent
    copy_slow_actor(source,target)
    for name in ['count_ppo','stage_ppo','placement_ppo']:
        old,new = getattr(source,name),getattr(target,name)
        for key,value in old.policy.state_dict().items():
            if key.startswith('critic.'):
                continue
            copied = new.policy.state_dict()[key]
            if copied.shape == value.shape:
                torch.testing.assert_close(copied,value,atol=0,rtol=0)
            else:
                torch.testing.assert_close(copied[:,:value.shape[1]],value,atol=0,rtol=0)
                assert copied[:,value.shape[1]:].count_nonzero() == 0
        assert not new.optimizer.state


def test_history_is_previous_window_weighted_and_matches_training_evaluation(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from edge_drl.agents.slow_history import history_features
    from edge_drl.comparison.window_feedback import SlowWindowFeedbackMixin
    from edge_drl.comparison.scheme import BaseComparisonScheme
    env=tiny_replay_env();env.reset();agent=make_agent(env)
    class Probe(SlowWindowFeedbackMixin,BaseComparisonScheme):
        pass
    probe=Probe();probe.agent=SimpleNamespace(slow_agent=make_agent(env).slow_agent);probe._initialize_window_feedback()
    template=next(request for slot in env._comparison_trace.slots for request in slot)
    request=replace(template,service_id=1)
    for nodes,weight,compute,transport in [([0,1,2],10.,.02,.01),([0,2,3],30.,.04,.03)]:
        settlement=dict(stage_nodes=nodes,valid=True,request_count=weight,latency_s=.3,
            compute_delays={f'stage-{i}':compute for i in range(3)},
            link_delays={'ingress':transport,'stage-0':transport,'stage-1':transport})
        agent.observe_step_reward(-.3,3,False,weight=weight,latency_s=.3,
            record_fast=False,service_id=1,stage_nodes=nodes,slow_stage_costs=[.1]*3,settlement_info=settlement)
        probe.observe_step(env,[request],[],{'group_infos':[settlement]})
    assert not agent.slow_agent.last_route_history
    assert not probe.agent.slow_agent.last_route_history
    agent.flush_slow_window_reward(False,env);probe._finalize_window_feedback()
    assert agent.slow_agent.last_route_history == probe.agent.slow_agent.last_route_history
    node,stage=history_features(agent.slow_agent.last_route_history,6,2,3,1,1)
    np.testing.assert_array_equal(node[:,0],[0.,.25,.75,0.,0.,0.])
    assert node[1,3] == pytest.approx(np.log1p(20))
    assert stage[4,0] == pytest.approx(np.log1p(35))
    assert stage[4,1] == pytest.approx(np.log1p(25))
    path=tmp_path/'history.pt';save_checkpoint(agent,path,{})
    restored=make_agent(env);load_checkpoint(restored,path)
    assert restored.slow_agent.last_route_history == agent.slow_agent.last_route_history
    agent.reset_episode_context()
    assert not agent.slow_agent.last_route_history
    assert agent.window_route_history is None


def test_baseline_gradient_diagnostic_measures_its_stated_estimator_without_updates():
    from edge_drl.models.slow_diagnostics import baseline_diagnostics,score_gradient_diagnostics,window_policy_change
    env=tiny_replay_env();env.reset();slow=make_agent(env).slow_agent;slow.plan_deployment(env)
    head=slow.count_ppo
    states=np.stack(head.buffer.states[:3]);masks=np.stack(head.buffer.masks[:3])
    actions=np.asarray(head.buffer.actions[:3]);returns=np.asarray([-1.,-2.,-3.],dtype=np.float32)
    ids=np.arange(3);before=deepcopy(head.policy.state_dict())
    diagnostics=score_gradient_diagnostics(head,states,masks,actions,returns,returns,ids,1.)
    assert diagnostics['score_gradient_mc_variance'] > 0
    assert diagnostics['score_gradient_baseline_variance'] == 0
    assert diagnostics['score_gradient_variance_ratio'] == 0
    for key,value in head.policy.state_dict().items():
        torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    assert all(p.grad is None for p in head.policy.parameters())
    baseline=baseline_diagnostics(returns,returns,np.ones(3),ids,1.)
    assert baseline['baseline_residual_variance_ratio'] == 0
    changes=window_policy_change(head,states,masks,actions,np.asarray(head.buffer.logprobs[:3]),ids)
    assert changes['window_log_ratio_abs_max'] < 1e-6
    assert changes['window_kl_proxy_max'] < 1e-10


def test_old_continuous_count_buffer_roundtrip_preserves_latent_actions(tmp_path):
    from tests.test_joint_credit_v2 import make_agent as old_agent
    env=tiny_replay_env();env.reset();agent=old_agent(env)
    agent.slow_agent.plan_deployment(env);agent.slow_agent.assign_pending_reward(-2.,True)
    before=agent.slow_agent.count_ppo.buffer.actions.copy()
    assert any(0 < value < 1 for value in before)
    path=tmp_path/'continuous_pending.pt';save_checkpoint(agent,path,{})
    restored=old_agent(env);load_checkpoint(restored,path)
    np.testing.assert_array_equal(restored.slow_agent.count_ppo.buffer.actions,before)
