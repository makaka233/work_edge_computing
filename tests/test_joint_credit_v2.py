"""Joint repair: exact constraints, prefix credit, rollback and isolation."""
from argparse import Namespace
from copy import deepcopy
import csv
import sys
import subprocess

import numpy as np
import pytest
import torch

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.checkpoint_selection import checkpoint_probe, recent_workload_score
from edge_drl.models.guarded_chunks import grouped_chunks
from edge_drl.models.prefix_credit import prefix_advantages
from edge_drl.sampling import interval_seconds
from tests.comparison_helpers import tiny_replay_env
from train_dual_ppo import apply_proposed_design, append_log, save_checkpoint, load_checkpoint


def make_agent(env):
    return HierarchicalPPOAgent.from_env(env, replicas_per_stage=6,
        training_repair="batch-credit-v1", fast_chain_context="chain-kkt-v3",
        slow_allocation_mode="stage-budget-v3", slow_deployment_context="deployment-aware-v2",
        slow_learning_repair="prefix-credit-v2", slow_minibatch_size=8,
        slow_critic_replay_windows=12, slow_critic_holdout_windows=1, slow_critic_k_epochs=3,
        slow_k_epochs=2, slow_stage_k_epochs=2, slow_placement_k_epochs=2,
        fast_update_chunks=4, fast_independent_critic=True, fast_minibatch_size=8,
        slow_tail_latency_coef=0., slow_deployment_memory_coef=0.,
        slow_deployment_storage_coef=0., slow_deadline_violation_coef=0., slow_colocation_coef=0.)


def collect():
    torch.manual_seed(14)
    env = tiny_replay_env(); env.reset()
    agent = make_agent(env)
    for window in range(3):
        deployment = agent.slow_agent.plan_deployment(env)
        assert env.check_deployment_feasible(deployment)[0]
        env.apply_deployment(deployment)
        agent.slow_agent.assign_pending_reward(-1.-window, done=True)
    return env, agent


def test_stage_quotas_are_one_per_stage_and_all_coverage_is_legal():
    env, agent = collect()
    slow = agent.slow_agent
    assert len(slow.count_ppo.buffer.actions) == len(slow.stage_ppo.buffer.actions) == 15
    for window in range(3):
        indices = np.flatnonzero(np.asarray(slow.stage_window_ids) == window)
        selected = [slow.stage_ppo.buffer.actions[i] for i in indices]
        assert len(set(selected)) == len(selected)
    assert all(0 <= a <= 1 for a in slow.count_ppo.buffer.actions)
    assert all(env.deployment[s.service_id, :len(s.stages)].any(axis=1).all() for s in env.scenario.services)


def test_zero_and_full_quotas_are_representable_without_candidate_caps(monkeypatch):
    for fraction, expected in [(0., 5), (1., 30)]:
        env = tiny_replay_env(); env.reset(); agent = make_agent(env)
        monkeypatch.setattr(agent.slow_agent.count_ppo, "act", lambda *a, **k: (fraction, 0., 0.))
        deployment = agent.slow_agent.plan_deployment(env, deterministic=True, record=False)
        assert deployment.sum() == expected
        assert env.check_deployment_feasible(deployment)[0]


def test_prefix_lambda_one_is_exact_mc_and_windows_do_not_leak():
    returns = np.array([-2., -2., -5., -5., -5.])
    values = np.array([.4, -.5, -2., -3., -4.])
    ids = np.array([0, 0, 1, 1, 1])
    np.testing.assert_allclose(prefix_advantages(returns, values, ids, 1.), returns-values, atol=1e-6)
    np.testing.assert_allclose(prefix_advantages(returns, values, ids, 0.), [-.9, -1.5, -1., -1., -1.])
    result = prefix_advantages(returns, values, ids, .95)
    returns[2:] = -500.
    np.testing.assert_allclose(result[:2], prefix_advantages(returns, values, ids, .95)[:2])


def test_prefix_actor_uses_previous_reliability_and_true_returns(monkeypatch):
    _, agent = collect()
    slow = agent.slow_agent
    slow.update()
    for head in ["count", "stage", "placement"]:
        assert slow.last_update_diagnostics[head+"_prefix_trace_lambda"] == 1.
        ppo = getattr(slow, head+"_ppo")
        learner = ppo.slow_conditional_learning
        assert learner.prefix_credit
        assert {float(y) for x in learner.replay for y in x['returns']} == {-1., -2., -3.}
        assert np.isfinite(slow.last_update_diagnostics[head+"_actor_gradient_norm"])


def test_fast_independent_critic_cannot_update_actor():
    env = tiny_replay_env(); env.reset(); agent = make_agent(env)
    agent.maybe_update_deployment(env, record=False)
    requests = list(next(slot for slot in env._comparison_trace.slots if slot))
    env.current_requests = requests
    agent.fast_agent.schedule_batch(env, requests, record=True)
    ppo = agent.fast_agent.ppo
    states = torch.as_tensor(np.stack(ppo.buffer.states))
    masks = torch.as_tensor(np.stack(ppo.buffer.masks))
    _, values = ppo.policy(states, masks)
    values.square().mean().backward()
    assert all(p.grad is None or not bool(p.grad.any()) for n,p in ppo.policy.named_parameters()
               if not n.startswith("critic."))


def test_grouped_chunks_cover_every_group_once_per_epoch():
    groups = np.repeat([0, 1, 2], [9, 12, 7])
    chunks = grouped_chunks(len(groups), groups, 4)
    assert len(chunks) == 4
    assert sorted(np.concatenate(chunks)) == list(range(len(groups)))
    assert all(set(groups[c]) == {0,1,2} for c in chunks)
    assert len(grouped_chunks(5, [0,0,0,0,1], 4)) == 1


def test_fast_multiple_steps_and_rejected_epoch_rollback(monkeypatch):
    from tests.test_training_credit_repair import filled_guarded_ppo
    ppo = filled_guarded_ppo()
    ppo.guarded_update_chunks = 4
    before = deepcopy(ppo.policy.state_dict())
    moments = deepcopy(ppo.optimizer.state_dict())
    monkeypatch.setattr(ppo, "_buffer_policy_diagnostics",
                        lambda *a,**k: {"approx_kl":1., "max_group_approx_kl":1.})
    metrics = ppo.update()
    assert metrics["optimizer_steps"] == 0
    assert metrics["kl_early_stop"] == 1
    # Critic output is PopArt-rescaled before the epoch; Actor must be exact.
    for name,p in ppo.policy.state_dict().items():
        if not name.startswith("critic."):
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    assert not moments["state"]
    assert not ppo.optimizer.state  # No rejected Adam history left behind.


def test_fast_chunks_accept_multiple_steps_without_changing_kl_limit():
    from tests.test_training_credit_repair import filled_guarded_ppo
    ppo=filled_guarded_ppo()
    ppo.guarded_update_chunks=4
    for group in ppo.optimizer.param_groups:
        group['lr']=1e-3
    metrics=ppo.update()
    assert metrics['optimizer_steps'] > metrics['epochs_completed']
    assert metrics['max_group_approx_kl'] <= ppo.target_kl
    assert metrics['samples_seen_fraction'] == metrics['min_group_seen_fraction'] == 1.


def test_recent_score_ignores_ancient_policy_scale():
    recent = [{"latency_s":.1+.1*(i%5)/10, "load_multiplier":.8+(i%5)/10,
               "compute_gcycles":4.+(i%3)} for i in range(80)]
    ancient = [{"latency_s":1000., "load_multiplier":.7, "compute_gcycles":10.}]*200
    assert recent_workload_score(recent,10,1.,5.) == recent_workload_score(ancient+recent,10,1.,5.)


def test_probe_does_not_mutate_training_agent_or_random_streams():
    env = tiny_replay_env(); env.reset(); env.config.temporal_sampling_mode = "stratified-jitter"
    agent = make_agent(env)
    weights = deepcopy(agent.fast_agent.ppo.policy.state_dict())
    cpu = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    result = checkpoint_probe(agent, env.config, seed=2026, samples=2, hold_cap=0.)
    assert result["sampled_steps"] == 2
    assert np.isfinite(result["mean_latency_s"])
    assert torch.equal(cpu, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    for name,p in agent.fast_agent.ppo.policy.state_dict().items():
        torch.testing.assert_close(p, weights[name], rtol=0, atol=0)
    assert not agent.slow_agent.window_returns
    assert not agent.slow_agent.count_ppo.slow_conditional_learning.replay


def test_new_checkpoint_restores_credit_state_and_rejects_wrong_critic(tmp_path):
    env, agent = collect(); agent.slow_agent.update()
    path=tmp_path/'latest.pt'; save_checkpoint(agent,path,{})
    restored=make_agent(env); load_checkpoint(restored,path)
    assert restored.slow_agent.count_ppo.slow_conditional_learning.prefix_credit
    assert restored.slow_agent.count_ppo.slow_conditional_learning.trace_reliability == agent.slow_agent.count_ppo.slow_conditional_learning.trace_reliability


def test_csv_append_preserves_old_header_and_exports_new_fields(tmp_path):
    path=tmp_path/'training.csv'
    path.write_text('update,latency\n1,.2\n',encoding='utf-8')
    append_log(path, {'latency':.1,'sampled_request_events':3,'update':2})
    with path.open() as h:
        rows=list(csv.DictReader(h))
    assert rows[-1] == {'update':'2','latency':'0.1'}
    assert 'sampled_request_events' in (tmp_path/'training_extensions_v2.csv').read_text()


def test_new_preset_is_opt_in_and_uses_mean_objective(monkeypatch):
    monkeypatch.setattr(sys,'argv',['train_dual_ppo.py'])
    args=apply_proposed_design(Namespace(proposed_design='auto',load_checkpoint=None))
    assert args.proposed_design == 'legacy'
    args=apply_proposed_design(Namespace(proposed_design='joint-credit-v2',load_checkpoint=None))
    assert args.slow_allocation_mode == 'stage-budget-v3'
    assert args.slow_tail_latency_coef == args.slow_deployment_memory_coef == 0
    assert args.fast_update_chunks == 4 and args.fast_independent_critic


def test_sampling_cap_is_shared_and_actual_events_are_unweighted():
    assert interval_seconds(600,6,15) == 15
    assert interval_seconds(600,6,100) == 100
    env=tiny_replay_env(); env.reset()
    agent=make_agent(env); agent.maybe_update_deployment(env,record=False)
    requests=env.current_requests
    paths=agent.fast_agent.schedule_batch(env,requests,record=False)
    env.step(paths,represented_seconds=100)
    assert env.metrics['sampled_request_events'] == len(requests)
    assert env.metrics['request_events'] == len(requests)*100


def test_competition_observation_matches_its_own_full_cost_difference():
    from edge_drl.agents.slow_context import SlowDeploymentContext
    env=tiny_replay_env(); env.reset()
    deployment=np.zeros_like(env.deployment)
    for service in env.scenario.services:
        deployment[service.service_id,:len(service.stages),0]=True
    context=SlowDeploymentContext(env,None,competition_aware=True)
    _,pool,before=context.inventory(deployment)
    gains=context.candidate_gains(deployment,pool,before)
    after=deployment.copy(); after[0,0,2]=True
    after_cost=context.inventory(after)[2]
    assert gains[0,2] == pytest.approx(max(float(before.sum()-after_cost.sum()),0.),abs=1e-9)
    # Moving another service in the SAME serial phase changes stage 0's cost.
    moved=deployment.copy(); moved[1,0,0]=False; moved[1,0,1]=True
    assert context.inventory(moved)[2][0] < before[0]


def test_explicit_actor_transfer_preserves_actor_and_resets_changed_critic(tmp_path):
    from tests.test_training_credit_repair import repaired_agent
    from train_dual_ppo import initialize_fast_checkpoint
    env=tiny_replay_env(); env.reset()
    source=repaired_agent(env)
    args=Namespace(seed=10,physical_seed=10,training_repair='batch-credit-v1',
                   service_calibration_document=None,init_fast_actor_only=True)
    path=tmp_path/'source.pt';save_checkpoint(source,path,{'args':vars(args)})
    destination=make_agent(env)
    critic=deepcopy(destination.fast_agent.ppo.policy.critic.state_dict())
    result=initialize_fast_checkpoint(destination,path,args)
    assert result['scope']=='fast_actor_only_new_critic'
    for name,p in destination.fast_agent.ppo.policy.state_dict().items():
        if not name.startswith('critic.'):
            torch.testing.assert_close(p,source.fast_agent.ppo.policy.state_dict()[name],rtol=0,atol=0)
    for name,p in destination.fast_agent.ppo.policy.critic.state_dict().items():
        torch.testing.assert_close(p,critic[name],rtol=0,atol=0)
    assert not destination.fast_agent.ppo.optimizer.state


def test_comparison_training_protocol_honors_cap_and_samples_same_seconds(monkeypatch):
    from edge_drl.comparison.metrics import evaluate_scheme_episode
    from edge_drl.comparison.types import ExperimentPoint
    from tests.test_comparison_runner import FixedScheme
    samples=[]
    for _ in range(2):
        env=tiny_replay_env();env.config.temporal_sampling_mode='stratified-jitter'
        original=env.sample_interval_requests
        selected=[]
        def sample(dt):
            original(dt)
            selected.append([r.request_id for r in env.current_requests])
        monkeypatch.setattr(env,'sample_interval_requests',sample)
        result=evaluate_scheme_episode(scheme=FixedScheme(),env=env,
            point=ExperimentPoint('request_load',1.,'nominal'),eval_seed=2026,routing_repeat=0,
            sampled_seconds_per_window=6,sampling_protocol='training-equivalent-v2',
            sampling_max_interval_seconds=15)
        assert result.settlement_steps == 40
        assert len(selected)==40
        samples.append(selected)
    assert samples[0]==samples[1]


def test_real_entrypoint_joint_updates_probe_and_same_directory_resume(tmp_path):
    command=[sys.executable,'train_dual_ppo.py','--pressure-profile','mec-latency200',
        '--proposed-design','joint-credit-v2','--training-design','trajectory-simultaneous',
        '--updates','2','--num-users','10000','--num-edge-nodes','6','--num-service-types','2',
        '--topology-k-nearest','2',
        '--mean-requests-per-minute','600','--episode-minutes','60',
        '--deployment-interval-minutes','10','--fast-windows-per-update','6',
        '--slow-windows-per-update','6','--sampled-seconds-per-window','1',
        '--sampling-max-interval-seconds','600','--replicas-per-stage','0',
        '--fast-k-epochs','2','--slow-k-epochs','2','--slow-stage-k-epochs','2',
        '--slow-placement-k-epochs','2','--slow-critic-k-epochs','2',
        '--slow-critic-replay-windows','12','--slow-critic-holdout-windows','1',
        '--slow-critic-holdout-episodes','1','--checkpoint-probe-interval','1',
        '--eval-interval','0','--device','cpu','--progress-interval-seconds','0',
        '--run-root',str(tmp_path),'--run-name','joint']
    completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
    assert completed.returncode==0, completed.stdout+'\n'+completed.stderr
    run=tmp_path/'joint'
    best=torch.load(run/'checkpoints/best.pt',map_location='cpu')
    assert best['metadata']['selection_kind']=='post_update_nominal_probe_v2'
    assert best['fast_independent_critic']
    assert best['slow_allocation_mode']=='stage-budget-v3'
    assert best['metadata']['probe_result']['sampled_steps']==1
    command[command.index('--updates')+1]='1'
    command[command.index('--proposed-design')+1]='auto'
    command += ['--load-checkpoint',str(run/'checkpoints/latest.pt'),'--append-log']
    completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
    assert completed.returncode==0, completed.stdout+'\n'+completed.stderr
    with (run/'logs/training.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert [int(row['update']) for row in rows]==[1,2,3]
    assert all(int(row['slow_updated'])==1 and float(row['fast_optimizer_steps'])>0 for row in rows)
    assert all(float(row['sampled_request_events']) < float(row['request_events']) for row in rows)
    latest=torch.load(run/'checkpoints/latest.pt',map_location='cpu')
    assert latest['metadata']['completed_updates']==3
    assert latest['metadata']['args']['proposed_design']=='joint-credit-v2'
