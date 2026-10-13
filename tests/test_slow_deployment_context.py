"""Regression for the Slow observation/conditional-baseline audit findings."""
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pytest
import torch

from edge_drl.agents.drl import HierarchicalPPOAgent
from train_dual_ppo import (save_checkpoint, load_checkpoint, initialize_fast_checkpoint,
                            resolve_slow_deployment_context, rollout, SlowLearningRateController)
from tests.comparison_helpers import tiny_replay_env
from edge_drl.agents.slow_context import SlowDeploymentContext


def make_agent(env, context="deployment-aware-v2"):
    return HierarchicalPPOAgent.from_env(
        env, replicas_per_stage=6, training_repair="batch-credit-v1",
        fast_chain_context="chain-kkt-v3", slow_allocation_mode="budget-continuous-v2",
        slow_deployment_context=context, slow_k_epochs=1, slow_stage_k_epochs=1,
        slow_placement_k_epochs=1, slow_critic_k_epochs=4, fast_k_epochs=1,
    )


def test_old_stage_observation_alias_is_fixed():
    env = tiny_replay_env(); env.reset()
    deployment = np.zeros_like(env.deployment)
    deployment[0, 0, 0] = deployment[0, 1, 5] = True
    deployment[1, :3, 2] = True
    swapped = deployment.copy()
    swapped[0, 0], swapped[0, 1] = deployment[0, 1], deployment[0, 0]
    for context in ("legacy", "deployment-aware-v2"):
        slow = make_agent(env, context).slow_agent
        demand = slow._node_service_demand(env)
        remaining_memory = env.service_memory_capacities() - deployment.sum(axis=(0, 1))
        remaining_storage = env.service_storage_capacities() - 2 * deployment.sum(axis=(0, 1))
        states = [slow._build_incremental_state(env, demand, remaining_memory, remaining_storage, d)
                  for d in (deployment, swapped)]
        # Same total budget, replica counts, overlap and all node resource columns.
        assert np.array_equal(*states) == (context == "legacy")


def test_budget_and_stage_compute_scale_are_observable():
    env = tiny_replay_env(); env.reset()
    slow = make_agent(env).slow_agent
    demand = slow._node_service_demand(env)
    memory, storage = env.service_memory_capacities(), env.service_storage_capacities()
    before = slow._build_incremental_state(env, demand, memory, storage, env.deployment)
    slow.pending_budget_fraction = .6
    slow.pending_budget_target_memory_gb = 10
    slow.pending_budget_used_memory_gb = 3
    after = slow._build_incremental_state(env, demand, memory, storage, env.deployment)
    assert not np.array_equal(before, after)
    env.config.task_compute_scale = 2
    scaled = slow._build_incremental_state(env, demand, memory, storage, env.deployment)
    assert not np.array_equal(after, scaled)


def test_physical_proxy_uses_real_user_mass_not_log_scaled_observation():
    env=tiny_replay_env(); env.reset()
    slow=make_agent(env).slow_agent
    transformed=slow._node_service_demand(env)
    context=SlowDeploymentContext(env, transformed)
    np.testing.assert_allclose(context.mass[::3], [0.6,0.4])
    assert context.mass[::3].sum() == pytest.approx(1.0)
    assert not np.allclose(context.mass[::3], transformed.sum(axis=0))
    assert context.arrival * context.mass[0] * context.compute[0] == pytest.approx(env._arrival_rate_per_minute()/60 * .6 * .8)


@pytest.mark.parametrize("context", ["legacy", "deployment-aware-v2"])
def test_stage_critic_reads_stage_features_only_in_new_architecture(context):
    env = tiny_replay_env(); env.reset()
    policy = make_agent(env, context).slow_agent.stage_ppo.policy
    width = policy.edge_offset + 6 * 6 * policy.edge_feature_dim
    states = torch.rand(2, width, requires_grad=True)
    mask = torch.ones((2, policy.action_dim), dtype=torch.bool)
    values = policy(states, mask)[1]
    gradient = torch.autograd.grad(values.sum(), states, allow_unused=True, retain_graph=True)[0]
    has_stage_gradient = gradient is not None and bool(torch.count_nonzero(gradient[:, policy.stage_offset:policy.edge_offset]))
    assert has_stage_gradient == (context == "deployment-aware-v2")
    values.sum().backward()
    for name, parameter in policy.named_parameters():
        if not name.startswith("critic."):
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_value_only_training_preserves_actor_probabilities_and_adam():
    env = tiny_replay_env(); env.reset()
    slow = make_agent(env).slow_agent
    for reward in (-1.0, -2.0):
        env.apply_deployment(slow.plan_deployment(env))
        slow.assign_pending_reward(reward, True)
    ppo = slow.stage_ppo
    states = torch.as_tensor(np.stack(ppo.buffer.states))
    masks = torch.as_tensor(np.stack(ppo.buffer.masks))
    probabilities = ppo.policy(states, masks)[0].probs.detach().clone()
    actor_weights = {n:p.detach().clone() for n,p in ppo.policy.named_parameters() if not n.startswith("critic.")}
    targets = np.asarray([-1.0, -2.0])[slow.stage_window_ids]
    ppo._fit_conditional_critic(targets, np.ones(len(targets)))
    torch.testing.assert_close(probabilities, ppo.policy(states, masks)[0].probs)
    for name, parameter in ppo.policy.named_parameters():
        if name in actor_weights:
            assert torch.equal(parameter, actor_weights[name])
            assert parameter not in ppo.optimizer.state


def test_team_returns_and_critic_window_weights_and_full_batch_coverage(monkeypatch):
    env = tiny_replay_env(); env.reset()
    slow = make_agent(env).slow_agent
    for reward in (-1.0, -2.0):
        env.apply_deployment(slow.plan_deployment(env))
        slow.assign_pending_reward(reward, True)
    for ppo, ids in ((slow.count_ppo, slow.count_window_ids), (slow.stage_ppo, slow.stage_window_ids),
                     (slow.placement_ppo, slow.placement_window_ids)):
        original = ppo.update_from_returns
        def check(returns, *, _original=original, _ids=np.asarray(ids), **kwargs):
            np.testing.assert_allclose(returns, np.asarray([-1.0,-2.0])[_ids])
            weights = kwargs["critic_sample_weights"]
            assert weights[_ids==0].sum() == pytest.approx(weights[_ids==1].sum())
            result = _original(returns, **kwargs)
            assert result["samples_seen_fraction"] == 1
            assert result["approx_kl"] <= ppo.target_kl
            assert result["critic_extra_optimizer_steps"] > 0
            return result
        monkeypatch.setattr(ppo,"update_from_returns",check)
    slow.update()


def test_new_checkpoint_roundtrip_and_old_architecture_auto_resume(tmp_path):
    env = tiny_replay_env(); env.reset()
    source = make_agent(env)
    rollout(env, source, max_requests=6, train_mode="joint", reward_scale=10)
    source.flush_slow_window_reward(done=True, env=env); source.update()
    path = tmp_path / "new.pt"
    save_checkpoint(source,path,{"args":{"slow_deployment_context":"deployment-aware-v2"}})
    restored = make_agent(env); load_checkpoint(restored,path)
    for a,b in zip(source.slow_agent.count_ppo.policy.parameters(),restored.slow_agent.count_ppo.policy.parameters()):
        assert torch.equal(a,b)
    assert restored.slow_agent.count_ppo.value_normalization_state() == source.slow_agent.count_ppo.value_normalization_state()
    assert len(restored.slow_agent.stage_ppo.optimizer.param_groups)==2
    with pytest.raises(ValueError,match="deployment context"):
        load_checkpoint(make_agent(env,"legacy"),path)
    args=Namespace(load_checkpoint=str(path),slow_deployment_context="auto")
    assert resolve_slow_deployment_context(args)=="deployment-aware-v2"
    old=tmp_path / "old.pt"; save_checkpoint(make_agent(env,"legacy"),old,{})
    args.load_checkpoint=str(old)
    assert resolve_slow_deployment_context(args)=="legacy"


def test_initialize_fast_only_is_not_resume_and_rejects_environment_change(tmp_path):
    env=tiny_replay_env(); env.reset()
    source=make_agent(env,"legacy")
    source.fast_agent.ppo._normalized_value_targets(np.array([-4.,-2.]),np.ones(2))
    args=Namespace(seed=10,physical_seed=10,fast_policy_kind="gat_node_scorer",fast_chain_context="chain-kkt-v3",
                   training_repair="batch-credit-v1",service_calibration_document=None,wireless_uplink_mbps=150.)
    path=tmp_path / "fast.pt"; save_checkpoint(source,path,{"args":vars(args)})
    destination=make_agent(env)
    slow_before=deepcopy(destination.slow_agent.stage_ppo.policy.state_dict())
    rng_before=torch.get_rng_state().clone()
    details=initialize_fast_checkpoint(destination,path,args)
    assert not details["resume"]
    assert torch.equal(torch.get_rng_state(),rng_before)
    assert not destination.fast_agent.ppo.optimizer.state
    for name,value in slow_before.items():assert torch.equal(value,destination.slow_agent.stage_ppo.policy.state_dict()[name])
    for name,value in source.fast_agent.ppo.policy.state_dict().items():assert torch.equal(value,destination.fast_agent.ppo.policy.state_dict()[name])
    assert destination.fast_agent.ppo.value_normalization_state()==source.fast_agent.ppo.value_normalization_state()
    args.wireless_uplink_mbps=300
    with pytest.raises(ValueError,match="configuration conflict"):
        initialize_fast_checkpoint(destination,path,args)


def test_count_decay_cannot_slow_stage_or_conditional_critics():
    env=tiny_replay_env();env.reset()
    agent=make_agent(env)
    controller=SlowLearningRateController(enabled=True,patience=1,factor=.7,min_delta=0,
        min_lr=1e-6,independent_stage=True)
    kwargs=dict(ready=True,slow_updated=True,agent=agent,count_approx_kl=.01,
                stage_approx_kl=0.,placement_approx_kl=0.,count_target_kl=.015,placement_target_kl=.015)
    controller.observe(.2,**kwargs)
    before=agent.slow_agent.stage_ppo.optimizer.param_groups[0]['lr']
    count_before=agent.slow_agent.count_ppo.optimizer.param_groups[0]['lr']
    assert controller.observe(.3,**kwargs)
    assert agent.slow_agent.stage_ppo.optimizer.param_groups[0]['lr']==before
    assert agent.slow_agent.count_ppo.optimizer.param_groups[0]['lr'] == pytest.approx(.7 * count_before)
    assert agent.slow_agent.count_ppo.optimizer.param_groups[1]['lr'] == .0005
