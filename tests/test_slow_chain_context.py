from copy import deepcopy

import numpy as np
import pytest
import torch

from edge_drl.agents.slow_chain_context import SlowChainContext, CHAIN_NODE_DIM, signed_log_ms
from tests.comparison_helpers import tiny_replay_env
from tests.test_integer_slow_quota import make_agent
from train_dual_ppo import save_checkpoint, load_checkpoint


def covered(env):
    deployment = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        for stage in service.stages:
            deployment[service.service_id, stage.stage_id, (service.service_id + stage.stage_id) % 6] = True
    return deployment


def test_chain_costs_preserve_direction_and_only_use_existing_neighbors():
    env = tiny_replay_env(); env.reset()
    context = SlowChainContext(env, None)
    context.propagation = np.arange(36, dtype=float).reshape(6, 6) / 1000
    context.inverse_bandwidth.fill(0.)
    deployment = covered(env)
    upstream, downstream, home = context._chain_costs(deployment, 1, 1)
    np.testing.assert_allclose(upstream, context.propagation[1])
    np.testing.assert_allclose(downstream, context.propagation[:, 3])
    np.testing.assert_allclose(home, context.home_weights[:, 1] @ context.propagation)
    deployment[1, 2] = False
    np.testing.assert_array_equal(context._chain_costs(deployment, 1, 1)[1], np.zeros(6))
    np.testing.assert_array_equal(context._chain_costs(deployment, 0, 1)[1], np.zeros(6))


def test_signed_compute_proxy_matches_full_surrogate_change_including_externality():
    env = tiny_replay_env(); env.reset()
    deployment = covered(env)
    # Concentrate the competing service on the candidate under examination.
    deployment[1, 0] = False; deployment[1, 0, 5] = True
    context = SlowChainContext(env, None)
    _, pooled, before = context.inventory(deployment)
    gains = context.candidate_gains(deployment, pooled, before, clip=False)
    for service in env.scenario.services:
        for stage in service.stages:
            i = service.service_id * context.stages + stage.stage_id
            for node in np.flatnonzero(~deployment[service.service_id, stage.stage_id]):
                after = deployment.copy(); after[service.service_id, stage.stage_id, node] = True
                assert gains[i, node] == pytest.approx(float(before.sum() - context.inventory(after)[2].sum()), abs=1e-9)
    assert signed_log_ms(-.01) < 0 < signed_log_ms(.01)


def test_opportunity_feature_counts_lost_other_stage_positions():
    env = tiny_replay_env(); env.reset()
    context = SlowChainContext(env, None)
    deployment = covered(env)
    # Give one node only enough RAM for one additional stage.
    selected = deployment.reshape(context.size, 6)
    used = selected.T @ context.memory
    context.memory_capacity[:] = used + 1.1 * context.memory[0]
    memory = context.memory_capacity - used
    storage = context.storage_capacity - selected.T @ context.storage
    expected = np.zeros(6)
    for node in range(6):
        for other in np.flatnonzero(context.valid):
            if other == 0 or selected[other, node]:
                continue
            fits = memory[node] + 1e-9 >= context.memory[other] and storage[node] + 1e-9 >= context.storage[other]
            fits_after = (memory[node] - context.memory[0] + 1e-9 >= context.memory[other]
                          and storage[node] - context.storage[0] + 1e-9 >= context.storage[other])
            expected[node] += int(fits and not fits_after)
    expected /= int(context.valid.sum()) - 1
    np.testing.assert_allclose(context._opportunity_loss(deployment, 0), expected)


def test_chain_inputs_are_causal_and_additive_without_changing_old_features():
    from edge_drl.agents.slow_context import SlowDeploymentContext
    env = tiny_replay_env(); env.reset()
    deployment = covered(env)
    old = SlowDeploymentContext(env, None, competition_aware=True)
    new = SlowChainContext(env, None)
    kwargs = dict(budget=np.zeros(6), service_id=1, stage_id=1)
    globals_old, nodes_old = old.features(deployment, **kwargs)
    globals_new, nodes_new = new.features(deployment, **kwargs)
    np.testing.assert_array_equal(globals_old, globals_new)
    np.testing.assert_array_equal(nodes_old, nodes_new[:, :-CHAIN_NODE_DIM])
    # A different current/future request batch does not enter this observation.
    env.current_requests = []
    after = SlowChainContext(env, None).features(deployment, **kwargs)
    np.testing.assert_array_equal(after[1], nodes_new)
    assert np.isfinite(nodes_new).all()
    assert ((new.port_headroom >= 0) & (new.port_headroom <= 1)).all()


def test_chain_actions_returns_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(4)
    env = tiny_replay_env(); env.reset()
    agent = make_agent(env, slow_deployment_context='deployment-chain-v3')
    slow = agent.slow_agent
    for index in range(3):
        deployment = slow.plan_deployment(env)
        assert env.check_deployment_feasible(deployment)[0]
        slow.assign_pending_reward(-1. - index, done=True)
    for head in (slow.count_ppo, slow.stage_ppo, slow.placement_ppo):
        assert all(len(state) == head.policy.edge_offset + 6*6*head.policy.edge_feature_dim for state in head.buffer.states)
        assert all(np.isfinite(state).all() for state in head.buffer.states)
    assert slow.update()['placement_optimizer_steps'] > 0
    path = tmp_path/'chain.pt'; save_checkpoint(agent, path, {'update': 1})
    restored = make_agent(env, slow_deployment_context='deployment-chain-v3')
    load_checkpoint(restored, path)
    np.testing.assert_array_equal(slow.plan_deployment(env, deterministic=True, record=False),
                                  restored.slow_agent.plan_deployment(env, deterministic=True, record=False))
    with pytest.raises(ValueError, match='context'):
        load_checkpoint(make_agent(env), path)


def test_actor_transfer_preserves_all_head_probabilities_and_new_columns_learn():
    from edge_drl.agents.slow_transfer import copy_slow_actor
    torch.manual_seed(73)
    env = tiny_replay_env(); env.reset()
    source = make_agent(env).slow_agent
    target = make_agent(env, slow_deployment_context='deployment-chain-v3').slow_agent
    critics = {name: deepcopy(getattr(target, name).policy.critic.state_dict())
               for name in ('count_ppo', 'stage_ppo', 'placement_ppo')}
    source.plan_deployment(env)
    copy_slow_actor(source, target)
    for name in critics:
        original, new = getattr(source, name), getattr(target, name)
        policy = original.policy
        states = np.stack(original.buffer.states[:4])
        masks = torch.from_numpy(np.stack(original.buffer.masks[:4]))
        nodes = states[:, policy.node_offset:policy.stage_offset].reshape(-1, 6, policy.node_feature_dim)
        # Even nonzero new inputs must initially contribute exactly zero.
        expanded_nodes = np.concatenate([nodes, np.ones((len(states), 6, CHAIN_NODE_DIM))], axis=-1)
        pieces = [states[:, :policy.node_offset], expanded_nodes.reshape(len(states), -1)]
        if policy.action_mode == 'stage':
            stages = states[:, policy.stage_offset:policy.edge_offset].reshape(-1, policy.stage_action_count, policy.stage_feature_dim)
            expanded_stages = np.concatenate([stages, np.ones((*stages.shape[:2], 4))], axis=-1)
            pieces.append(expanded_stages.reshape(len(states), -1))
        pieces.append(states[:, policy.edge_offset:])
        expanded = torch.tensor(np.concatenate(pieces, axis=1), dtype=torch.float32)
        before = original.policy(torch.from_numpy(states), masks)[0].probs
        after = new.policy(expanded, masks)[0].probs
        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)
        for key, value in new.policy.critic.state_dict().items():
            torch.testing.assert_close(value, critics[name][key], rtol=0, atol=0)
        assert not new.optimizer.state
        actions = torch.from_numpy(np.array(original.buffer.actions[:4]))
        loss = -new.policy(expanded, masks)[0].log_prob(actions).mean()
        loss.backward()
        assert new.policy.node_encoder[0].weight.grad[:, -CHAIN_NODE_DIM:].abs().sum() > 0
    with pytest.raises(ValueError, match='discard'):
        copy_slow_actor(target, source)


def test_inference_load_does_not_restore_training_rng_but_full_resume_does(tmp_path):
    env = tiny_replay_env(); env.reset(); agent = make_agent(env)
    path = tmp_path/'rng.pt'
    np.random.seed(712); torch.manual_seed(712)
    source_numpy, source_torch = np.random.get_state(), torch.get_rng_state().clone()
    save_checkpoint(agent, path, {})
    np.random.seed(929); torch.manual_seed(929)
    current_numpy, current_torch = np.random.get_state(), torch.get_rng_state().clone()
    load_checkpoint(agent, path, restore_rng=False)
    np.testing.assert_array_equal(np.random.get_state()[1], current_numpy[1])
    assert np.random.get_state()[2:] == current_numpy[2:]
    torch.testing.assert_close(torch.get_rng_state(), current_torch, rtol=0, atol=0)
    load_checkpoint(agent, path)
    np.testing.assert_array_equal(np.random.get_state()[1], source_numpy[1])
    assert np.random.get_state()[2:] == source_numpy[2:]
    torch.testing.assert_close(torch.get_rng_state(), source_torch, rtol=0, atol=0)
