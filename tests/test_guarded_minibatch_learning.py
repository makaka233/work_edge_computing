"""Learning-signal regressions, not evidence of deployment convergence."""
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pytest
import torch

from edge_drl.models.ppo import PPOAgent
from edge_drl.agents.drl import HierarchicalPPOAgent
from tests.comparison_helpers import tiny_replay_env
from train_dual_ppo import save_checkpoint, load_checkpoint, resolve_ppo_update_design, rollout


def make_ppo(lr=1e-3, kl=.015):
    return PPOAgent(obs_dim=4, action_dim=2, hidden_dim=32, lr=lr,
                    k_epochs=3, minibatch_size=32, entropy_coef=0, value_coef=0,
                    target_kl=kl, full_batch_kl_stop=True, guarded_epochs=True,
                    guarded_minibatch_epochs=True)


def fill(agent, n=256):
    for i in range(n):
        state = np.array([1, i / n, 0, -1], np.float32)
        mask = np.ones(2, bool)
        a, lp, v = agent.act(state, mask)
        agent.buffer.add(state=state, mask=mask, action=a, logprob=lp,
                         value=v, reward=float(a), done=True, sample_group=float(i % 4))


def test_minibatches_are_real_steps_and_all_groups_have_guarded_coverage():
    torch.manual_seed(35); np.random.seed(35)
    agent = make_ppo()
    fill(agent)
    metrics = agent.update()
    assert metrics["optimizer_steps"] > metrics["epochs_completed"]
    assert metrics["samples_seen_fraction"] == metrics["min_group_seen_fraction"] == 1
    assert metrics["exact_kl"] <= .015
    assert metrics["max_group_exact_kl"] <= .015
    assert metrics["max_group_approx_kl"] <= .015
    assert metrics["actor_grad_norm"] > 0


def test_known_reward_preference_learns_not_just_parameter_motion():
    torch.manual_seed(17); np.random.seed(17)
    agent = make_ppo()
    probe = torch.tensor([[1., .5, 0, -1]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    before = agent.policy(probe, mask)[0].probs[0, 1].item()
    for _ in range(8):
        fill(agent)
        agent.update()
    after = agent.policy(probe, mask)[0].probs[0, 1].item()
    assert after > before + .2


def test_fast_gat_learns_node_cost_preference_with_rotating_best_node():
    torch.manual_seed(47); np.random.seed(47)
    agent = PPOAgent(obs_dim=22, action_dim=4, hidden_dim=32, lr=2e-3,
                     global_dim=2, node_feature_dim=1, edge_feature_dim=1, num_nodes=4,
                     policy_kind="gat_node_scorer", detach_critic_backbone=True,
                     independent_critic=True, k_epochs=2, minibatch_size=32,
                     guarded_epochs=True, guarded_minibatch_epochs=True, full_batch_kl_stop=True,
                     target_kl=.015, entropy_coef=0, value_coef=0)
    probes = np.array([np.r_[.4, 1., np.where(np.arange(4) == best, .1, 1.), np.ones(16)]
                       for best in range(4)], dtype=np.float32)
    mask = torch.ones((4, 4), dtype=torch.bool)
    def best_prob():
        with torch.no_grad():
            return agent.policy(torch.as_tensor(probes), mask)[0].probs.diagonal().mean().item()
    before = best_prob()
    for _ in range(10):
        for i in range(128):
            state = probes[i % 4]
            valid = np.ones(4, bool)
            a, lp, v = agent.act(state, valid)
            agent.buffer.add(state=state, mask=valid, action=a, logprob=lp, value=v,
                             reward=-float(state[2 + a]), done=True, sample_group=float(i % 2))
        agent.update()
    assert best_prob() > before + .15


def test_rejected_whole_epoch_restores_nonempty_adam_and_all_parameters(monkeypatch):
    torch.manual_seed(17); np.random.seed(17)
    agent = make_ppo()
    fill(agent); agent.update()  # ensure moments are populated before rollback
    fill(agent)
    parameters = deepcopy(agent.policy.state_dict())
    moments = deepcopy(agent.optimizer.state_dict())
    monkeypatch.setattr(agent, "_exact_buffer_kl", lambda *a: {"exact_kl": 1., "max_group_exact_kl": 1.})
    metrics = agent.update()
    assert metrics["optimizer_steps"] == metrics["samples_seen_fraction"] == 0
    assert metrics["kl_backtracks"] == 7
    for name, value in agent.policy.state_dict().items():
        assert torch.equal(value, parameters[name])
    current = agent.optimizer.state_dict()
    assert moments["param_groups"] == current["param_groups"]
    for p, entries in moments["state"].items():
        for name, tensor in entries.items():
            assert torch.equal(tensor, current["state"][p][name])


def test_nonfinite_step_is_retried_before_next_distribution_forward(monkeypatch):
    torch.manual_seed(17); np.random.seed(17)
    agent = make_ppo()
    fill(agent)
    step = agent.optimizer.step
    attempts = [0]
    def corrupt_first_three_steps(*a, **kw):
        result = step(*a, **kw)
        if attempts[0] < 3:
            with torch.no_grad():
                next(agent.policy.parameters()).fill_(float("nan"))
        attempts[0] += 1
        return result
    monkeypatch.setattr(agent.optimizer, "step", corrupt_first_three_steps)
    metrics = agent.update()
    assert metrics["kl_backtracks"] >= 3
    assert metrics["samples_seen_fraction"] == 1
    assert agent._policy_parameters_finite()


def new_agent(env):
    return HierarchicalPPOAgent.from_env(
        env, replicas_per_stage=6, training_repair="batch-credit-v1", fast_chain_context="chain-kkt-v3",
        slow_allocation_mode="budget-continuous-v2", slow_deployment_context="deployment-aware-v2",
        ppo_update_design="guarded-minibatch-v2", slow_k_epochs=1, slow_stage_k_epochs=1,
        slow_placement_k_epochs=1, fast_k_epochs=2, fast_minibatch_size=8,
    )


def test_fast_baseline_reads_raw_predecision_state_without_actor_gradients():
    env = tiny_replay_env(); env.reset()
    ppo = new_agent(env).fast_agent.ppo
    policy = ppo.policy
    states = torch.rand((2, policy.edge_offset + 6 * 6 * policy.edge_feature_dim), requires_grad=True)
    masks = torch.ones((2, 6), dtype=torch.bool)
    values = policy(states, masks)[1]
    values.sum().backward()
    assert torch.count_nonzero(states.grad[:, :policy.global_dim]) > 0
    for name, p in policy.named_parameters():
        if not name.startswith("critic."):
            assert p.grad is None or torch.count_nonzero(p.grad) == 0
    assert ppo.optimizer.param_groups[1]["role"] == "critic"


def test_all_heads_update_and_new_checkpoint_loads_in_normal_evaluator(tmp_path):
    from edge_drl.comparison.checkpoint import build_proposed_agent
    env = tiny_replay_env(); env.reset()
    agent = new_agent(env)
    stats = rollout(env, agent, max_requests=10, reward_scale=10)
    assert stats["max_fast_kkt_reward_residual_s"] < 1e-8
    agent.flush_slow_window_reward(done=True, env=env)
    metrics = agent.update()
    assert metrics["fast"]["optimizer_steps"] > 2
    assert metrics["fast"]["critic_extra_optimizer_steps"] == 4
    assert all(agent.slow_agent.last_update_diagnostics[h + "exact_kl"] <= .015
               for h in ("count_", "stage_", "placement_"))
    configuration = dict(replicas_per_stage=6, training_repair="batch-credit-v1", fast_chain_context="chain-kkt-v3",
                         slow_allocation_mode="budget-continuous-v2", slow_deployment_context="deployment-aware-v2",
                         ppo_update_design="guarded-minibatch-v2")
    path = tmp_path / "new.pt"
    save_checkpoint(agent, path, {})
    restored, _ = build_proposed_agent(env, path, configuration, device="cpu")
    for name, value in restored.fast_agent.ppo.policy.state_dict().items():
        assert torch.equal(value, agent.fast_agent.ppo.policy.state_dict()[name])
    assert resolve_ppo_update_design(Namespace(ppo_update_design="auto", load_checkpoint=str(path))) == "guarded-minibatch-v2"
    old = HierarchicalPPOAgent.from_env(env, training_repair="batch-credit-v1", fast_chain_context="chain-kkt-v3",
                                       slow_allocation_mode="budget-continuous-v2", slow_deployment_context="deployment-aware-v2")
    with pytest.raises(ValueError, match="PPO update design"):
        load_checkpoint(old, path)
    old_path = tmp_path / "old.pt"
    save_checkpoint(old, old_path, {})
    assert resolve_ppo_update_design(Namespace(ppo_update_design="auto", load_checkpoint=None,
        init_fast_checkpoint=str(old_path))) == "full-rollout-v1"


def test_fresh_auto_uses_repair_but_legacy_api_is_unchanged():
    assert resolve_ppo_update_design(Namespace(ppo_update_design="auto", load_checkpoint=None,
        training_repair="batch-credit-v1", slow_deployment_context="deployment-aware-v2")) == "guarded-minibatch-v2"
    env = tiny_replay_env(); env.reset()
    old = HierarchicalPPOAgent.from_env(env)
    assert old.fast_agent.ppo_update_design == "full-rollout-v1"
    assert not old.fast_agent.ppo.policy.independent_critic
