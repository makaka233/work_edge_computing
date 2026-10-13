"""Credit, KL and checkpoint invariants; not a claim of training convergence."""

from argparse import Namespace
from dataclasses import replace
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from edge_drl.agents.drl import HierarchicalPPOAgent, FastSchedulingPPOAgent, slow_obs_dim, slow_incremental_obs_dim
from edge_drl.models.ppo import PPOAgent
from tests.comparison_helpers import tiny_replay_env
from train_dual_ppo import load_checkpoint, save_checkpoint, resolve_training_repair, rollout, build_episode_metrics_row


def repaired_agent(env):
    return HierarchicalPPOAgent.from_env(
        env, replicas_per_stage=6, fast_chain_context="chain-kkt-v3",
        slow_allocation_mode="budget-continuous-v2", training_repair="batch-credit-v1",
        slow_k_epochs=1, slow_stage_k_epochs=1, slow_placement_k_epochs=1,
        slow_critic_k_epochs=1, fast_k_epochs=2, fast_minibatch_size=8,
    )


def test_batch_return_credits_first_action_for_later_congestion():
    fast = FastSchedulingPPOAgent(2, 1, chain_context="chain-kkt-v3", batch_credit=True)
    for second_reward, expected in [(-3.0, -4.0), (-1.0, -2.0)]:
        fast.ppo.buffer.clear()
        fast.ppo.buffer.values.extend([0.0, 0.0])
        fast.assign_schedule_rewards([-1.0], done=False, batch_end=False)
        fast.assign_schedule_rewards([second_reward], done=False, batch_end=True)
        assert fast.ppo.buffer.dones == [False, True]
        _, returns = fast.ppo._gae_advantages_and_returns()
        assert returns[0] == expected


def test_batch_weights_do_not_square_request_multiplicity():
    fast = FastSchedulingPPOAgent(2, 2, chain_context="chain-kkt-v3", batch_credit=True)
    fast.assign_schedule_rewards([-1.0, -2.0], done=False, weight=300.0, multiplicity=3.0, batch_end=True)
    assert fast.ppo.buffer.rewards == [-3.0, -6.0]
    assert fast.ppo.buffer.weights == [100.0, 100.0]
    assert fast.ppo.buffer.dones == [False, True]
    with pytest.raises(ValueError, match="explicit batch_end"):
        fast.assign_schedule_rewards([-1.0], False)


def test_fast_critic_cannot_move_actor_encoder():
    fast = FastSchedulingPPOAgent(2, 2, chain_context="chain-kkt-v3", batch_credit=True)
    policy = fast.ppo.policy
    width = policy.global_dim + 2 * policy.node_feature_dim + 4 * policy.edge_feature_dim
    states = torch.randn(3, width)
    _, values = policy(states, torch.ones((3, 2), dtype=torch.bool))
    values.square().mean().backward()
    for name, parameter in policy.named_parameters():
        if not name.startswith("critic."):
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    assert any(p.grad is not None for p in policy.critic.parameters())


def test_value_normalization_preserves_raw_values_and_policy():
    agent = PPOAgent(obs_dim=4, action_dim=2, normalize_value_targets=True)
    states = torch.randn(7, 4)
    masks = torch.ones((7, 2), dtype=torch.bool)
    with torch.no_grad():
        dist, before = agent.policy(states, masks)
        probabilities = dist.probs.clone()
    for returns in (np.arange(7, dtype=np.float32) * 200 - 1000, np.arange(7, dtype=np.float32) * 10):
        agent._normalized_value_targets(returns, np.ones(7))
        with torch.no_grad():
            new_dist, normalized = agent.policy(states, masks)
        torch.testing.assert_close(agent._raw_values(normalized), before, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(new_dist.probs, probabilities)


def filled_guarded_ppo():
    torch.manual_seed(35)
    agent = PPOAgent(obs_dim=4, action_dim=2, lr=1e-2, k_epochs=2,
                     minibatch_size=3, target_kl=1e-3, full_batch_kl_stop=True,
                     guarded_epochs=True, value_coef=0.0)
    for i in range(12):
        state = np.array([i / 12, 1, -1, 0.5], dtype=np.float32)
        mask = np.ones(2, dtype=bool)
        action, logprob, value = agent.act(state, mask)
        agent.buffer.add(state=state, mask=mask, action=action, logprob=logprob,
                         value=value, reward=float(action * 2 - 1), done=True)
        agent.buffer.sample_groups.append(float(i % 4))
    return agent


def test_guarded_epochs_cover_all_groups_and_bound_post_step_kl():
    agent = filled_guarded_ppo()
    metrics = agent.update()
    assert metrics["optimizer_steps"] > 0
    assert metrics["optimizer_steps"] <= 2
    assert metrics["samples_seen_fraction"] == metrics["min_group_seen_fraction"] == 1.0
    assert metrics["max_group_approx_kl"] <= agent.target_kl
    assert metrics["approx_kl"] <= agent.target_kl


def test_failed_kl_steps_restore_parameters_and_adam(monkeypatch):
    agent = filled_guarded_ppo()
    before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
    diagnostics = agent._buffer_policy_diagnostics
    calls = [0]
    def reject(*args, **kwargs):
        result = diagnostics(*args, **kwargs)
        if calls[0] < 7:
            result["approx_kl"] = result["max_group_approx_kl"] = 1.0
        calls[0] += 1
        return result
    monkeypatch.setattr(agent, "_buffer_policy_diagnostics", reject)
    metrics = agent.update()
    assert metrics["optimizer_steps"] == 0
    assert metrics["kl_backtracks"] == 7
    assert not agent.optimizer.state
    for key, value in agent.policy.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_slow_observes_catalogue_cost_changes():
    env = tiny_replay_env()
    env.reset()
    agent = repaired_agent(env)
    slow = agent.slow_agent
    demand = slow._node_service_demand(env)
    memory, storage = env.service_memory_capacities(), env.service_storage_capacities()
    before = slow._build_window_state(env, demand, memory, storage)
    assert len(before) == slow_obs_dim(6, 2, True)
    service = env.scenario.services[0]
    env.scenario.services[0] = replace(service, input_mb_mean=service.input_mb_mean * 5,
                                     stages=tuple(replace(s, compute_gcycles_mean=s.compute_gcycles_mean * 10) for s in service.stages))
    after = slow._build_window_state(env, demand, memory, storage)
    assert not np.array_equal(before, after)
    stage_state = slow._build_incremental_state(env, demand, memory, storage, env.deployment)
    assert len(stage_state) == slow_incremental_obs_dim(6, 2, 3, True)


def test_repaired_rollout_credit_closes_and_updates_both_agents():
    env = tiny_replay_env()
    env.reset()
    agent = repaired_agent(env)
    stats = rollout(env, agent, max_requests=10, train_mode="joint", reward_scale=10)
    assert stats["requests"] > 0
    assert len(agent.fast_agent.ppo.buffer.rewards) == len(agent.fast_agent.ppo.buffer.actions)
    assert agent.fast_agent.ppo.buffer.dones[-1]
    assert stats["max_fast_kkt_reward_residual_s"] < 1e-8
    agent.flush_slow_window_reward(done=True, env=env)
    metrics = agent.update()
    assert metrics["fast"]["samples_seen_fraction"] == 1.0
    assert metrics["slow"]["team_window_credit"] == 1.0
    assert metrics["slow"]["stage_optimizer_steps"] > 0


def test_slow_team_credit_does_not_depend_on_sparse_probe_defaults(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    agent = repaired_agent(env)
    slow = agent.slow_agent
    for reward in (-1.0, -2.0):
        deployment = slow.plan_deployment(env, record=True)
        env.apply_deployment(deployment)
        slow.assign_pending_reward(reward, done=True, count_stage_returns={(0, 0): -999.0})
    # Deliberately poison old local credit; new actor targets must remain actual window returns.
    slow.count_action_returns[:] = [-999.0] * len(slow.count_action_returns)
    slow.stage_action_returns[:] = [-999.0] * len(slow.stage_action_returns)
    for ppo, ids in ((slow.count_ppo, slow.count_window_ids), (slow.stage_ppo, slow.stage_window_ids),
                     (slow.placement_ppo, slow.placement_window_ids)):
        original = ppo.update_from_returns
        expected = np.array([-1.0, -2.0])[ids]
        def capture(returns, *, _expected=expected, _original=original, **kwargs):
            np.testing.assert_allclose(returns, _expected)
            assert kwargs["actor_use_value_baseline"]
            assert kwargs.get("actor_baseline_values") is None  # own pre-action conditional critic
            assert kwargs["auxiliary_advantage_coef"] == 0.0
            return _original(returns, **kwargs)
        monkeypatch.setattr(ppo, "update_from_returns", capture)
    metrics = slow.update()
    assert metrics["trajectory_return_mean"] == -1.5


def test_checkpoint_restores_repair_mode_and_value_scale(tmp_path):
    env = tiny_replay_env()
    env.reset()
    source = repaired_agent(env)
    source.fast_agent.ppo._normalized_value_targets(np.array([-200.0, -50.0]), np.ones(2))
    path = tmp_path / "repair.pt"
    save_checkpoint(source, path, {"args": {"training_repair": "batch-credit-v1"}})
    restored = repaired_agent(env)
    load_checkpoint(restored, path)
    assert restored.fast_agent.ppo.value_normalization_state() == source.fast_agent.ppo.value_normalization_state()
    state = source.fast_agent._build_state(env, next(slot[0] for slot in env._comparison_trace.slots if slot), 0, [])
    mask = np.ones(6, dtype=bool)
    assert source.fast_agent.ppo.act(state, mask, deterministic=True) == restored.fast_agent.ppo.act(state, mask, deterministic=True)
    legacy = HierarchicalPPOAgent.from_env(env, slow_allocation_mode="budget-continuous-v2", fast_chain_context="chain-kkt-v3")
    with pytest.raises(ValueError, match="repair version"):
        load_checkpoint(legacy, path)
    assert resolve_training_repair(Namespace(training_repair="auto", load_checkpoint=str(path))) == "batch-credit-v1"
    legacy_path = tmp_path / "legacy.pt"
    save_checkpoint(legacy, legacy_path, {})
    assert resolve_training_repair(Namespace(training_repair="auto", load_checkpoint=str(legacy_path))) == "legacy"


def test_episode_first_last_latency_not_weighted_into_same_mean(monkeypatch):
    monkeypatch.setattr("train_dual_ppo.aggregate_rollout_stats", lambda _: {
        "requests": 20.0, "avg_reward": -0.2, "avg_train_reward": -0.2,
        "first_window_avg_latency_s": 0.2, "last_window_avg_latency_s": 0.2,
        "window_latency_delta_s": 0.0,
    })
    context = {key: 1 for key in (
        "episode", "update", "demand_seed", "environment_seed", "load_multiplier",
        "load_group", "load_target_probability", "start_minute", "rollout", "window")}
    context["training_phase"] = "joint"
    row = build_episode_metrics_row([
        {"first_window_avg_latency_s": 0.3, "last_window_avg_latency_s": 0.3},
        {"first_window_avg_latency_s": 0.1, "last_window_avg_latency_s": 0.1},
    ], [context, context])
    assert row["first_window_avg_latency_s"] == 0.3
    assert row["last_window_avg_latency_s"] == 0.1
    assert row["window_latency_delta_s"] == pytest.approx(-0.2)


def test_repaired_monolithic_keeps_collapsed_view_and_batch_credit():
    from edge_drl.comparison.monolithic import collapse_scenario, compact_request_for_monolithic_policy, specialize_monolithic_fast_agent, MONOLITHIC_FAST_DESIGN
    env = tiny_replay_env()
    env.reset()
    env.scenario = collapse_scenario(env.scenario)
    original_requests = list(next(slot for slot in env._comparison_trace.slots if len(slot) > 1))
    requests = [compact_request_for_monolithic_policy(r) for r in original_requests]
    env.current_requests, env.current_request = requests, requests[0]
    agent = repaired_agent(env)
    specialize_monolithic_fast_agent(agent, MONOLITHIC_FAST_DESIGN)
    deployment = agent.slow_agent.plan_deployment(env, record=False)
    env.apply_deployment(deployment)
    schedules = agent.fast_agent.schedule_batch(env, requests)
    assert all(len(path) == 1 for path in schedules)
    for i, request in enumerate(requests):
        costs = agent.fast_agent.consume_pending_kkt_stage_costs()
        agent.fast_agent.assign_schedule_rewards([-c for c in costs], False,
                                                batch_end=i == len(requests) - 1,
                                                multiplicity=float(request.request_count))
    assert sum(agent.fast_agent.ppo.buffer.dones) == 1
    assert not agent.fast_agent._batch_reward_queue()
    assert agent.fast_agent.update()["samples_seen_fraction"] == 1.0


def test_new_profile_training_and_resume_entrypoint(tmp_path):
    # Small plumbing check, not a representative convergence experiment.
    common = [
        sys.executable, "train_dual_ppo.py", "--pressure-profile", "mec-latency200",
        "--num-edge-nodes", "8", "--num-service-types", "2", "--num-users", "10000",
        "--episode-minutes", "20", "--fast-windows-per-update", "2", "--slow-windows-per-update", "2",
        "--updates", "1", "--sampled-seconds-per-window", "1", "--eval-interval", "0",
        "--sampling-max-interval-seconds", "0",
        "--slow-critic-k-epochs", "1", "--slow-k-epochs", "1", "--slow-stage-k-epochs", "1",
        "--slow-placement-k-epochs", "1", "--fast-k-epochs", "2", "--fast-minibatch-size", "16",
        "--device", "cpu", "--run-root", str(tmp_path), "--progress-interval-seconds", "0",
    ]
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(common + ["--run-name", "fresh"], env=environment,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert "training_repair=batch-credit-v1" in completed.stdout
    path = tmp_path / "fresh" / "checkpoints" / "latest.pt"
    assert path.is_file()
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["training_repair"] == "batch-credit-v1"
    assert checkpoint["fast_value_normalization"]["initialized"]
    with (tmp_path / "fresh" / "logs" / "training.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[-1]["fast_batch_credit"]) == 1
    assert float(rows[-1]["slow_team_window_credit"]) == 1
    assert float(rows[-1]["fast_min_group_seen_fraction"]) == 1
    assert float(rows[-1]["fast_max_group_approx_kl"]) <= .015
    subprocess.run(common + ["--run-name", "resumed", "--load-checkpoint", str(path)], env=environment,
                   check=True, capture_output=True, text=True, timeout=120)
    with (tmp_path / "resumed" / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["args"]["training_repair"] == "batch-credit-v1"
