import numpy as np
import torch

from edge_drl.agents.drl import (
    HierarchicalPPOAgent,
    _discounted_window_returns,
    fast_obs_dim,
    slow_obs_dim,
)
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from edge_drl.models.ppo import (
    PPOAgent,
    _balance_group_weights,
    _center_advantages_by_group,
    _normalize_advantages_by_group,
    _stratified_minibatches,
)
from train_dual_ppo import _stage_congestion_externality_costs, _stage_latency_costs, rollout


def test_observation_dimensions():
    assert slow_obs_dim(16, 3) == 17 + 16 * (6 + 3) + 16 * 16 * 4
    assert fast_obs_dim(16) == 12 + 16 * 9 + 16 * 16 * 3
    assert fast_obs_dim(16, chain_context="chain-v2") == 16 + 16 * 13 + 16 * 16 * 3


def test_fast_chain_v2_state_and_bounded_oracle_are_valid():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=102,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_minutes=10,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=6,
        fast_chain_context="chain-v2",
    )
    agent.maybe_update_deployment(env, deterministic=True, record=False)
    request = env.current_requests[0]
    state = agent.fast_agent._build_state(env, request, 0, [])
    assert state.shape == (fast_obs_dim(16, chain_context="chain-v2"),)

    partial: list[int] = []
    deliberately_bad: list[int] = []
    for stage_id in range(len(request.stage_compute_gcycles)):
        candidates = np.flatnonzero(agent.fast_agent._build_mask(env, request, stage_id, partial))
        selected = max(
            candidates.tolist(),
            key=lambda node_id: agent.fast_agent._stage_candidate_proxy_cost(
                env, request, stage_id, int(node_id), partial
            ),
        )
        deliberately_bad.append(int(selected))
        partial.append(int(selected))
    regrets = agent.fast_agent.stage_counterfactual_regrets(env, request, deliberately_bad)
    oracle = agent.fast_agent.chain_oracle_schedule(
        env,
        request,
        beam_width=8,
        candidates_per_stage=4,
    )

    assert len(regrets) == len(request.stage_compute_gcycles)
    assert all(regret >= 0.0 for regret in regrets)
    assert any(regret > 0.0 for regret in regrets)
    assert env.evaluate_schedule(request, oracle)["valid"]


def test_slow_discounted_returns_stop_at_episode_boundaries():
    rewards = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float32)
    dones = np.asarray([False, False, True, False, True])

    returns = _discounted_window_returns(rewards, dones, gamma=0.5)

    np.testing.assert_allclose(returns, [2.75, 3.5, 3.0, 20.0, 20.0])


def test_slow_window_critic_uses_temporal_holdout_and_replay():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=101,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    slow = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4).slow_agent
    obs_dim = slow_obs_dim(16, 3)

    slow.window_states = [
        np.full(obs_dim, index / 12.0, dtype=np.float32) for index in range(12)
    ]
    slow.window_episode_ids = [0] * 6 + [1] * 6
    first = slow._update_window_critic(np.linspace(-20.0, -5.0, 12, dtype=np.float32))
    assert first["replay_size"] == 12.0
    assert first["train_size"] == 0.0
    assert first["holdout_size"] == 12.0

    slow.window_states = [
        np.full(obs_dim, 1.0 + index / 12.0, dtype=np.float32) for index in range(12)
    ]
    slow.window_episode_ids = [2] * 6 + [3] * 6
    second = slow._update_window_critic(np.linspace(-18.0, -3.0, 12, dtype=np.float32))
    assert second["replay_size"] == 24.0
    assert second["train_size"] == 12.0
    assert second["holdout_size"] == 12.0
    assert second["train_episode_count"] == 2.0
    assert second["holdout_episode_count"] == 2.0
    assert np.isfinite(second["value_loss"])
    assert np.isfinite(second["raw_value_mse"])
    assert np.isfinite(second["train_explained_variance"])
    assert np.isfinite(second["holdout_explained_variance"])
    assert slow.critic_return_std > 0.0


def test_count_advantages_are_centered_within_service_stage():
    advantages = np.asarray([10.0, 14.0, -5.0, 3.0], dtype=np.float32)
    weights = np.asarray([1.0, 3.0, 2.0, 2.0], dtype=np.float32)
    group_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)

    centered = _center_advantages_by_group(advantages, weights, group_ids)

    assert np.allclose(centered, [-3.0, 1.0, -4.0, 4.0])
    assert np.isclose(np.average(centered[:2], weights=weights[:2]), 0.0)
    assert np.isclose(np.average(centered[2:], weights=weights[2:]), 0.0)


def test_fast_load_groups_are_balanced_normalized_and_stratified():
    group_ids = np.asarray([0] * 4 + [1] * 8 + [2] * 12 + [3] * 16, dtype=np.float32)
    weights = np.ones(len(group_ids), dtype=np.float32)
    balanced = _balance_group_weights(weights, group_ids)
    group_weight_sums = [balanced[group_ids == group_id].sum() for group_id in range(4)]
    assert np.allclose(group_weight_sums, group_weight_sums[0])

    advantages = np.arange(len(group_ids), dtype=np.float32)
    normalized = _normalize_advantages_by_group(advantages, balanced, group_ids)
    for group_id in range(4):
        mask = group_ids == group_id
        assert np.isclose(np.average(normalized[mask], weights=balanced[mask]), 0.0, atol=1e-6)
        assert np.isclose(np.average(np.square(normalized[mask]), weights=balanced[mask]), 1.0, atol=1e-5)

    batches = _stratified_minibatches(len(group_ids), 10, group_ids)
    assert len(batches) == 4
    assert sorted(np.concatenate(batches).tolist()) == list(range(len(group_ids)))
    assert all(set(group_ids[batch].tolist()) == {0.0, 1.0, 2.0, 3.0} for batch in batches)


def test_fast_load_groups_follow_configured_target_distribution():
    group_ids = np.asarray([0] * 4 + [1] * 8 + [2] * 12 + [3] * 16, dtype=np.float32)
    weights = np.ones(len(group_ids), dtype=np.float32)
    targets = np.asarray([0.20, 0.50, 0.25, 0.05], dtype=np.float32)

    balanced = _balance_group_weights(weights, group_ids, targets)
    group_weight_sums = np.asarray(
        [balanced[group_ids == group_id].sum() for group_id in range(4)]
    )

    np.testing.assert_allclose(group_weight_sums / group_weight_sums.sum(), targets, atol=1e-6)


def test_fast_full_batch_kl_stop_covers_every_load_before_stopping():
    torch.manual_seed(71)
    np.random.seed(71)
    ppo = PPOAgent(
        obs_dim=2,
        action_dim=2,
        hidden_dim=8,
        lr=0.05,
        k_epochs=4,
        minibatch_size=4,
        target_kl=1e-8,
        group_balanced_updates=True,
        full_batch_kl_stop=True,
    )
    mask = np.asarray([True, True])
    for index in range(16):
        state = np.asarray([float(index % 2), float(index // 2) / 8.0], dtype=np.float32)
        probabilities, value = ppo.action_probabilities(state, mask)
        action = index % 2
        ppo.buffer.states.append(state)
        ppo.buffer.masks.append(mask.copy())
        ppo.buffer.actions.append(action)
        ppo.buffer.logprobs.append(float(np.log(probabilities[action])))
        ppo.buffer.values.append(value)
        ppo.buffer.rewards.append(1.0 if action == int(state[0]) else -1.0)
        ppo.buffer.dones.append(True)
        ppo.buffer.weights.append(1.0)
        ppo.buffer.sample_groups.append(float(index // 4))

    metrics = ppo.update()

    assert metrics["kl_early_stop"] == 1.0
    assert metrics["epochs_completed"] >= 1.0
    assert metrics["optimizer_steps"] >= 4.0
    assert metrics["samples_seen_fraction"] == 1.0
    assert metrics["min_group_seen_fraction"] == 1.0
    assert metrics["full_batch_kl_checks"] >= 1.0
    assert metrics["group_count"] == 4.0
    assert set(ppo.last_group_diagnostics) == {0.0, 1.0, 2.0, 3.0}


def test_count_actor_can_bypass_failed_value_baseline():
    ppo = PPOAgent(obs_dim=2, action_dim=2, hidden_dim=8, k_epochs=1, minibatch_size=4)
    states = [np.asarray([float(index), 1.0], dtype=np.float32) for index in range(4)]
    mask = np.asarray([True, True])
    for state in states:
        action, logprob, _ = ppo.act(state, mask, deterministic=False)
        ppo.buffer.states.append(state)
        ppo.buffer.masks.append(mask.copy())
        ppo.buffer.actions.append(action)
        ppo.buffer.logprobs.append(logprob)
    # Deliberately unusable critic predictions must not alter the direct-return
    # Count actor advantages.
    ppo.buffer.values.extend([100.0, -100.0, 50.0, -50.0])
    metrics = ppo.update_from_returns(
        np.asarray([10.0, 14.0, -5.0, 3.0], dtype=np.float32),
        advantage_group_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        actor_use_value_baseline=False,
    )
    assert np.isclose(metrics["advantage_mean"], 0.0)
    assert np.isclose(metrics["advantage_std"], np.sqrt(10.0))


def test_component_actor_mixes_normalized_window_advantage():
    ppo = PPOAgent(obs_dim=2, action_dim=2, hidden_dim=8, k_epochs=1, minibatch_size=4)
    mask = np.asarray([True, True])
    for index in range(4):
        state = np.asarray([float(index), 1.0], dtype=np.float32)
        action, logprob, value = ppo.act(state, mask, deterministic=False)
        ppo.buffer.states.append(state)
        ppo.buffer.masks.append(mask.copy())
        ppo.buffer.actions.append(action)
        ppo.buffer.logprobs.append(logprob)
        ppo.buffer.values.append(value)

    metrics = ppo.update_from_returns(
        np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        actor_use_value_baseline=False,
        auxiliary_advantages=np.asarray([2.0, 2.0, -2.0, -2.0], dtype=np.float32),
        auxiliary_advantage_coef=0.5,
    )

    assert np.isclose(metrics["auxiliary_advantage_mean"], 0.0)
    assert np.isclose(metrics["auxiliary_advantage_std"], 2.0)
    assert np.isclose(metrics["auxiliary_advantage_coef"], 0.5)
    assert metrics["combined_advantage_std"] > 1.0


def test_component_actor_attributes_shared_window_credit_to_local_actions():
    ppo = PPOAgent(obs_dim=2, action_dim=2, hidden_dim=8, k_epochs=1, minibatch_size=4)
    mask = np.asarray([True, True])
    for index in range(4):
        state = np.asarray([float(index), 1.0], dtype=np.float32)
        action, logprob, value = ppo.act(state, mask, deterministic=False)
        ppo.buffer.states.append(state)
        ppo.buffer.masks.append(mask.copy())
        ppo.buffer.actions.append(action)
        ppo.buffer.logprobs.append(logprob)
        ppo.buffer.values.append(value)

    metrics = ppo.update_from_returns(
        np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        actor_use_value_baseline=False,
        auxiliary_advantages=np.asarray([-2.0, -2.0, 2.0, 2.0], dtype=np.float32),
        auxiliary_advantage_coef=0.5,
        auxiliary_attribution=np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        auxiliary_attribution_coef=0.5,
    )

    assert np.isclose(metrics["auxiliary_attribution_mean_abs"], 1.0)
    assert np.isclose(metrics["auxiliary_attribution_coef"], 0.5)


def test_fast_ppo_reports_full_batch_diagnostics_and_strict_kl_stop():
    torch.manual_seed(7)
    np.random.seed(7)
    ppo = PPOAgent(
        obs_dim=2,
        action_dim=2,
        hidden_dim=8,
        lr=0.05,
        k_epochs=4,
        minibatch_size=4,
        target_kl=1e-8,
    )
    mask = np.asarray([True, True])
    for index in range(16):
        state = np.asarray([float(index % 2), float(index // 2) / 8.0], dtype=np.float32)
        probabilities, value = ppo.action_probabilities(state, mask)
        action = index % 2
        ppo.buffer.states.append(state)
        ppo.buffer.masks.append(mask.copy())
        ppo.buffer.actions.append(action)
        ppo.buffer.logprobs.append(float(np.log(probabilities[action])))
        ppo.buffer.values.append(value)
        ppo.buffer.rewards.append(1.0 if action == int(state[0]) else -1.0)
        ppo.buffer.dones.append(True)
        ppo.buffer.weights.append(1.0)

    metrics = ppo.update()

    assert metrics["entropy"] >= 0.0
    assert 0.0 <= metrics["normalized_entropy"] <= 1.0
    assert metrics["approx_kl"] >= 0.0
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["advantage_std"] > 0.0
    assert metrics["epochs_completed"] <= 4.0
    assert metrics["kl_early_stop"] == 1.0


def test_disabled_count_critic_does_not_update_critic_head():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=10,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    ppo = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4).slow_agent.count_ppo
    mask = np.ones(4, dtype=bool)
    for index in range(4):
        state = np.zeros(slow_obs_dim(16, 3), dtype=np.float32)
        state[index] = 1.0
        action, logprob, value = ppo.act(state, mask, deterministic=False)
        ppo.buffer.states.append(state)
        ppo.buffer.masks.append(mask.copy())
        ppo.buffer.actions.append(action)
        ppo.buffer.logprobs.append(logprob)
        ppo.buffer.values.append(value)
    critic_before = [parameter.detach().cpu().numpy().copy() for parameter in ppo.policy.critic.parameters()]

    ppo.update_from_returns(
        np.asarray([1.0, -1.0, 2.0, -2.0], dtype=np.float32),
        advantage_group_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        actor_use_value_baseline=False,
    )

    critic_after = [parameter.detach().cpu().numpy() for parameter in ppo.policy.critic.parameters()]
    assert ppo.value_coef == 0.0
    assert all(np.array_equal(before, after) for before, after in zip(critic_before, critic_after))


def test_dual_ppo_rollout_and_update():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=11,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    assert agent.slow_migration_coef == 0

    request_count = 0
    ticks = 0
    while ticks < 2 and not env.done:
        requests = list(env.current_requests)
        actions = agent.act_batch(env)
        _, reward, done, batch_info = env.step(actions)
        for group_idx, (request, action, info) in enumerate(zip(requests, actions, batch_info["group_infos"])):
            agent.observe_step_reward(
                -float(info["latency_s"]),
                len(request.stage_compute_gcycles),
                bool(done and group_idx == len(requests) - 1),
                weight=request.request_count,
            )
            assert len(action) == len(request.stage_compute_gcycles)
            assert info["latency_s"] >= 0
            request_count += 1
        assert np.isfinite(reward)
        ticks += 1

    agent.flush_slow_window_reward(done=True)
    assert len(agent.slow_agent.count_ppo.buffer) > 0
    assert len(agent.slow_agent.placement_ppo.buffer) > 0
    assert len(agent.slow_agent.window_returns) == 1
    assert len(agent.slow_agent.count_ppo.buffer.rewards) == 0
    assert len(agent.slow_agent.placement_ppo.buffer.rewards) == 0
    assert len(agent.fast_agent.ppo.buffer) >= request_count
    assert len(agent.fast_agent.ppo.buffer.weights) == len(agent.fast_agent.ppo.buffer)
    assert min(agent.fast_agent.ppo.buffer.weights) >= 1.0

    losses = agent.update()
    assert "slow" in losses
    assert "fast" in losses
    assert np.isfinite(losses["slow"]["loss"])
    assert np.isfinite(losses["slow"]["count_loss"])
    assert np.isfinite(losses["slow"]["placement_loss"])
    assert losses["slow"]["count_value_loss"] > 0.0
    assert losses["slow"]["placement_value_loss"] > 0.0
    assert np.isfinite(losses["slow"]["count_explained_variance"])
    assert np.isfinite(losses["slow"]["placement_explained_variance"])
    assert np.isfinite(losses["slow"]["count_post_explained_variance"])
    assert np.isfinite(losses["slow"]["placement_post_explained_variance"])
    assert losses["slow"]["window_count"] == 1
    assert np.isclose(losses["slow"]["placement_entropy_coef"], 0.005)
    assert losses["slow"]["placement_updates_completed"] == 1
    assert losses["slow"]["count_global_advantage_coef"] == 0.0
    assert losses["slow"]["placement_global_advantage_coef"] == 0.0
    assert losses["slow"]["count_global_advantage_configured_coef"] == 0.25
    assert losses["slow"]["placement_global_advantage_configured_coef"] == 0.35
    assert losses["slow"]["global_advantage_reliability"] == 0.0
    assert losses["slow"]["window_critic_replay_size"] == 1.0
    assert losses["slow"]["window_critic_train_size"] == 0.0
    assert losses["slow"]["window_critic_holdout_size"] == 1.0
    assert np.isfinite(losses["slow"]["window_critic_holdout_explained_variance"])
    assert np.isclose(agent.slow_agent.placement_entropy_coefficient(), 0.005)
    assert np.isfinite(losses["slow"]["critic_explained_variance"])
    assert np.isfinite(losses["fast"]["loss"])
    assert len(agent.slow_agent.count_ppo.buffer) == 0
    assert len(agent.slow_agent.placement_ppo.buffer) == 0
    assert len(agent.slow_agent.window_returns) == 0
    assert len(agent.fast_agent.ppo.buffer) == 0


def test_fast_batch_inference_keeps_request_major_buffer_order():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=18,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
            demand_load_multiplier=1.37,
            demand_load_group=2,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    requests = list(env.current_requests)
    actions = agent.fast_agent.schedule_batch(env, requests, deterministic=True, record=True)

    expected_transitions = sum(len(request.stage_compute_gcycles) for request in requests)
    assert len(agent.fast_agent.ppo.buffer) == expected_transitions
    assert agent.fast_agent.ppo.buffer.sample_groups == [2.0] * expected_transitions
    offset = 0
    for request_idx, request in enumerate(requests):
        for stage_id in range(len(request.stage_compute_gcycles)):
            expected_state = agent.fast_agent._build_state(
                env,
                request,
                stage_id,
                actions[request_idx][:stage_id],
            )
            np.testing.assert_allclose(agent.fast_agent.ppo.buffer.states[offset], expected_state)
            offset += 1


def test_fast_virtual_reservation_changes_later_microbatch_state():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=181,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        fast_reservation_microbatch_size=1,
    )
    agent.maybe_update_deployment(env, deterministic=True, record=False)
    request = env.current_requests[0]
    for stage_id in range(len(request.stage_compute_gcycles)):
        env.deployment[request.service_id, stage_id] = True
    schedules = agent.fast_agent.schedule_batch(env, [request, request], deterministic=True, record=True)
    stage_count = len(request.stage_compute_gcycles)
    first_state = agent.fast_agent.ppo.buffer.states[0]
    second_state = agent.fast_agent.ppo.buffer.states[stage_count]
    selected_node = schedules[0][0]
    candidate_pressure_offset = 12 + selected_node * 9 + 8

    assert not np.allclose(first_state, second_state)
    assert second_state[candidate_pressure_offset] > first_state[candidate_pressure_offset]


def test_fast_state_includes_current_compute_pressure():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=19,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=6_000.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    agent.maybe_update_deployment(env)

    node_work, stage_work = agent.fast_agent._expected_current_compute_workload(env)
    assert node_work.shape == (16,)
    assert node_work.sum() > 0.0
    assert stage_work

    request = env.current_requests[0]
    state = agent.fast_agent._build_state(env, request, 0, [])
    node_features = state[12 : 12 + 16 * 9].reshape(16, 9)
    assert np.all(node_features[:, 6] >= 0.0)
    assert np.all(node_features[:, 7] >= 0.0)
    assert np.all(node_features[:, 8] >= 0.0)
    assert np.any(node_features[:, 6] > 0.0)


def test_rollout_reports_latency_components():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=20,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    stats = rollout(env, agent, max_requests=6, rollout_unit="requests")

    for key in (
        "avg_compute_latency_s",
        "avg_link_latency_s",
        "avg_access_latency_s",
        "avg_propagation_latency_s",
        "avg_instantaneous_compute_work_gcycles",
        "avg_instantaneous_compute_pressure",
        "avg_max_instantaneous_compute_pressure",
        "avg_p95_instantaneous_compute_pressure",
    ):
        assert key in stats
        assert stats[key] >= 0.0
    assert np.isclose(
        stats["avg_latency_s"],
        stats["avg_compute_latency_s"]
        + stats["avg_link_latency_s"]
        + stats["avg_access_latency_s"]
        + stats["avg_propagation_latency_s"]
        + stats["avg_penalty_latency_s"],
    )


def test_fast_stage_costs_sum_to_request_latency():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=21,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=6_000.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    agent.maybe_update_deployment(env)
    request = env.current_requests[0]
    schedule = agent.fast_agent.schedule(env, request=request, record=False)
    info = env.evaluate_schedule(request, schedule)

    stage_costs = _stage_latency_costs(info, env, request)
    assert len(stage_costs) == len(request.stage_compute_gcycles)
    assert all(cost >= 0.0 for cost in stage_costs)
    assert np.isclose(sum(stage_costs), info["latency_s"])

    controllable_costs = _stage_latency_costs(
        info,
        env,
        request,
        include_access=False,
    )
    assert np.isclose(
        sum(controllable_costs),
        info["latency_s"] - info["access_delay_s"],
    )


def test_fast_stage_externality_is_positive_under_shared_kkt_resources():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=211,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=6_000.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    agent.maybe_update_deployment(env, deterministic=True, record=False)
    request = env.current_requests[0]
    schedule = agent.fast_agent.schedule(env, request=request, deterministic=True, record=False)
    infos = env.evaluate_batch_schedules([request, request], [schedule, schedule])

    externality_costs = _stage_congestion_externality_costs(infos[0], env, request)
    assert len(externality_costs) == len(request.stage_compute_gcycles)
    assert sum(externality_costs) > 0.0


def test_fast_terminal_reward_is_not_repeated_across_stages():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=22,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    agent.fast_agent.assign_last_schedule_reward(-2.0, stage_count=3, done=False)

    assert agent.fast_agent.ppo.buffer.rewards == [0.0, 0.0, -2.0]
    assert agent.fast_agent.ppo.buffer.dones == [False, False, True]


def test_deterministic_eval_does_not_write_rollout_buffers():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=13,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    stats = rollout(env, agent, max_requests=4, deterministic=True, record=False)
    assert stats["requests"] >= 4
    assert len(agent.slow_agent.count_ppo.buffer) == 0
    assert len(agent.slow_agent.placement_ppo.buffer) == 0
    assert len(agent.fast_agent.ppo.buffer) == 0


def test_fast_update_preserves_slow_window_buffer():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=17,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            deployment_interval_minutes=1,
            mean_requests_per_minute=60.0,
        )
    )
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    rollout(env, agent, max_requests=1, rollout_unit="window")

    assert agent.completed_slow_windows == 1
    assert len(agent.fast_agent.ppo.buffer) > 0
    fast_metrics = agent.update_fast()
    assert np.isfinite(fast_metrics["loss"])
    assert len(agent.fast_agent.ppo.buffer) == 0
    assert agent.completed_slow_windows == 1

    slow_metrics = agent.update_slow()
    assert slow_metrics["window_count"] == 1
    assert agent.completed_slow_windows == 0


def test_slow_collection_records_a_multi_window_episode_boundary():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=171,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_minutes=6,
            deployment_interval_minutes=1,
            mean_requests_per_minute=10.0,
        )
    )
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        slow_window_gamma=0.95,
    )

    for window in range(6):
        rollout(
            env,
            agent,
            max_requests=1,
            rollout_unit="window",
            reset_env=window == 0,
        )

    assert agent.completed_slow_windows == 6
    assert agent.slow_agent.window_dones == [False, False, False, False, False, True]
    immediate_mean = float(np.mean(agent.slow_agent.window_returns))
    slow_metrics = agent.update_slow()
    assert slow_metrics["window_count"] == 6
    assert not np.isclose(slow_metrics["trajectory_return_mean"], immediate_mean)


def test_fast_only_rollout_records_only_fast_buffer():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=19,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    stats = rollout(env, agent, max_requests=5, train_mode="fast-only", reward_scale=0.1)
    assert stats["requests"] >= 5
    assert len(agent.slow_agent.count_ppo.buffer) == 0
    assert len(agent.slow_agent.placement_ppo.buffer) == 0
    assert len(agent.fast_agent.ppo.buffer) >= 5


def test_fast_agent_supports_node_scorer_fallback():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=23,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4, fast_policy_kind="node_scorer")
    stats = rollout(env, agent, max_requests=4, train_mode="fast-only", reward_scale=0.1)
    assert stats["requests"] >= 4
    assert len(agent.fast_agent.ppo.buffer) >= 4


def test_slow_agent_uses_explicit_count_and_unique_placements():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=29,
            num_users=10_000,
            num_edge_nodes=32,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    assert agent.slow_agent.count_ppo.policy.actor[-1].out_features == 2
    assert agent.slow_agent.count_ppo.policy.action_dim == 4
    assert agent.slow_agent.count_ppo.value_coef == 0.0
    assert np.isclose(agent.slow_agent.count_ppo.optimizer.param_groups[0]["lr"], 2e-4)
    assert np.isclose(agent.slow_agent.count_ppo.target_kl, 0.015)
    assert agent.slow_agent.placement_ppo.policy.actor[-1].out_features == 1
    assert agent.slow_agent.placement_ppo.policy.detach_critic_backbone
    assert np.isclose(agent.slow_agent.placement_ppo.optimizer.param_groups[0]["lr"], 1.5e-4)
    assert np.isclose(agent.slow_agent.placement_ppo.target_kl, 0.015)
    assert np.isclose(agent.slow_agent.placement_ppo.entropy_coef, 0.005)
    assert np.isclose(agent.slow_agent.placement_entropy_coefficient(), 0.005)
    agent.slow_agent.placement_updates_completed = 64
    assert np.isclose(agent.slow_agent.placement_entropy_schedule_coefficient(), 0.005)
    agent.slow_agent.placement_updates_completed = 96
    assert np.isclose(agent.slow_agent.placement_entropy_schedule_coefficient(), 0.00425)
    agent.slow_agent.placement_updates_completed = 128
    assert np.isclose(agent.slow_agent.placement_entropy_schedule_coefficient(), 0.0035)
    agent.slow_agent.placement_updates_completed = 0
    assert np.isclose(agent.fast_agent.ppo.optimizer.param_groups[0]["lr"], 2e-4)
    assert np.isclose(agent.fast_agent.ppo.entropy_coef, 0.001)
    assert agent.slow_agent.count_global_advantage_coef == 0.25
    assert agent.slow_agent.placement_global_advantage_coef == 0.35
    assert agent.slow_agent.placement_global_attribution_coef == 0.50
    assert agent.slow_agent.global_advantage_ev_full == 0.20
    assert agent.slow_agent.critic_replay_windows == 96
    assert agent.slow_agent.critic_holdout_windows == 12
    assert agent.slow_agent.critic_holdout_episodes == 2
    assert agent.slow_agent.critic_k_epochs == 8
    assert np.isclose(agent.slow_agent.critic_optimizer.param_groups[0]["lr"], 5e-4)
    agent.slow_agent.last_window_critic_holdout_ev = -0.1
    assert agent.slow_agent.global_advantage_reliability() == 0.0
    agent.slow_agent.last_window_critic_holdout_ev = 0.1
    assert np.isclose(agent.slow_agent.global_advantage_reliability(), 0.5)
    agent.slow_agent.last_window_critic_holdout_ev = 0.3
    assert agent.slow_agent.global_advantage_reliability() == 1.0
    agent.slow_agent.last_window_critic_holdout_ev = 0.0
    assert agent.slow_idle_replica_coef == 0.05
    assert agent.slow_placement_idle_coef == 0.02

    empty_fast_metrics = agent.fast_agent.update()
    assert np.isclose(empty_fast_metrics["entropy_coef"], 0.001)
    assert np.isclose(empty_fast_metrics["entropy_next_coef"], 0.001)
    assert np.isclose(agent.fast_agent.entropy_current_coef, 0.001)

    agent.slow_agent.placement_entropy_current_coef = 0.005
    adapted_coef = agent.slow_agent._adapt_placement_entropy_coefficient(observed_entropy=1.0)
    assert np.isclose(adapted_coef, 0.0066)

    def count_two(state, mask, deterministic=False):
        assert mask[1]
        return 1, 0.0, 0.0

    def first_available_node(state, mask, deterministic=False):
        return int(np.where(mask)[0][0]), 0.0, 0.0

    agent.slow_agent.count_ppo.act = count_two
    agent.slow_agent.placement_ppo.act = first_available_node
    deployment = agent.slow_agent.plan_deployment(env, deterministic=False, record=False)

    for service in env.scenario.services:
        for stage in service.stages:
            assert deployment[service.service_id, stage.stage_id].sum() == 2
    feasible, reason = env.check_deployment_feasible(deployment)
    assert feasible, reason


def test_deterministic_slow_deployment_respects_count_policy():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=31,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)

    def count_three_probabilities(state, mask):
        assert mask[2]
        probabilities = np.zeros_like(mask, dtype=np.float32)
        probabilities[2] = 1.0
        return probabilities, 0.0

    def first_available_node(state, mask, deterministic=False):
        return int(np.where(mask)[0][0]), 0.0, 0.0

    agent.slow_agent.count_ppo.action_probabilities = count_three_probabilities
    agent.slow_agent.placement_ppo.act = first_available_node
    deployment = agent.slow_agent.plan_deployment(env, deterministic=True, record=False)

    for service in env.scenario.services:
        for stage in service.stages:
            assert deployment[service.service_id, stage.stage_id].sum() == 3
    feasible, reason = env.check_deployment_feasible(deployment)
    assert feasible, reason


def test_slow_count_policy_is_unimodal_over_ordered_replica_actions():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=30,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=8)
    state = np.zeros(slow_obs_dim(16, 3), dtype=np.float32)
    probabilities, _ = agent.slow_agent.count_ppo.action_probabilities(
        state,
        np.ones(8, dtype=bool),
    )

    peak = int(np.argmax(probabilities))
    assert np.all(np.diff(probabilities[: peak + 1]) >= -1e-7)
    assert np.all(np.diff(probabilities[peak:]) <= 1e-7)
    assert np.isclose(probabilities.sum(), 1.0)
    initial_scale = float(np.exp(agent.slow_agent.count_ppo.policy.actor[-1].bias[1].item()))
    assert np.isclose(initial_scale, 8.0 / 6.0)
    assert np.isclose(agent.slow_agent.count_ppo.policy.count_min_scale, 1.0)


def test_deterministic_slow_count_uses_conservative_distribution_mean():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=32,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)

    def uniform_count_probabilities(state, mask):
        probabilities = mask.astype(np.float32)
        probabilities /= probabilities.sum()
        return probabilities, 0.0

    def first_available_node(state, mask, deterministic=False):
        return int(np.where(mask)[0][0]), 0.0, 0.0

    agent.slow_agent.count_ppo.action_probabilities = uniform_count_probabilities
    agent.slow_agent.placement_ppo.act = first_available_node
    deployment = agent.slow_agent.plan_deployment(env, deterministic=True, record=False)

    # Uniform support over 1..4 has expectation 2.5; ceiling avoids the old
    # arbitrary low-count argmax when Count has not learned yet.
    for service in env.scenario.services:
        for stage in service.stages:
            assert deployment[service.service_id, stage.stage_id].sum() == 3


def test_slow_count_return_uses_dense_latency_and_continuous_replica_credit():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=33,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=600.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        slow_reward_scale=1.0,
        slow_count_latency_coef=1.0,
        slow_idle_replica_coef=1.0,
        slow_count_unused_replica_coef=0.5,
        slow_count_shortage_coef=0.0,
        slow_deployment_memory_coef=0.0,
        slow_deployment_storage_coef=0.0,
        slow_deadline_violation_coef=0.0,
    )
    service = env.scenario.services[0]
    stage = service.stages[0]
    stage_key = (service.service_id, stage.stage_id)
    env.deployment[service.service_id, stage.stage_id] = False
    env.deployment[service.service_id, stage.stage_id, :4] = True
    agent.window_stage_latency_samples[stage_key] = [(0.2, 100.0)]
    agent.window_stage_weights[stage_key] = 100.0

    agent.window_stage_node_weights[stage_key] = {node_id: 25.0 for node_id in range(4)}
    balanced_returns, _, balanced_metrics = agent._factorized_stage_returns(env)
    assert np.isclose(balanced_metrics["slow_count_effective_replicas_per_stage"], 4.0)
    assert np.isclose(balanced_metrics["slow_count_redundant_replica_fraction"], 0.0)
    assert np.isclose(balanced_metrics["slow_count_unused_replica_fraction"], 0.0)
    assert np.isclose(balanced_returns[stage_key], -0.2)

    agent.window_stage_node_weights[stage_key] = {0: 100.0}
    concentrated_returns, _, concentrated_metrics = agent._factorized_stage_returns(env)
    assert np.isclose(concentrated_metrics["slow_count_effective_replicas_per_stage"], 1.0)
    assert np.isclose(concentrated_metrics["slow_count_redundant_replica_fraction"], 0.75)
    assert np.isclose(concentrated_metrics["slow_count_unused_replica_fraction"], 0.75)
    assert np.isclose(concentrated_metrics["slow_count_unused_replica_cost"], 0.375)
    assert np.isclose(concentrated_metrics["slow_count_replica_imbalance_fraction"], 0.0)
    assert np.isclose(concentrated_metrics["slow_count_replica_efficiency_cost"], 0.375)
    assert np.isclose(concentrated_returns[stage_key], -0.575)

    agent.window_stage_node_weights[stage_key] = {0: 75.0, 1: 25.0}
    imbalanced_returns, _, imbalanced_metrics = agent._factorized_stage_returns(env)
    assert np.isclose(imbalanced_metrics["slow_count_effective_replicas_per_stage"], 1.6)
    assert np.isclose(imbalanced_metrics["slow_count_redundant_replica_fraction"], 0.6)
    assert np.isclose(imbalanced_metrics["slow_count_unused_replica_fraction"], 0.5)
    assert np.isclose(imbalanced_metrics["slow_count_replica_imbalance_fraction"], 0.1)
    assert np.isclose(imbalanced_metrics["slow_count_replica_efficiency_cost"], 0.35)
    assert np.isclose(imbalanced_returns[stage_key], -0.55)

    # Legacy configurations without explicit unused-replica credit retain the
    # former total-redundancy cost exactly.
    agent.slow_count_unused_replica_coef = 0.0
    agent.window_stage_node_weights[stage_key] = {0: 100.0}
    legacy_returns, _, legacy_metrics = agent._factorized_stage_returns(env)
    assert np.isclose(legacy_metrics["slow_count_replica_efficiency_cost"], 0.75)
    assert np.isclose(legacy_returns[stage_key], -0.95)


def test_slow_window_return_includes_tail_latency_feedback():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=37,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        slow_tail_latency_coef=0.5,
        slow_deadline_violation_coef=0.0,
    )
    agent.slow_agent.pending_window_id = 0
    agent.observe_step_reward(
        reward=-0.1,
        stage_count=0,
        done=False,
        weight=1.0,
        latency_s=0.1,
        penalty_latency_s=0.0,
        deadline_s=0.2,
    )
    agent.observe_step_reward(
        reward=-0.5,
        stage_count=0,
        done=False,
        weight=1.0,
        latency_s=0.5,
        penalty_latency_s=0.0,
        deadline_s=0.2,
    )
    agent.flush_slow_window_reward(done=True)

    assert np.isclose(agent.slow_agent.last_window_feedback["avg_latency_s"], 0.3)
    assert np.isclose(agent.slow_agent.last_window_feedback["p95_latency_s"], 0.5)
    assert np.isclose(agent.last_slow_window_metrics["slow_tail_latency_cost"], 0.4)
    assert np.isclose(agent.slow_agent.window_returns[0], -0.4)


def test_slow_reward_tracks_cross_stage_colocation():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=38,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        slow_colocation_coef=0.05,
    )
    agent.slow_agent.pending_window_id = 0
    agent.observe_step_reward(
        reward=-0.1,
        stage_count=0,
        done=False,
        weight=1.0,
        latency_s=0.1,
        cross_stage_transitions=2.0,
        stage_transitions=4.0,
    )
    agent.flush_slow_window_reward(done=True)

    assert np.isclose(agent.last_slow_window_metrics["slow_cross_stage_transition_rate"], 0.5)
    assert np.isclose(agent.last_slow_window_metrics["slow_colocation_rate"], 0.5)
    assert np.isclose(agent.last_slow_window_metrics["slow_colocation_cost"], 0.025)
    assert np.isclose(agent.slow_agent.window_returns[0], -0.1 - 0.025)


def test_slow_colocation_credit_is_weighted_by_transition_data():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=381,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        slow_colocation_coef=0.11,
    )
    service = next(service for service in env.scenario.services if len(service.stages) >= 2)
    agent.slow_agent.pending_window_id = 0
    common = {
        "stage_count": 2,
        "done": False,
        "record_fast": False,
        "weight": 1.0,
        "latency_s": 0.1,
        "deadline_s": 1.0,
        "stage_transitions": 1.0,
        "service_id": service.service_id,
        "slow_stage_costs": [0.05, 0.05],
    }
    agent.observe_step_reward(
        reward=-0.1,
        cross_stage_transitions=1.0,
        stage_nodes=[0, 1],
        stage_transition_data_mb=[10.0],
        **common,
    )
    agent.observe_step_reward(
        reward=-0.1,
        cross_stage_transitions=0.0,
        stage_nodes=[0, 0],
        stage_transition_data_mb=[1.0],
        **common,
    )
    agent.flush_slow_window_reward(done=True, env=env)

    assert np.isclose(agent.last_slow_window_metrics["slow_cross_stage_transition_rate"], 0.5)
    assert np.isclose(agent.last_slow_window_metrics["slow_data_weighted_cross_stage_rate"], 10.0 / 11.0)
    assert np.isclose(agent.last_slow_window_metrics["slow_colocation_cost"], 0.1)


def test_slow_shared_return_reaches_factorized_actions_without_critic_gate():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=382,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        slow_shared_return_coef=0.25,
        slow_tail_latency_coef=0.0,
        slow_colocation_coef=0.0,
        slow_deadline_violation_coef=0.0,
    )
    service = next(service for service in env.scenario.services if len(service.stages) >= 2)
    agent.slow_agent.pending_window_id = 0
    agent.observe_step_reward(
        reward=-0.4,
        stage_count=2,
        done=False,
        record_fast=False,
        weight=1.0,
        latency_s=0.4,
        deadline_s=1.0,
        service_id=service.service_id,
        stage_nodes=[0, 0],
        slow_stage_costs=[0.2, 0.2],
    )
    agent.flush_slow_window_reward(done=True, env=env)

    assert np.isclose(agent.last_slow_window_metrics["slow_window_return"], -0.4)
    assert np.isclose(agent.last_slow_window_metrics["slow_shared_return_credit"], -0.1)


def test_factorized_placement_return_receives_local_colocation_credit():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=39,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        slow_reward_scale=1.0,
        slow_tail_latency_coef=0.0,
        slow_colocation_coef=0.2,
        slow_placement_compute_coef=0.0,
        slow_deadline_violation_coef=0.0,
    )
    service = next(service for service in env.scenario.services if len(service.stages) >= 2)
    agent.observe_step_reward(
        reward=-0.2,
        stage_count=2,
        done=False,
        record_fast=False,
        weight=1.0,
        latency_s=0.2,
        deadline_s=1.0,
        cross_stage_transitions=1.0,
        stage_transitions=1.0,
        service_id=service.service_id,
        stage_nodes=[0, 1],
        slow_stage_costs=[0.1, 0.1],
    )

    _, placement_returns, _ = agent._factorized_stage_returns(env)

    first_key = (service.service_id, 0)
    second_key = (service.service_id, 1)
    assert np.isclose(agent.window_stage_cross_transitions[first_key], 1.0)
    assert np.isclose(agent.window_stage_cross_transitions[second_key], 1.0)
    assert np.isclose(placement_returns[first_key], -0.3)
    assert np.isclose(placement_returns[second_key], -0.3)


def test_factorized_placement_returns_distinguish_selected_nodes():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=40,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=60.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=4,
        slow_reward_scale=1.0,
        slow_tail_latency_coef=0.0,
        slow_colocation_coef=0.2,
        slow_placement_idle_coef=0.1,
        slow_placement_compute_coef=0.5,
        slow_deadline_violation_coef=0.0,
    )
    service = next(service for service in env.scenario.services if len(service.stages) >= 2)
    env.deployment[service.service_id, 0] = False
    env.deployment[service.service_id, 1] = False
    env.deployment[service.service_id, 0, [0, 2]] = True
    env.deployment[service.service_id, 1, [1, 2]] = True
    env.node_compute_load[[0, 1, 2]] = [0.4, 0.6, 0.1]
    agent.observe_step_reward(
        reward=-0.4,
        stage_count=2,
        done=False,
        record_fast=False,
        weight=3.0,
        latency_s=0.4,
        deadline_s=1.0,
        cross_stage_transitions=3.0,
        stage_transitions=3.0,
        service_id=service.service_id,
        stage_nodes=[0, 1],
        slow_stage_costs=[0.1, 0.3],
    )
    agent.observe_step_reward(
        reward=-0.4,
        stage_count=2,
        done=False,
        record_fast=False,
        weight=1.0,
        latency_s=0.4,
        deadline_s=1.0,
        cross_stage_transitions=0.0,
        stage_transitions=1.0,
        service_id=service.service_id,
        stage_nodes=[2, 2],
        slow_stage_costs=[0.2, 0.2],
    )

    _, placement_returns, credit_metrics = agent._factorized_stage_returns(env)

    assert np.isclose(placement_returns[(service.service_id, 0, 0)], -0.5)
    assert np.isclose(placement_returns[(service.service_id, 0, 2)], -0.25)
    assert np.isclose(placement_returns[(service.service_id, 1, 1)], -0.8)
    assert np.isclose(placement_returns[(service.service_id, 1, 2)], -0.25)
    assert np.isclose(credit_metrics["slow_placement_node_compute_load"], 0.3)
    assert np.isclose(credit_metrics["slow_placement_node_compute_cost"], 0.15)
