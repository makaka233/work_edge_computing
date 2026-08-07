import numpy as np

from edge_drl.agents.drl import HierarchicalPPOAgent, fast_obs_dim, slow_obs_dim
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from train_dual_ppo import rollout


def test_observation_dimensions():
    assert slow_obs_dim(16, 3) == 17 + 16 * (6 + 3) + 16 * 16 * 4
    assert fast_obs_dim(16) == 12 + 16 * 9 + 16 * 16 * 3


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
    assert losses["slow"]["window_count"] == 1
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
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    requests = list(env.current_requests)
    actions = agent.fast_agent.schedule_batch(env, requests, deterministic=True, record=True)

    expected_transitions = sum(len(request.stage_compute_gcycles) for request in requests)
    assert len(agent.fast_agent.ppo.buffer) == expected_transitions
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
    assert agent.slow_agent.count_ppo.policy.actor[-1].out_features == 4
    assert agent.slow_agent.placement_ppo.policy.actor[-1].out_features == 1

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

    def count_three(state, mask, deterministic=False):
        assert mask[2]
        return 2, 0.0, 0.0

    def first_available_node(state, mask, deterministic=False):
        return int(np.where(mask)[0][0]), 0.0, 0.0

    agent.slow_agent.count_ppo.act = count_three
    agent.slow_agent.placement_ppo.act = first_available_node
    deployment = agent.slow_agent.plan_deployment(env, deterministic=True, record=False)

    for service in env.scenario.services:
        for stage in service.stages:
            assert deployment[service.service_id, stage.stage_id].sum() == 3
    feasible, reason = env.check_deployment_feasible(deployment)
    assert feasible, reason


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
