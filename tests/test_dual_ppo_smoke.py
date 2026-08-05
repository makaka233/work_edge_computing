import numpy as np

from edge_drl.agents.drl import HierarchicalPPOAgent, fast_obs_dim, slow_obs_dim
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from train_dual_ppo import rollout


def test_observation_dimensions():
    assert slow_obs_dim(16, 3) == 8 + 16 * 5 + 16 * 3
    assert fast_obs_dim(16) == 9 + 16 * 5 + 16 * 16 * 3


def test_dual_ppo_rollout_and_update():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=11,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=2.0,
            request_aggregation_window_seconds=0.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    assert agent.slow_migration_coef == 0

    request_count = 0
    while request_count < 6 and not env.done:
        request = env.current_request
        assert request is not None
        action = agent.act(env)
        _, reward, done, info = env.step(action)
        agent.observe_step_reward(reward, len(request.stage_compute_gcycles), done, weight=request.request_count)
        assert len(action) == len(request.stage_compute_gcycles)
        assert np.isfinite(reward)
        assert info["latency_s"] >= 0
        request_count += 1

    agent.flush_slow_window_reward(done=True)
    assert len(agent.slow_agent.count_ppo.buffer) > 0
    assert len(agent.slow_agent.placement_ppo.buffer) > 0
    assert len(agent.slow_agent.window_returns) == 1
    assert len(agent.slow_agent.count_ppo.buffer.rewards) == 0
    assert len(agent.slow_agent.placement_ppo.buffer.rewards) == 0
    assert len(agent.fast_agent.ppo.buffer) >= request_count

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


def test_deterministic_eval_does_not_write_rollout_buffers():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=13,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=2.0,
            request_aggregation_window_seconds=0.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    stats = rollout(env, agent, max_requests=4, deterministic=True, record=False)
    assert stats["requests"] == 4
    assert len(agent.slow_agent.count_ppo.buffer) == 0
    assert len(agent.slow_agent.placement_ppo.buffer) == 0
    assert len(agent.fast_agent.ppo.buffer) == 0


def test_fast_only_rollout_records_only_fast_buffer():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=19,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=2.0,
            request_aggregation_window_seconds=0.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    stats = rollout(env, agent, max_requests=5, train_mode="fast-only", reward_scale=0.1)
    assert stats["requests"] == 5
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
            mean_requests_per_minute=2.0,
            request_aggregation_window_seconds=0.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4, fast_policy_kind="node_scorer")
    stats = rollout(env, agent, max_requests=4, train_mode="fast-only", reward_scale=0.1)
    assert stats["requests"] == 4
    assert len(agent.fast_agent.ppo.buffer) >= 4


def test_slow_agent_uses_explicit_count_and_unique_placements():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=29,
            num_users=10_000,
            num_edge_nodes=32,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=2.0,
            request_aggregation_window_seconds=0.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    assert agent.slow_agent.count_ppo.policy.actor.out_features == 4
    assert agent.slow_agent.placement_ppo.policy.actor.out_features == env.config.num_edge_nodes

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
            mean_requests_per_minute=2.0,
            request_aggregation_window_seconds=0.0,
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
