import numpy as np

from edge_drl.agents.drl import HierarchicalPPOAgent, fast_obs_dim, slow_obs_dim
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig
from train_dual_ppo import rollout


def test_observation_dimensions():
    assert slow_obs_dim(16, 3) == 6 + 16 * 5 + 16 * 3
    assert fast_obs_dim(16) == 8 + 16 * 5


def test_dual_ppo_rollout_and_update():
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=11,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
            mean_requests_per_minute=2.0,
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)

    request_count = 0
    while request_count < 6 and not env.done:
        request = env.current_request
        assert request is not None
        action = agent.act(env)
        _, reward, done, info = env.step(action)
        agent.observe_step_reward(reward, len(request.stage_compute_gcycles), done)
        assert len(action) == len(request.stage_compute_gcycles)
        assert np.isfinite(reward)
        assert info["latency_s"] >= 0
        request_count += 1

    agent.flush_slow_window_reward(done=True)
    assert len(agent.slow_agent.ppo.buffer) > 0
    assert len(agent.fast_agent.ppo.buffer) >= request_count

    losses = agent.update()
    assert "slow" in losses
    assert "fast" in losses
    assert np.isfinite(losses["slow"]["loss"])
    assert np.isfinite(losses["fast"]["loss"])
    assert len(agent.slow_agent.ppo.buffer) == 0
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
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    stats = rollout(env, agent, max_requests=4, deterministic=True, record=False)
    assert stats["requests"] == 4
    assert len(agent.slow_agent.ppo.buffer) == 0
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
        )
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    stats = rollout(env, agent, max_requests=5, train_mode="fast-only", reward_scale=0.1)
    assert stats["requests"] == 5
    assert len(agent.slow_agent.ppo.buffer) == 0
    assert len(agent.fast_agent.ppo.buffer) >= 5
