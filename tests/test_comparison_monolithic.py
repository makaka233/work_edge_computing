from types import SimpleNamespace

import numpy as np

from edge_drl.comparison.monolithic import (
    MONOLITHIC_FAST_DESIGN,
    MONOLITHIC_FAST_DESIGN_V2,
    MONOLITHIC_FAST_DESIGN_V3,
    MonolithicFastSchedulingPPOAgent,
    MonolithicScheme,
    MonolithicTemperatureGATPolicy,
    StableMonolithicFastSchedulingPPOAgent,
    adapt_request_for_monolithic_evaluation,
    collapse_scenario,
    collapse_trace,
    collapse_deployment_for_monolithic,
    compact_request_for_monolithic_policy,
    expand_monolithic_schedule,
    specialize_monolithic_fast_agent,
)
from edge_drl.agents.drl import FastSchedulingPPOAgent, HierarchicalPPOAgent, fast_obs_dim
from edge_drl.models.ppo import GraphAttentionNodeScoringActorCritic
from tests.comparison_helpers import tiny_replay_env
from train_monolithic import _optimized_monolithic_args


def test_monolithic_collapses_each_service_to_one_aggregated_stage() -> None:
    env = tiny_replay_env()
    env.reset()
    mono = collapse_scenario(env.scenario)
    for original, aggregated in zip(env.scenario.services, mono.services):
        assert len(aggregated.stages) == 1
        assert aggregated.stages[0].compute_gcycles_mean == sum(
            stage.compute_gcycles_mean for stage in original.stages
        )
        assert aggregated.stages[0].memory_gb == sum(stage.memory_gb for stage in original.stages)
        assert aggregated.stages[0].storage_gb == sum(stage.storage_gb for stage in original.stages)
        assert aggregated.stages[0].output_mb_mean == 0.0
    collapsed = collapse_trace(env._comparison_trace)
    for slot in collapsed.slots:
        for request in slot:
            assert len(request.stage_compute_gcycles) == 1
            assert request.stage_output_mb == (0.0,)


def test_monolithic_evaluation_preserves_stage_shape_and_shared_node_projection() -> None:
    env = tiny_replay_env()
    env.reset()
    original = next(request for slot in env._comparison_trace.slots for request in slot)
    adapted = adapt_request_for_monolithic_evaluation(original)
    assert len(adapted.stage_compute_gcycles) == len(original.stage_compute_gcycles)
    assert adapted.stage_compute_gcycles[0] == sum(original.stage_compute_gcycles)
    assert adapted.stage_compute_gcycles[1:] == (0.0,) * (len(original.stage_compute_gcycles) - 1)
    assert adapted.stage_output_mb == (0.0,) * len(original.stage_output_mb)
    assert expand_monolithic_schedule([4], len(adapted.stage_compute_gcycles)) == [4, 4]


def test_monolithic_optimization_uses_strict_ledger_and_optional_full_resources() -> None:
    base = {
        "fast_reservation_microbatch_size": 16,
        "fast_lr": 2e-4,
        "fast_entropy_target": 0.18,
        "fast_entropy_normalized_control": True,
        "slow_deployment_memory_coef": 0.03,
        "slow_deployment_storage_coef": 0.01,
        "slow_idle_replica_coef": 0.05,
        "slow_count_unused_replica_coef": 0.015,
        "slow_placement_idle_coef": 0.02,
    }
    optimized = _optimized_monolithic_args(
        base,
        reservation_microbatch_size=1,
        allow_full_resource_use=True,
    )
    assert optimized["fast_reservation_microbatch_size"] == 1
    assert optimized["fast_lr"] == 2e-4
    assert optimized["fast_entropy_target"] == 0.18
    assert "fast_entropy_max_coef" not in optimized
    assert "fast_entropy_adaptation_rate" not in optimized
    assert optimized["fast_entropy_normalized_control"] is True
    assert optimized["monolithic_fast_design"] == MONOLITHIC_FAST_DESIGN
    assert optimized["monolithic_fast_logit_temperature_max"] == 3.0
    assert optimized["monolithic_allow_full_resource_use"] is True
    assert optimized["slow_deployment_memory_coef"] == 0.0
    assert optimized["slow_deployment_storage_coef"] == 0.0
    assert optimized["slow_idle_replica_coef"] == 0.01
    assert optimized["slow_count_unused_replica_coef"] == 0.005
    assert optimized["slow_placement_idle_coef"] == 0.005
    assert base["fast_reservation_microbatch_size"] == 16
    assert base["fast_lr"] == 2e-4


def test_monolithic_fast_specialization_is_opt_in_and_shape_compatible() -> None:
    env = tiny_replay_env()
    env.reset()
    shared_agent = HierarchicalPPOAgent.from_env(
        env,
        fast_chain_context="chain-kkt-v3",
    )
    untouched_agent = HierarchicalPPOAgent.from_env(
        env,
        fast_chain_context="chain-kkt-v3",
    )

    specialize_monolithic_fast_agent(shared_agent, MONOLITHIC_FAST_DESIGN)

    assert type(shared_agent.fast_agent) is MonolithicFastSchedulingPPOAgent
    assert type(shared_agent.fast_agent.ppo.policy) is GraphAttentionNodeScoringActorCritic
    assert type(untouched_agent.fast_agent) is FastSchedulingPPOAgent
    original = next(request for slot in env._comparison_trace.slots for request in slot)
    compact = compact_request_for_monolithic_policy(original)
    env.current_requests = [compact]
    env.current_request = compact
    state = shared_agent.fast_agent._build_state(
        env,
        compact,
        0,
        [],
        kkt_compute_sqrt=np.zeros((1, env.config.num_edge_nodes)),
        kkt_link_sqrt=np.zeros(
            (1, env.config.num_edge_nodes, env.config.num_edge_nodes)
        ),
    )
    assert state.shape == (
        fast_obs_dim(env.config.num_edge_nodes, chain_context="chain-kkt-v3"),
    )
    assert np.all(np.isfinite(state))


def test_monolithic_fast_v2_checkpoint_keeps_original_observation_specialization() -> None:
    env = tiny_replay_env()
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        fast_chain_context="chain-kkt-v3",
    )

    specialize_monolithic_fast_agent(agent, MONOLITHIC_FAST_DESIGN_V2)

    assert type(agent.fast_agent) is MonolithicFastSchedulingPPOAgent
    assert type(agent.fast_agent.ppo.policy) is GraphAttentionNodeScoringActorCritic


def test_stable_monolithic_temperature_tracks_entropy_controller() -> None:
    env = tiny_replay_env()
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        fast_chain_context="chain-kkt-v3",
        fast_entropy_coef=0.001,
        fast_entropy_max_coef=0.04,
    )
    specialize_monolithic_fast_agent(
        agent,
        MONOLITHIC_FAST_DESIGN_V3,
        logit_temperature_max=3.0,
    )
    fast = agent.fast_agent

    assert fast._sync_logit_temperature() == 1.0
    fast.entropy_current_coef = fast.entropy_max_coef
    assert np.isclose(fast._sync_logit_temperature(), 3.0)
    assert np.isclose(fast.ppo.policy.logit_temperature, 3.0)


def test_monolithic_v4_marks_only_the_end_of_each_request_batch_terminal() -> None:
    env = tiny_replay_env()
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        fast_chain_context="chain-kkt-v3",
    )
    specialize_monolithic_fast_agent(agent, MONOLITHIC_FAST_DESIGN)
    fast = agent.fast_agent
    fast._batch_reward_queue().append(2)

    fast.assign_schedule_rewards([-0.2], done=False, weight=3.0)
    fast.assign_schedule_rewards([-0.3], done=False, weight=4.0)

    assert fast.ppo.buffer.rewards[-2:] == [-0.2, -0.3]
    assert fast.ppo.buffer.dones[-2:] == [False, True]
    assert fast.ppo.buffer.weights[-2:] == [3.0, 4.0]
    assert fast._batch_reward_queue() == []


def test_monolithic_v4_observation_tracks_remaining_batch_demand() -> None:
    env = tiny_replay_env()
    env.reset()
    requests = [
        compact_request_for_monolithic_policy(request)
        for slot in env._comparison_trace.slots
        for request in slot
    ][:2]
    assert len(requests) == 2
    env.current_requests = requests
    env.current_request = requests[0]
    agent = HierarchicalPPOAgent.from_env(
        env,
        fast_chain_context="chain-kkt-v3",
    )
    specialize_monolithic_fast_agent(agent, MONOLITHIC_FAST_DESIGN)
    zeros_compute = np.zeros((1, env.config.num_edge_nodes))
    zeros_link = np.zeros(
        (1, env.config.num_edge_nodes, env.config.num_edge_nodes)
    )

    full_state = agent.fast_agent._build_state(
        env,
        requests[0],
        0,
        [],
        kkt_compute_sqrt=zeros_compute,
        kkt_link_sqrt=zeros_link,
        remaining_requests=requests,
    )
    tail_state = agent.fast_agent._build_state(
        env,
        requests[0],
        0,
        [],
        kkt_compute_sqrt=zeros_compute,
        kkt_link_sqrt=zeros_link,
        remaining_requests=requests[1:],
    )

    assert full_state.shape == tail_state.shape
    assert not np.array_equal(full_state, tail_state)


def test_monolithic_previous_deployment_keeps_only_complete_service_replicas() -> None:
    env = tiny_replay_env()
    env.reset()
    assert env.scenario is not None
    deployment = np.zeros_like(env.deployment, dtype=bool)
    deployment[0, 0, 1] = True
    deployment[0, 1, 1] = True
    deployment[0, 0, 2] = True

    collapsed = collapse_deployment_for_monolithic(env.scenario, deployment)

    assert collapsed[0, 0, 1]
    assert not collapsed[0, 0, 2]
    assert not np.any(collapsed[0, 1:])


def test_monolithic_evaluation_builds_previous_window_slow_feedback() -> None:
    env = tiny_replay_env()
    env.reset()
    request = next(
        request
        for slot in env._comparison_trace.slots
        for request in slot
    )
    scheme = object.__new__(MonolithicScheme)
    scheme.agent = SimpleNamespace(
        slow_agent=SimpleNamespace(last_window_feedback={})
    )
    scheme._window_latency_samples = []
    scheme._window_penalty_latency_sum = 0.0
    scheme._window_deadline_violations = 0.0
    scheme._window_invalid_actions = 0.0
    scheme._window_weight = 0.0
    scheme._window_max_node_load = 0.0
    scheme._window_max_link_load = 0.0
    scheme._window_memory_fraction = 0.4
    scheme._window_storage_fraction = 0.5

    scheme.observe_step(
        env,
        [request],
        [[0] * len(request.stage_compute_gcycles)],
        {
            "group_infos": [
                {
                    "request_count": 3.0,
                    "latency_s": 0.25,
                    "penalty_latency_s": 0.05,
                    "valid": True,
                }
            ]
        },
    )
    scheme._finalize_window_feedback()

    feedback = scheme.agent.slow_agent.last_window_feedback
    assert feedback["avg_latency_s"] == 0.25
    assert feedback["p95_latency_s"] == 0.25
    assert np.isclose(feedback["avg_penalty_latency_s"], 0.05)
    assert feedback["deployment_memory_fraction"] == 0.4
    assert feedback["deployment_storage_fraction"] == 0.5
    assert scheme._window_weight == 0.0
