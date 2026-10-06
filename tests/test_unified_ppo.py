from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import pytest

from edge_drl.agents.unified import UnifiedPPOAgent, unified_obs_dim
from edge_drl.allocators.kkt import ComputeDemand, LinkDemand
from edge_drl.allocators.learned_share import (
    allocate_compute_learned_share,
    allocate_link_learned_share,
)
from edge_drl.comparison.metrics import evaluate_scheme_episode
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.comparison.unified import UnifiedPPOScheme
from train_unified_ppo import atomic_torch_save, collect_window
from tests.comparison_helpers import tiny_replay_env


def test_learned_share_uses_policy_priority_without_kkt_workload_rule() -> None:
    compute = [
        ComputeDemand("low", 0, 4.0, allocation_weight=1.0),
        ComputeDemand("high", 0, 4.0, allocation_weight=3.0),
    ]
    allocations, delays, total = allocate_compute_learned_share(
        compute, np.asarray([8.0])
    )
    assert np.isclose(allocations["low"], 2.0)
    assert np.isclose(allocations["high"], 6.0)
    assert np.isclose(delays["low"], 2.0)
    assert np.isclose(delays["high"], 2.0 / 3.0)
    assert np.isclose(total, 8.0 / 3.0)

    link = [
        LinkDemand("low", 0, 1, 4.0, allocation_weight=1.0),
        LinkDemand("high", 0, 1, 4.0, allocation_weight=3.0),
    ]
    link_capacity = np.full((2, 2), np.inf)
    link_capacity[0, 1] = 8.0
    allocations, delays, total = allocate_link_learned_share(link, link_capacity)
    assert np.isclose(allocations["low"], 2.0)
    assert np.isclose(allocations["high"], 6.0)
    assert np.isclose(delays["low"], 2.0)
    assert np.isclose(delays["high"], 2.0 / 3.0)
    assert np.isclose(total, 8.0 / 3.0)


def test_environment_threads_learned_priorities_through_joint_settlement() -> None:
    env = tiny_replay_env()
    env.reset()
    deployment = np.ones_like(env.deployment, dtype=bool)
    env.apply_deployment(deployment)
    env.config.compute_allocation_mode = "learned-share"
    env.config.link_allocation_mode = "learned-share"

    source = next(request for slot in env._comparison_trace.slots for request in slot)
    request = replace(
        source,
        home_node=0,
        request_count=1,
        input_mb=0.0,
        stage_compute_gcycles=(4.0,) * len(source.stage_compute_gcycles),
        stage_output_mb=(0.0,) * len(source.stage_output_mb),
    )
    schedule = [0] * len(request.stage_compute_gcycles)
    low = [1.0] * len(schedule)
    high = [4.0] * len(schedule)
    infos = env.evaluate_batch_schedules(
        [request, request],
        [schedule, schedule],
        batch_resource_weights=[low, high],
    )
    assert all(info["valid"] for info in infos)
    assert infos[1]["compute_delay_s"] < infos[0]["compute_delay_s"]


def test_unified_agent_produces_feasible_actions_and_macro_credit() -> None:
    env = tiny_replay_env()
    env.reset()
    env.config.compute_allocation_mode = "learned-share"
    env.config.link_allocation_mode = "learned-share"
    agent = UnifiedPPOAgent.from_env(
        env,
        hidden_dim=32,
        minibatch_size=16,
        device="cpu",
    )
    assert agent.ppo.policy.shared[0].in_features == unified_obs_dim(
        env.config.num_edge_nodes, env.config.num_service_types, env.config.max_service_stages)

    start = agent.begin_environment_step()
    deployment = agent.plan_deployment(env, deterministic=True, record=True)
    feasible, reason = env.check_deployment_feasible(deployment)
    assert feasible, reason
    env.apply_deployment(deployment)

    requests = list(next(slot for slot in env._comparison_trace.slots if slot))
    env.current_requests = requests
    env.current_request = requests[0]
    schedules, weights = agent.schedule_batch(
        env, requests, deterministic=True, record=True
    )
    assert len(schedules) == len(requests) == len(weights)
    assert all(
        len(schedule) == len(request.stage_compute_gcycles)
        for schedule, request in zip(schedules, requests)
    )
    assert all(0.1 <= weight <= 10.0 for request_weights in weights
               for pair in request_weights for weight in pair)
    _, reward, _, info = env.step(schedules, batch_resource_weights=weights)
    assert all(group["valid"] for group in info["group_infos"])

    agent.finish_environment_step(start, reward, done=False, group_infos=info["group_infos"])
    agent.finish_window()
    new_returns = np.asarray(agent.action_returns[start:], dtype=np.float64)
    new_weights = np.asarray(agent.action_weights[start:], dtype=np.float64)
    new_modes = agent.action_modes[start:]
    assert len(new_returns) > 1
    deployment = np.asarray([mode != "schedule" for mode in new_modes])
    scheduling = ~deployment
    assert np.allclose(new_returns[deployment], reward * agent.reward_scale)
    assert np.isclose(new_returns[scheduling][0], reward * agent.reward_scale)
    assert np.all(np.diff(new_returns[scheduling]) >= -1e-8)
    assert np.isclose(new_weights[deployment].sum(), 1.0)
    assert np.isclose(new_weights[scheduling].sum(), 1.0)
    assert np.count_nonzero(agent.ppo.buffer.rewards[start:]) == 0
    metrics = agent.update()
    assert np.isfinite(metrics["loss"])
    assert metrics["samples_seen_fraction"] == 1.0
    assert metrics["ledger_settlement_error_s"] < 1e-6


def test_macro_credit_discounts_real_steps_not_component_actions() -> None:
    env = tiny_replay_env()
    env.reset()
    agent = UnifiedPPOAgent.from_env(
        env,
        hidden_dim=32,
        environment_gamma=0.5,
        reward_scale=1.0,
        device="cpu",
    )
    agent.begin_window()

    for reward in (-2.0, -4.0):
        start = agent.begin_environment_step()
        state = np.zeros(agent.ppo.policy.shared[0].in_features, dtype=np.float32)
        mask = np.ones(agent.action_dim, dtype=bool)
        agent._record_action(
            state=state,
            mask=mask,
            action=0,
            logprob=0.0,
            value=0.0,
            mode="schedule",
            env=env,
        )
        agent._record_action(
            state=state,
            mask=mask,
            action=0,
            logprob=0.0,
            value=0.0,
            mode="schedule",
            env=env,
        )
        agent.finish_environment_step(start, reward, done=False)

    agent.finish_window()
    # Step 1 target is the normalized discounted mean
    # (-2 + 0.5 * -4) / (1 + 0.5), shared by both component factors.
    assert np.allclose(agent.action_returns[:2], [-8.0 / 3.0, -8.0 / 3.0])
    assert np.allclose(agent.action_returns[2:], [-4.0, -4.0])
    assert np.allclose(agent.action_weights, [0.5, 0.5, 0.5, 0.5])


def test_unified_state_dict_round_trip() -> None:
    env = tiny_replay_env()
    env.reset()
    first = UnifiedPPOAgent.from_env(env, hidden_dim=32, device="cpu")
    second = UnifiedPPOAgent.from_env(env, hidden_dim=32, device="cpu")
    second.load_state_dict(first.state_dict())
    for left, right in zip(first.ppo.policy.parameters(), second.ppo.policy.parameters()):
        assert np.array_equal(left.detach().numpy(), right.detach().numpy())


def test_unified_checkpoint_adapter_completes_sampled_evaluation(tmp_path) -> None:
    training_env = tiny_replay_env()
    training_env.reset()
    agent = UnifiedPPOAgent.from_env(training_env, hidden_dim=32, device="cpu")
    checkpoint = tmp_path / "unified.pt"
    torch.save(
        {
            "agent": agent.state_dict(),
            "metadata": {"args": {"hidden_dim": 32}},
        },
        checkpoint,
    )

    evaluation_env = tiny_replay_env()
    scheme = UnifiedPPOScheme(evaluation_env, checkpoint, device="cpu")
    result = evaluate_scheme_episode(
        scheme=scheme,
        env=evaluation_env,
        point=ExperimentPoint("request_load", 1.0, "nominal"),
        eval_seed=2026,
        routing_repeat=0,
        sampled_seconds_per_window=6,
    )
    assert result.settlement_steps == 6
    assert result.logical_steps == 600
    assert result.invalid_action_rate == 0.0
    assert evaluation_env.config.compute_allocation_mode == "learned-share"
    assert evaluation_env.config.link_allocation_mode == "learned-share"


def test_six_sample_window_snaps_exactly_to_episode_boundary() -> None:
    env = tiny_replay_env()
    env.reset()
    env.config.compute_allocation_mode = "learned-share"
    env.config.link_allocation_mode = "learned-share"
    agent = UnifiedPPOAgent.from_env(env, hidden_dim=32, device="cpu")

    row = collect_window(
        env,
        agent,
        sampled_seconds=6,
        deterministic=True,
        record=False,
        progress_label="",
        progress_interval_seconds=0.0,
    )

    assert row["settlement_steps"] == 6.0
    assert row["logical_steps"] == 600.0
    assert env.current_time_minute == 10.0
    assert env.done


def test_atomic_checkpoint_save_retries_transient_windows_lock(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "latest.pt"
    torch.save({"old": True}, target)
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(self, destination):
        nonlocal attempts
        if self.name == "latest.pt.tmp" and attempts < 2:
            attempts += 1
            raise PermissionError(5, "simulated Windows lock")
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    saved = atomic_torch_save({"new": True}, target)

    assert saved == target
    assert attempts == 2
    assert torch.load(target, map_location="cpu", weights_only=False) == {"new": True}
    assert not target.with_suffix(".pt.tmp").exists()


def test_paired_priorities_control_compute_and_link_independently() -> None:
    env = tiny_replay_env()
    env.reset()
    env.apply_deployment(np.ones_like(env.deployment, dtype=bool))
    env.config.compute_allocation_mode = "learned-share"
    env.config.link_allocation_mode = "learned-share"
    source = next(request for slot in env._comparison_trace.slots for request in slot)
    request = replace(source, home_node=0, request_count=1, input_mb=4.0,
                      stage_compute_gcycles=(4.,) * len(source.stage_compute_gcycles))
    schedule = [1] * len(request.stage_compute_gcycles)
    first = [(1., 4.)] * len(schedule)
    second = [(4., 1.)] * len(schedule)
    infos = env.evaluate_batch_schedules([request, request], [schedule, schedule],
                                        batch_resource_weights=[first, second])
    assert infos[0]["compute_delay_s"] > infos[1]["compute_delay_s"]
    assert infos[0]["link_delay_s"] < infos[1]["link_delay_s"]


def test_exact_increment_ledger_telescopes_for_multiplicities_and_routes() -> None:
    env = tiny_replay_env()
    env.reset()
    env.apply_deployment(np.ones_like(env.deployment, dtype=bool))
    env.config.compute_allocation_mode = "learned-share"
    env.config.link_allocation_mode = "learned-share"
    agent = UnifiedPPOAgent.from_env(env, hidden_dim=32)
    source = next(request for slot in env._comparison_trace.slots for request in slot)
    requests = [replace(source, home_node=0, request_count=count,
                        input_mb=2.0 + count,
                        stage_compute_gcycles=tuple(count + 1. for _ in source.stage_compute_gcycles))
                for count in (1, 3, 2)]
    schedules = [[(stage + index) % env.config.num_edge_nodes
                  for stage in range(len(source.stage_compute_gcycles))] for index in range(3)]
    weights = [[(.3 + index, 4. - index)] * len(schedule)
               for index, schedule in enumerate(schedules)]
    increments = []
    for request, schedule, priorities in zip(requests, schedules, weights):
        node = request.home_node
        for stage, (target, pair) in enumerate(zip(schedule, priorities)):
            increments.append(agent._add_stage_cost(env, request, stage, node, target, pair))
            node = target
    infos = env.evaluate_batch_schedules(requests, schedules, batch_resource_weights=weights)
    objective = sum(r.request_count * info["latency_s"] for r, info in zip(requests, infos))
    assert np.isclose(sum(increments), objective, rtol=1e-8)
    assert all(cost >= 0. for cost in increments)


def test_recorded_joint_log_probability_matches_policy_and_live_ledger() -> None:
    env = tiny_replay_env()
    env.reset()
    env.apply_deployment(np.ones_like(env.deployment, dtype=bool))
    agent = UnifiedPPOAgent.from_env(env, hidden_dim=32)
    source = next(request for slot in env._comparison_trace.slots for request in slot)
    requests = [source, source]
    agent.schedule_batch(env, requests, deterministic=False, record=True)
    buffer = agent.ppo.buffer
    logs, _, _, _, _ = agent.ppo.policy.evaluate(
        torch.as_tensor(np.asarray(buffer.states)), torch.as_tensor(np.asarray(buffer.masks)),
        torch.as_tensor(buffer.actions), torch.as_tensor(np.asarray(agent.ppo.units)),
        torch.as_tensor(np.asarray(agent.ppo.active)))
    assert np.allclose(logs.detach().numpy(), buffer.logprobs, atol=1e-6)
    # Candidate features for request 2 must include request 1's reservations.
    first, second = buffer.states[0], buffer.states[len(source.stage_compute_gcycles)]
    node_feature_start = 21
    node_feature_stride = 21
    first_mass = [first[node_feature_start + node * node_feature_stride + 16]
                  for node in range(env.config.num_edge_nodes)]
    second_mass = [second[node_feature_start + node * node_feature_stride + 16]
                   for node in range(env.config.num_edge_nodes)]
    assert max(first_mass) == 0.
    assert max(second_mass) > 0.


def test_recorded_sampled_window_and_second_update_close_without_stale_buffers() -> None:
    env = tiny_replay_env()
    env.config.compute_allocation_mode = "learned-share"
    env.config.link_allocation_mode = "learned-share"
    agent = UnifiedPPOAgent.from_env(env, hidden_dim=32, minibatch_size=32, k_epochs=2)
    for _ in range(2):
        env.reset()
        row = collect_window(env, agent, sampled_seconds=6, deterministic=False,
                             record=True, progress_label="", progress_interval_seconds=0.)
        assert row["settlement_steps"] == 6.
        metrics = agent.update()
        assert np.isfinite(metrics["loss"])
        assert metrics["ledger_settlement_error_s"] < 1e-6
        assert metrics["samples_seen_fraction"] == 1.
        assert len(agent.ppo.units) == len(agent.ppo.active) == len(agent.ppo.buffer.actions) == 0


def test_old_unified_checkpoint_is_explicitly_rejected() -> None:
    env = tiny_replay_env()
    env.reset()
    agent = UnifiedPPOAgent.from_env(env, hidden_dim=32)
    with pytest.raises(ValueError, match="fresh run"):
        agent.load_state_dict({"policy": {}})
