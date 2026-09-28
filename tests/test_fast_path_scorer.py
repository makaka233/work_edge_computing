from __future__ import annotations

import numpy as np
import pytest
import torch

from edge_drl.agents.drl import FastSchedulingPPOAgent, HierarchicalPPOAgent, fast_path_obs_dim
from edge_drl.comparison.checkpoint import build_proposed_agent
from edge_drl.models.ppo import NodeScoringActorCritic
from tests.comparison_helpers import tiny_replay_env
from train_dual_ppo import load_checkpoint


def _path_env_and_request():
    env = tiny_replay_env()
    env.reset()
    placement = np.zeros_like(env.deployment)
    for service in env.scenario.services:
        placement[service.service_id, : len(service.stages), 0] = True
    placement[0, 0] = False
    placement[0, 0, [0, 1]] = True
    placement[0, 1] = False
    placement[0, 1, [2, 3]] = True
    env.apply_deployment(placement)
    request = next(
        request
        for slot in env._comparison_trace.slots
        for request in slot
        if request.service_id == 0
    )
    return env, request


def test_path_candidates_obey_deployment_and_mask_padding() -> None:
    env, request = _path_env_and_request()
    agent = FastSchedulingPPOAgent(
        num_nodes=6,
        max_service_stages=3,
        policy_kind="path_scorer",
        path_candidate_limit=8,
    )

    paths, state, mask = agent._path_candidates(env, request)

    assert set(paths) == {(0, 2), (0, 3), (1, 2), (1, 3)}
    assert mask.tolist() == [True] * 4 + [False] * 4
    assert state.shape == (fast_path_obs_dim(3, 8),)
    assert np.isfinite(state).all()
    assert all(env.evaluate_schedule(request, list(path))["valid"] for path in paths)


def test_bounded_candidates_preserve_deployed_ingress_diversity() -> None:
    env, request = _path_env_and_request()
    agent = FastSchedulingPPOAgent(
        num_nodes=6,
        max_service_stages=3,
        policy_kind="path_scorer",
        path_candidate_limit=2,
    )
    paths, _, mask = agent._path_candidates(env, request)
    assert {path[0] for path in paths} == {0, 1}
    assert mask.tolist() == [True, True]


def test_path_action_and_reward_are_one_transition_per_request() -> None:
    env, request = _path_env_and_request()
    agent = FastSchedulingPPOAgent(
        num_nodes=6,
        max_service_stages=3,
        policy_kind="path_scorer",
        path_candidate_limit=2,
        reservation_microbatch_size=1,
    )

    schedules = agent.schedule_batch(env, [request, request], deterministic=True, record=True)
    assert len(schedules) == 2
    assert len(agent.ppo.buffer) == 2
    assert all(env.evaluate_schedule(request, schedule)["valid"] for schedule in schedules)
    assert all(mask.sum() == 2 for mask in agent.ppo.buffer.masks)
    for schedule in schedules:
        regrets = agent.stage_counterfactual_regrets(env, request, schedule)
        assert len(regrets) == len(schedule)
        assert regrets[0] >= 0.0
        assert all(regret == 0.0 for regret in regrets[1:])
    agent.assign_schedule_rewards([-0.2, -0.3], done=False, weight=2.0)
    agent.assign_schedule_rewards([-0.4, -0.5], done=False, weight=2.0)
    assert np.allclose(agent.ppo.buffer.rewards, [-0.5, -0.9])
    assert agent.ppo.buffer.dones == [True, True]
    assert agent.ppo.buffer.weights == [2.0, 2.0]
    metrics = agent.update()
    assert np.isfinite(metrics["loss"])


def test_path_candidate_set_changes_with_deployment() -> None:
    env, request = _path_env_and_request()
    agent = FastSchedulingPPOAgent(
        num_nodes=6,
        max_service_stages=3,
        policy_kind="path_scorer",
        path_candidate_limit=8,
    )
    first, _, _ = agent._path_candidates(env, request)
    placement = env.deployment.copy()
    placement[0, 1, 3] = False
    env.apply_deployment(placement)
    second, _, mask = agent._path_candidates(env, request)
    assert len(first) == 4
    assert set(second) == {(0, 2), (1, 2)}
    assert int(mask.sum()) == 2


def test_path_cost_guidance_affects_logits_without_changing_old_model() -> None:
    old = NodeScoringActorCritic(global_dim=1, node_feature_dim=4, num_nodes=2)
    guided = NodeScoringActorCritic(
        global_dim=1,
        node_feature_dim=4,
        num_nodes=2,
        candidate_cost_feature_index=3,
    )
    assert "cost_guidance_log_scale" not in old.state_dict()
    for parameter in guided.scorer.parameters():
        torch.nn.init.zeros_(parameter)
    states = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
    distribution, _ = guided(states, torch.tensor([[True, True]]))
    assert distribution.probs[0, 0] > distribution.probs[0, 1]
    masked_distribution, _ = guided(states, torch.tensor([[False, True]]))
    assert masked_distribution.probs[0, 0] == 0.0


def test_stage_checkpoint_cannot_be_resumed_as_path_policy(tmp_path) -> None:
    env, _ = _path_env_and_request()
    old = HierarchicalPPOAgent.from_env(env, fast_policy_kind="gat_node_scorer")
    path = HierarchicalPPOAgent.from_env(
        env, fast_policy_kind="path_scorer", fast_path_candidate_limit=8
    )
    checkpoint = tmp_path / "stage_fast.pt"
    torch.save({"fast_agent": old.fast_agent.ppo.policy.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="checkpoint Fast policy architecture differs"):
        load_checkpoint(path, checkpoint)


def test_comparison_loader_reconstructs_path_policy(tmp_path) -> None:
    env, request = _path_env_and_request()
    trained = HierarchicalPPOAgent.from_env(
        env, fast_policy_kind="path_scorer", fast_path_candidate_limit=8
    )
    checkpoint = tmp_path / "path_fast.pt"
    torch.save({"fast_agent": trained.fast_agent.ppo.policy.state_dict()}, checkpoint)
    loaded, _ = build_proposed_agent(
        env,
        checkpoint,
        {"fast_policy_kind": "path_scorer", "fast_path_candidate_limit": 8},
        device="cpu",
    )
    assert loaded.fast_agent.policy_kind == "path_scorer"
    assert len(loaded.fast_agent.schedule(env, request, deterministic=True, record=False)) == 2
