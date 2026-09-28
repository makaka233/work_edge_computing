from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import torch

from edge_drl.agents.drl import FastSchedulingPPOAgent, HierarchicalPPOAgent, fast_structured_obs_dim
from edge_drl.comparison.checkpoint import build_proposed_agent
from edge_drl.models.ppo import StructuredPathDistribution
from tests.comparison_helpers import tiny_replay_env


def _env_and_request():
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
        request for slot in env._comparison_trace.slots for request in slot if request.service_id == 0
    )
    return env, request


def test_dynamic_programming_matches_enumerated_path_distribution() -> None:
    unary = torch.tensor(
        [[[0.2, -0.1, 0.7], [0.3, 0.5, -0.3], [-0.2, 0.1, 0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    pair = torch.tensor(
        [[[[0.0, 0.1, -0.2], [0.3, 0.2, 0.0], [0.1, -0.1, 0.4]],
          [[0.2, 0.1, 0.3], [0.0, -0.2, 0.4], [0.2, 0.3, -0.1]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    node_mask = torch.tensor([[[True, True, False], [True, True, True], [False, True, True]]])
    pair_mask = torch.ones((1, 2, 3, 3), dtype=torch.bool)
    pair_mask[0, 0, 1, 0] = False
    dist = StructuredPathDistribution(unary, pair, node_mask, pair_mask)
    valid = []
    scores = []
    for path in product(range(3), repeat=3):
        if not all(node_mask[0, stage, node] for stage, node in enumerate(path)):
            continue
        if not all(pair_mask[0, stage, path[stage], path[stage + 1]] for stage in range(2)):
            continue
        score = sum(unary[0, stage, node] for stage, node in enumerate(path))
        score += sum(pair[0, stage, path[stage], path[stage + 1]] for stage in range(2))
        valid.append(path)
        scores.append(score)
    score_t = torch.stack(scores)
    expected_logp = torch.log_softmax(score_t, dim=0)
    actions = torch.tensor([sum(node * 3 ** stage for stage, node in enumerate(path)) for path in valid])
    actual_logp = torch.cat([dist.log_prob(action[None]) for action in actions])
    torch.testing.assert_close(actual_logp, expected_logp)
    torch.testing.assert_close(dist.entropy(), -(expected_logp.exp() * expected_logp).sum()[None])
    torch.testing.assert_close(dist.maximum_entropy(), torch.tensor([np.log(len(valid))], dtype=torch.float64))
    assert dist.mode().item() == actions[int(score_t.argmax())].item()
    (-dist.log_prob(actions[:1]) - 0.01 * dist.entropy()).sum().backward()
    assert torch.isfinite(unary.grad).all()
    assert torch.isfinite(pair.grad).all()


def test_structured_policy_scores_all_deployed_paths_without_top_k() -> None:
    env, request = _env_and_request()
    agent = FastSchedulingPPOAgent(
        num_nodes=6,
        max_service_stages=3,
        policy_kind="structured_path",
        path_candidate_limit=1,  # Must not constrain structured actions.
    )
    state, mask = agent._structured_path_state(env, request)
    assert state.shape == (fast_structured_obs_dim(6, 3),)
    assert np.isfinite(state).all()
    assert mask.reshape(3, 6).sum(axis=1).tolist() == [2, 2, 1]
    state_t = torch.as_tensor(state[None], dtype=torch.float32)
    mask_t = torch.as_tensor(mask[None])
    dist, _ = agent.ppo.policy(state_t, mask_t)
    paths = [(0, 2), (0, 3), (1, 2), (1, 3)]
    logprobs = torch.stack(
        [dist.log_prob(torch.tensor([first + second * 6]))[0] for first, second in paths]
    )
    assert torch.isfinite(logprobs).all()
    torch.testing.assert_close(torch.exp(logprobs).sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    assert agent._decode_structured_path(int(dist.mode().item()), 2) in [list(path) for path in paths]


def test_structured_path_rollout_and_ppo_update() -> None:
    env, request = _env_and_request()
    agent = FastSchedulingPPOAgent(
        num_nodes=6,
        max_service_stages=3,
        policy_kind="structured_path",
        path_candidate_limit=1,
        reservation_microbatch_size=1,
        k_epochs=1,
    )
    paths = agent.schedule_batch(env, [request, request], deterministic=True, record=True)
    assert len(paths) == len(agent.ppo.buffer) == 2
    assert all(env.evaluate_schedule(request, path)["valid"] for path in paths)
    assert not np.array_equal(agent.ppo.buffer.states[0], agent.ppo.buffer.states[1])
    for path in paths:
        regrets = agent.stage_counterfactual_regrets(env, request, path)
        assert len(regrets) == len(path)
        assert regrets[0] >= 0.0
        agent.assign_schedule_rewards([-0.2, -0.3], done=False)
    assert agent.ppo.buffer.rewards == [-0.5, -0.5]
    metrics = agent.update()
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["normalized_entropy"])


def test_structured_compute_proxy_matches_isolated_kkt_delay() -> None:
    env, request = _env_and_request()
    env.current_requests = [request]
    agent = FastSchedulingPPOAgent(num_nodes=6, max_service_stages=3, policy_kind="structured_path")
    state, _ = agent._structured_path_state(env, request)
    node_features = state[9 : 9 + 3 * 6 * 9].reshape(3, 6, 9)
    node_id = 0
    predicted = float(np.expm1(node_features[0, node_id, 6]))
    capacity = env.scenario.nodes[node_id].compute_gcycles_per_s * max(
        1.0 - 0.75 * float(env.node_compute_load[node_id]), 0.10
    )
    expected = float(request.stage_compute_gcycles[0]) / capacity
    assert np.isclose(predicted, expected, rtol=1e-5)


def test_structured_compute_proxy_uses_same_phase_sqrt_competition() -> None:
    env, request = _env_and_request()
    placement = env.deployment.copy()
    placement[0, 0, 1] = False
    env.apply_deployment(placement)
    other_compute = 4.0 * float(request.stage_compute_gcycles[0])
    other = replace(
        request,
        stage_compute_gcycles=(other_compute, *request.stage_compute_gcycles[1:]),
    )
    env.current_requests = [request, other]
    agent = FastSchedulingPPOAgent(num_nodes=6, max_service_stages=3, policy_kind="structured_path")
    state, _ = agent._structured_path_state(env, request)
    node_features = state[9 : 9 + 3 * 6 * 9].reshape(3, 6, 9)
    predicted = float(np.expm1(node_features[0, 0, 6]))
    own_compute = float(request.stage_compute_gcycles[0])
    capacity = env.scenario.nodes[0].compute_gcycles_per_s * max(
        1.0 - 0.75 * float(env.node_compute_load[0]), 0.10
    )
    expected = np.sqrt(own_compute) * (np.sqrt(own_compute) + np.sqrt(other_compute)) / capacity
    assert np.isclose(predicted, expected, rtol=1e-5)


def test_structured_link_proxy_matches_joint_serial_kkt() -> None:
    env, original = _env_and_request()
    placement = env.deployment.copy()
    placement[0, 0] = False
    placement[0, 0, 1] = True
    placement[0, 1] = False
    placement[0, 1, 2] = True
    env.apply_deployment(placement)
    request = replace(original, home_node=0, request_count=2)
    other = replace(
        request,
        request_id=request.request_id + 10_000,
        request_count=3,
        input_mb=4.0 * request.input_mb,
        stage_output_mb=tuple(4.0 * value for value in request.stage_output_mb),
    )
    env.current_requests = [request, other]
    agent = FastSchedulingPPOAgent(num_nodes=6, max_service_stages=3, policy_kind="structured_path")
    state, _ = agent._structured_path_state(env, request)
    node_features = state[9 : 9 + 3 * 6 * 9].reshape(3, 6, 9)
    edge_features = state[9 + 3 * 6 * 9 :].reshape(2, 6, 6, 5)
    actual = env.evaluate_batch_schedules([request, other], [[1, 2], [1, 2]])[0]
    expected_ingress = actual["link_delays"]["ingress"] + actual["logical_propagation_delays"]["ingress"]
    expected_interstage = actual["link_delays"]["stage-0"] + actual["logical_propagation_delays"]["stage-0"]
    assert np.isclose(np.expm1(node_features[0, 1, 8]), expected_ingress, rtol=1e-5)
    assert np.isclose(np.expm1(edge_features[0, 1, 2, 1]), expected_interstage, rtol=1e-5)
    assert edge_features[0, 1, 1, 4] == 1.0
    assert np.expm1(edge_features[0, 1, 1, 1]) == 0.0


def test_structured_link_reservation_changes_next_microbatch_cost() -> None:
    env, original = _env_and_request()
    request = replace(original, home_node=0)
    env.current_requests = [request]
    agent = FastSchedulingPPOAgent(num_nodes=6, max_service_stages=3, policy_kind="structured_path")
    baseline = agent._structured_link_transfer_costs(
        env, request, phase=0, data_mb=request.input_mb, reservation_link_sqrt_delta=None
    )
    _, _, incidence, _ = agent._structured_route_metrics(env)
    delta = np.zeros((5, incidence.shape[-1]), dtype=np.float64)
    delta[0] += float(request.request_count) * np.sqrt(request.input_mb) * incidence[0, 1]
    reserved = agent._structured_link_transfer_costs(
        env, request, phase=0, data_mb=request.input_mb, reservation_link_sqrt_delta=delta
    )
    assert reserved[0, 1] > baseline[0, 1]
    assert reserved[0, 0] == baseline[0, 0] == 0.0


def test_structured_link_proxy_does_not_compete_across_serial_phases() -> None:
    env, original = _env_and_request()
    placement = np.zeros_like(env.deployment)
    placement[0, 0, 1] = True
    placement[0, 1, 1] = True
    placement[1, 0, 0] = True
    placement[1, 1, 1] = True
    placement[1, 2, 1] = True
    env.apply_deployment(placement)
    target = replace(original, home_node=0)
    other = next(
        request for slot in env._comparison_trace.slots for request in slot if request.service_id == 1
    )
    other = replace(other, home_node=0)
    env.current_requests = [target, other]
    agent = FastSchedulingPPOAgent(num_nodes=6, max_service_stages=3, policy_kind="structured_path")
    state, _ = agent._structured_path_state(env, target)
    node_features = state[9 : 9 + 3 * 6 * 9].reshape(3, 6, 9)
    actual = env.evaluate_batch_schedules([target, other], [[1, 1], [0, 1, 1]])[0]
    predicted = float(np.expm1(node_features[0, 1, 8]))
    expected = actual["link_delays"]["ingress"] + actual["logical_propagation_delays"]["ingress"]
    assert np.isclose(predicted, expected, rtol=1e-5)


def test_comparison_loader_reconstructs_structured_policy(tmp_path) -> None:
    env, request = _env_and_request()
    trained = HierarchicalPPOAgent.from_env(env, fast_policy_kind="structured_path")
    checkpoint = tmp_path / "structured_fast.pt"
    torch.save({"fast_agent": trained.fast_agent.ppo.policy.state_dict()}, checkpoint)
    loaded, _ = build_proposed_agent(
        env, checkpoint, {"fast_policy_kind": "structured_path"}, device="cpu"
    )
    assert loaded.fast_agent.policy_kind == "structured_path"
    assert len(loaded.fast_agent.schedule(env, request, deterministic=True, record=False)) == 2
