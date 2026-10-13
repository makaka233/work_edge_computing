from dataclasses import replace

import numpy as np
import pytest
import torch

from edge_drl.agents.rollout_features import edge_features, route_plan
from tests.comparison_helpers import tiny_replay_env
from tests.test_training_credit_repair import repaired_agent
from train_dual_ppo import rollout


def scalar_edges(env, include_load):
    bw, prop = env.scenario.bandwidth_mb_s, env.scenario.propagation_ms
    max_bw = np.nanmax(np.where(np.isfinite(bw), bw, 0))
    max_prop = max(float(np.where(np.isfinite(prop), prop, 0).max()), 1e-9)
    result = []
    for left in range(len(bw)):
        for right in range(len(bw)):
            result.extend([float(env.scenario.adjacency[left, right]),
                           float((bw[left, right] if np.isfinite(bw[left, right]) else max_bw) / max(max_bw, 1e-9)),
                           float((prop[left, right] if np.isfinite(prop[left, right]) else max_prop) / max_prop)])
            if include_load:
                result.append(float(np.clip(env.link_load[left, right] if np.isfinite(env.link_load[left, right]) else 0, 0, 1)))
    return result


@pytest.mark.parametrize("include_load", [False, True])
def test_graph_features_match_reference_and_follow_load_changes(include_load):
    env = tiny_replay_env()
    env.reset()
    np.testing.assert_array_equal(edge_features(env, include_load=include_load), scalar_edges(env, include_load))
    env.link_load[0, 1] = .8
    np.testing.assert_array_equal(edge_features(env, include_load=include_load), scalar_edges(env, include_load))


def test_graph_and_path_cache_invalidates_on_inplace_bandwidth_change():
    env = tiny_replay_env()
    env.reset()
    old = route_plan(env)
    env.scenario.bandwidth_mb_s[0, 1] = env.scenario.bandwidth_mb_s[1, 0] = .01
    new = route_plan(env)
    assert old is not new
    assert len(env.shortest_path(0, 1)) > 2
    np.testing.assert_array_equal(edge_features(env), scalar_edges(env, False))
    # Reset replaces the scenario object and must also invalidate old snapshots.
    env.reset()
    assert route_plan(env) is not new


@pytest.mark.parametrize("network_mode", ["edge-only", "shared-nic"])
def test_vector_candidate_features_match_scalar_for_all_nodes(network_mode):
    env = tiny_replay_env()
    env.reset()
    env.config.network_constraint_mode = network_mode
    env.scenario.nodes = [replace(n, network_uplink_mb_s=20.) for n in env.scenario.nodes]
    agent = repaired_agent(env)
    env.apply_deployment(agent.slow_agent.plan_deployment(env, deterministic=True, record=False))
    fast = agent.fast_agent
    request = next(r for slot in env._comparison_trace.slots for r in slot if len(r.stage_compute_gcycles) == 3)
    rng = np.random.default_rng(12)
    compute = rng.uniform(0, 5, (3, 6))
    links = rng.uniform(0, 2, (3, 6, 6))
    env.link_load[:] = rng.uniform(0, .6, (6, 6))
    plan = route_plan(env)
    for stage in range(3):
        partial = [0] * stage
        vector = fast._kkt_candidate_features_all(env, request, stage, partial, compute, links, plan)
        scalar = np.array([fast._kkt_candidate_features(env, request, stage, node, partial, compute, links)
                           for node in range(6)])
        np.testing.assert_allclose(vector, scalar, rtol=1e-12, atol=1e-12)
        env.current_requests = [request]
        fast.vectorized_features = True
        state = fast._build_state(env, request, stage, partial, kkt_compute_sqrt=compute, kkt_link_sqrt=links)
        fast.vectorized_features = False
        reference = fast._build_state(env, request, stage, partial, kkt_compute_sqrt=compute, kkt_link_sqrt=links)
        np.testing.assert_array_equal(state, reference)


def test_acceleration_preserves_actions_logprobs_values_and_exact_credit():
    torch.manual_seed(24)
    env = tiny_replay_env()
    env.reset()
    env.config.network_constraint_mode = "shared-nic"
    env.scenario.nodes = [replace(n, network_uplink_mb_s=20.) for n in env.scenario.nodes]
    agent = repaired_agent(env)
    env.apply_deployment(agent.slow_agent.plan_deployment(env, deterministic=True, record=False))
    requests = list(next(slot for slot in env._comparison_trace.slots if len(slot) > 1))
    env.current_requests, env.current_request = requests, requests[0]
    fast = agent.fast_agent
    fast.vectorized_features = False
    torch.manual_seed(33)
    scalar_paths = fast.schedule_batch(env, requests, record=True)
    old_states = np.stack(fast.ppo.buffer.states)
    old_logprobs = fast.ppo.buffer.logprobs.copy()
    old_values = fast.ppo.buffer.values.copy()
    old_costs = [x.copy() for x in fast.pending_kkt_stage_costs]
    fast.ppo.buffer.clear()
    fast.pending_kkt_stage_costs.clear()
    fast.vectorized_features = True
    torch.manual_seed(33)
    paths = fast.schedule_batch(env, requests, record=True)
    assert paths == scalar_paths
    np.testing.assert_array_equal(fast.ppo.buffer.states, old_states)
    np.testing.assert_array_equal(fast.ppo.buffer.logprobs, old_logprobs)
    np.testing.assert_array_equal(fast.ppo.buffer.values, old_values)
    for actual, reference in zip(fast.pending_kkt_stage_costs, old_costs):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)


def test_repaired_rollout_skips_unused_legacy_credit_probes(monkeypatch):
    env = tiny_replay_env()
    env.reset()
    agent = repaired_agent(env)
    import sys
    from train_dual_ppo import parse_args
    def unexpected(*args, **kwargs):
        raise AssertionError("legacy local credit probe should not run in repaired team mode")
    monkeypatch.setattr(agent, "observe_replica_ablation", unexpected)
    monkeypatch.setattr(agent.fast_agent, "stage_selected_node_marginal_values", unexpected)
    monkeypatch.setattr(agent.fast_agent, "stage_counterfactual_regrets", unexpected)
    monkeypatch.setattr(sys, "argv", ["test", "--slow-allocation-mode", "budget-continuous-v2",
                                     "--slow-counterfactual-credit-coef", "1", "--slow-replica-ablation-samples", "2"])
    args = parse_args()
    stats = rollout(env, agent, max_requests=10, args=args)
    assert stats["requests"] > 0
