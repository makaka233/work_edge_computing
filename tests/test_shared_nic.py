import numpy as np
import pytest
import json
from pathlib import Path

from edge_drl.allocators.kkt import LinkDemand, allocate_link_kkt
from edge_drl.allocators.nic_kkt import allocate_link_nic_kkt, nic_link_rates
from edge_drl.env.environment import _allocate_serial_link_kkt
from tests.comparison_helpers import tiny_replay_env
from tests.test_training_credit_repair import repaired_agent
from train_dual_ppo import rollout


def test_outgoing_star_splits_one_nic_by_sqrt_data():
    capacity = np.full((3, 3), 100.0)
    demands = [LinkDemand("a", 0, 1, 1.0), LinkDemand("b", 0, 2, 4.0)]
    allocations, delays, total = allocate_link_nic_kkt(demands, capacity, np.array([90., 100., 100.]))
    assert allocations["a"] == pytest.approx(30.0, rel=1e-6)
    assert allocations["b"] == pytest.approx(60.0, rel=1e-6)
    assert total == pytest.approx(9.0 / 90.0, rel=1e-6)
    assert sum(delays.values()) == pytest.approx(total)


def test_incoming_star_is_limited_separately_from_outgoing():
    capacity = np.full((3, 3), 100.0)
    demands = [LinkDemand("a", 0, 2, 1.0), LinkDemand("b", 1, 2, 1.0), LinkDemand("c", 2, 0, 1.0)]
    allocations, _, _ = allocate_link_nic_kkt(demands, capacity, np.array([100., 100., 80.]))
    assert allocations["a"] == pytest.approx(40.0)
    assert allocations["b"] == pytest.approx(40.0)
    assert allocations["c"] == pytest.approx(80.0)


def test_unlimited_nics_equal_legacy_allocator():
    capacity = np.full((3, 3), 100.0)
    demands = [LinkDemand("a", 0, 1, .2, 3), LinkDemand("b", 0, 1, 1.0, 2)]
    reference = allocate_link_kkt(demands, capacity)
    actual = allocate_link_nic_kkt(demands, capacity, np.full(3, np.inf))
    assert actual == reference


def test_coupled_network_satisfies_ports_edges_and_kkt_gap():
    rng = np.random.default_rng(4)
    loads = rng.uniform(.1, 3, (6, 6))
    np.fill_diagonal(loads, 0)
    capacity = rng.uniform(10, 70, (6, 6))
    nic = rng.uniform(30, 100, 6)
    rates, cost, (u, v) = nic_link_rates(loads, capacity, nic)
    assert np.all(rates.sum(axis=0) <= nic + 1e-7)
    assert np.all(rates.sum(axis=1) <= nic + 1e-7)
    assert np.all(rates <= capacity)
    active = loads > 0
    free = active & (rates < capacity - 1e-5)
    np.testing.assert_allclose(loads[free] ** 2 / rates[free] ** 2, (u[:, None] + v)[free], rtol=1e-5)
    complementarity = np.sum(u * (nic - rates.sum(axis=1))) + np.sum(v * (nic - rates.sum(axis=0)))
    assert abs(complementarity) < cost * 1e-6


@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("fixture", ["shared_nic_training_failure.json", "shared_nic_joint_stagnation.json"])
def test_real_32_node_training_failure_is_certified(warm_start, fixture):
    """Both real seed-2026 failures, before/after joint fallback, stay fixed."""
    path = Path(__file__).parent / "fixtures" / fixture
    data = {key: np.asarray(value) for key, value in json.loads(path.read_text()).items()}
    initial = (data["outgoing"], data["incoming"]) if warm_start else None
    rates, cost, (u, v) = nic_link_rates(data["loads"], data["capacities"], data["nic"],
                                       initial_duals=initial)
    assert np.all(rates.sum(axis=0) <= data["nic"] + 1e-9)
    assert np.all(rates.sum(axis=1) <= data["nic"] + 1e-9)
    assert np.all(rates <= data["capacities"])
    active = data["loads"] > 0
    raw_rates = np.minimum(np.where(active, data["capacities"], 0),
                           data["loads"] / np.sqrt(np.maximum(u[:, None] + v, 1e-30)))
    dual_objective = np.sum(np.divide(data["loads"] ** 2, raw_rates,
                                     out=np.zeros_like(rates), where=active))
    dual_objective += np.dot(u, raw_rates.sum(axis=1) - data["nic"])
    dual_objective += np.dot(v, raw_rates.sum(axis=0) - data["nic"])
    assert -1e-10 <= (cost - dual_objective) / cost <= 1e-7


def test_joint_dual_fallback_matches_coordinate_solution():
    rng = np.random.default_rng(87)
    for _ in range(8):
        loads = rng.uniform(.01, 4, (8, 8))
        np.fill_diagonal(loads, 0)
        capacities = rng.uniform(20, 80, (8, 8))
        nic = rng.uniform(40, 200, 8)
        _, reference, _ = nic_link_rates(loads, capacities, nic)
        rates, cost, _ = nic_link_rates(loads, capacities, nic, max_iterations=1)
        assert cost == pytest.approx(reference, rel=2e-7)
        assert np.all(rates.sum(axis=0) <= nic + 1e-9)
        assert np.all(rates.sum(axis=1) <= nic + 1e-9)


def test_optimizer_success_is_polished_until_independently_certified(monkeypatch):
    import scipy.optimize
    original = scipy.optimize.minimize
    calls = []

    def inaccurate_success(*args, **kwargs):
        result = original(*args, **kwargs)
        result.x *= .99999  # A plausible tiny residual despite optimizer success.
        result.success = True
        result.message = "test: objective stagnated before physical certificate"
        calls.append(result)
        return result

    path = Path(__file__).parent / "fixtures" / "shared_nic_training_failure.json"
    data = {key: np.asarray(value) for key, value in json.loads(path.read_text()).items()}
    _, reference, _ = nic_link_rates(data["loads"], data["capacities"], data["nic"])
    monkeypatch.setattr(scipy.optimize, "minimize", inaccurate_success)
    rates, cost, (u, v) = nic_link_rates(data["loads"], data["capacities"], data["nic"], max_iterations=1)
    assert calls
    assert cost == pytest.approx(reference, rel=2e-7)
    assert np.all(rates.sum(axis=0) <= data["nic"] + 1e-9)
    assert np.all(rates.sum(axis=1) <= data["nic"] + 1e-9)
    active = data["loads"] > 0
    raw = np.minimum(np.where(active, data["capacities"], 0),
                     data["loads"] / np.sqrt(np.maximum(u[:, None] + v, 1e-30)))
    dual = np.sum(np.divide(data["loads"] ** 2, raw, out=np.zeros_like(raw), where=active))
    dual += np.dot(u, raw.sum(axis=1) - data["nic"])
    dual += np.dot(v, raw.sum(axis=0) - data["nic"])
    assert -1e-10 <= (cost - dual) / cost <= 1e-7


def test_32_node_sparse_mixed_payload_stress():
    rng = np.random.default_rng(721)
    for _ in range(128):
        loads = np.exp(rng.uniform(-5, 2, (32, 32)))
        loads *= rng.random((32, 32)) < rng.uniform(.04, .35)
        np.fill_diagonal(loads, 0)
        capacities = rng.uniform(60, 600, (32, 32))
        nic = rng.choice([100., 1000.], size=32)
        rates, cost, _ = nic_link_rates(loads, capacities, nic)
        assert np.isfinite(cost)
        assert np.all(rates.sum(axis=0) <= nic + 1e-8)
        assert np.all(rates.sum(axis=1) <= nic + 1e-8)


def test_sequential_prefix_warm_starts_preserve_cost_and_constraints():
    rng = np.random.default_rng(191)
    capacities = rng.uniform(60, 500, (32, 32))
    nic = rng.choice([100., 1000.], size=32)
    for _ in range(8):
        loads = np.zeros((32, 32))
        duals = None
        previous = 0.0
        for _ in range(64):
            left = int(rng.integers(32))
            right = int((left + rng.integers(1, 32)) % 32)
            loads[left, right] += np.sqrt(float(rng.choice([.004, .1, .3, 1., 8.])))
            rates, cost, duals = nic_link_rates(loads, capacities, nic, initial_duals=duals)
            assert cost >= previous - 1e-7 * max(cost, previous, 1e-12)
            assert np.all(rates.sum(axis=0) <= nic + 1e-8)
            assert np.all(rates.sum(axis=1) <= nic + 1e-8)
            previous = cost


@pytest.mark.parametrize("warm", [False, True])
def test_second_order_fallback_certifies_wide_scale_failure(warm, monkeypatch):
    data = np.load(Path(__file__).parent / "fixtures/shared_nic_wide_scale_stress.npz")
    calls = []
    original = np.linalg.lstsq
    def track(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(np.linalg, "lstsq", track)
    initial = (data["outgoing"], data["incoming"]) if warm else None
    rates, cost, (u, v) = nic_link_rates(data["loads"], data["capacities"], data["nic"], initial_duals=initial)
    assert calls  # must exercise the actual joint second-order fallback
    assert np.all(rates.sum(axis=0) <= data["nic"] + 1e-9)
    assert np.all(rates.sum(axis=1) <= data["nic"] + 1e-9)
    assert np.all(rates <= data["capacities"])
    active = data["loads"] > 0
    raw = np.minimum(np.where(active, data["capacities"], 0),
                     data["loads"] / np.sqrt(np.maximum(u[:, None] + v, 1e-30)))
    dual = np.sum(np.divide(data["loads"]**2, raw, out=np.zeros_like(raw), where=active))
    dual += np.dot(u, raw.sum(axis=1) - data["nic"]) + np.dot(v, raw.sum(axis=0) - data["nic"])
    assert -1e-10 <= (cost - dual) / cost <= 1e-7


def test_wide_dynamic_range_ports_have_same_precision_certificate():
    rng = np.random.default_rng(518)
    for _ in range(64):
        loads = np.exp(rng.uniform(-14, 4, (12, 12)))
        loads *= rng.random((12, 12)) < .3
        np.fill_diagonal(loads, 0)
        capacities = rng.choice([50., 125., 250., 500.], (12, 12))
        nic = rng.choice([125., 250., 1250.], 12)
        rates, cost, (u, v) = nic_link_rates(loads, capacities, nic)
        active = loads > 0
        raw = np.minimum(np.where(active, capacities, 0), loads / np.sqrt(np.maximum(u[:, None] + v, 1e-30)))
        dual = np.sum(np.divide(loads**2, raw, out=np.zeros_like(raw), where=active))
        dual += np.dot(u, raw.sum(axis=1) - nic) + np.dot(v, raw.sum(axis=0) - nic)
        assert -1e-10 <= (cost - dual) / cost <= 1e-7
        assert np.all(rates.sum(axis=0) <= nic + 1e-9)
        assert np.all(rates.sum(axis=1) <= nic + 1e-9)


@pytest.mark.parametrize("warm", [False, True])
def test_exact_user_update50_failure_keeps_original_tolerance(warm):
    # Old solver reproduces ratio=1.0000386 and gap=2.15e-6 on this matrix.
    data = np.load(Path(__file__).parent / "fixtures/shared_nic_update50_failure.npz")
    initial = (data["outgoing"], data["incoming"]) if warm else None
    rates, cost, (u, v) = nic_link_rates(data["loads"], data["capacities"], data["nic"], initial_duals=initial)
    active = data["loads"] > 0
    raw = np.minimum(np.where(active, data["capacities"], 0),
                     data["loads"] / np.sqrt(np.maximum(u[:, None] + v, 1e-30)))
    dual = np.sum(np.divide(data["loads"]**2, raw, out=np.zeros_like(raw), where=active))
    dual += np.dot(u, raw.sum(axis=1) - data["nic"]) + np.dot(v, raw.sum(axis=0) - data["nic"])
    assert -1e-10 <= (cost - dual) / cost <= 1e-7
    assert np.all(rates.sum(axis=0) <= data["nic"] + 1e-9)
    assert np.all(rates.sum(axis=1) <= data["nic"] + 1e-9)
    assert np.all(rates <= data["capacities"])


@pytest.mark.parametrize("mode", ["kkt", "equal-share", "learned-share"])
def test_shared_nic_supports_resource_comparison_modes(mode):
    demands = [LinkDemand("a", 0, 1, 1.0), LinkDemand("b", 0, 2, 1.0)]
    delays, _ = _allocate_serial_link_kkt(demands, np.full((3, 3), 100.0), np.full(3, 100.0), mode)
    assert sum(delays.values()) == pytest.approx(.04)


def test_shared_nic_environment_and_fast_prefix_rewards_agree():
    env = tiny_replay_env()
    env.config.network_constraint_mode = "shared-nic"
    env.reset()
    # Give the synthetic helper a finite NIC, since legacy nodes default to infinity.
    from dataclasses import replace
    env.scenario.nodes = [replace(n, network_uplink_mb_s=20.) for n in env.scenario.nodes]
    agent = repaired_agent(env)
    stats = rollout(env, agent, max_requests=12, reset_env=False)
    assert stats["max_fast_kkt_reward_residual_s"] < 1e-6
