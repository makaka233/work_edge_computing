from dataclasses import replace
import json

import numpy as np
import pytest

from edge_drl.env.calibration import apply_service_budgets, builtin_budget_document, validate_budget_document
from edge_drl.env.scenario import generate_grouped_request
from scripts.calibrate_environment import paired_sampling_audit, resource_report
from tests.comparison_helpers import tiny_replay_env
from tests.test_latency200_scenario import candidate_config
from edge_drl.env.environment import EdgeComputingEnv
from tests.test_training_credit_repair import repaired_agent


def audited_env():
    env = EdgeComputingEnv(candidate_config(service_calibration_mode="audited-v1", network_constraint_mode="shared-nic"))
    env.reset()
    return env


def test_all_service_budgets_account_for_resident_and_context_memory():
    env = audited_env()
    document = builtin_budget_document()
    validate_budget_document(document)
    for service, budget in zip(env.scenario.services, document["services"]):
        assert service.input_mb_mean == budget["input_mb"]
        for stage, row in zip(service.stages, budget["stages"]):
            assert stage.memory_gb == pytest.approx(stage.resident_memory_gb + stage.reserved_contexts * stage.context_memory_gb)
            assert stage.compute_gcycles_mean == pytest.approx(.072 * row["compute_ms"])
            assert stage.storage_gb >= row["weights_gb"]
    assert resource_report(env)["status"] == "engineering_budget_audit_not_device_measurement"


def test_measured_budget_requires_provenance_and_matching_units():
    document = builtin_budget_document()
    document["provenance"]["status"] = "measured"
    with pytest.raises(ValueError, match="profiling provenance"):
        validate_budget_document(document)
    document["provenance"]["status"] = "estimated"
    document["units"]["memory"] = "GiB"
    with pytest.raises(ValueError, match="units"):
        validate_budget_document(document)


def test_imported_measurements_change_compute_without_changing_unit_normalization(tmp_path):
    env = audited_env()
    document = builtin_budget_document()
    document["services"][0]["stages"][1]["compute_ms"] = 400.0
    file = tmp_path / "profiling.json"
    file.write_text(json.dumps(document), encoding="utf-8")
    calibrated = apply_service_budgets(env.scenario.services, file)
    assert calibrated[0].stages[1].compute_gcycles_mean == pytest.approx(28.8)


def test_fixed_formats_do_not_randomly_change_shape_or_pcm_duration():
    env = audited_env()
    rng = np.random.default_rng(8)
    for service_id in (0, 3, 7):
        service = env.scenario.services[service_id]
        for i in range(20):
            request = generate_grouped_request(rng=rng, request_id=i, arrival_minute=0,
                    request_count=1, user_id=0, home_node=0, service_id=service_id, services=env.scenario.services)
            assert request.input_mb == pytest.approx(service.input_mb_mean)
            assert request.stage_output_mb[0] == pytest.approx(service.stages[0].output_mb_mean)


def test_mean_anchored_sampling_and_point_count_coupling():
    env = audited_env()
    service = env.scenario.services[6]
    rng = np.random.default_rng(11)
    inputs, computes = [], []
    for i in range(5000):
        request = generate_grouped_request(rng=rng, request_id=i, arrival_minute=0,
                request_count=1, user_id=0, home_node=0, service_id=6, services=env.scenario.services)
        inputs.append(request.input_mb)
        computes.append(request.stage_compute_gcycles[0])
        assert request.stage_output_mb[0] / request.input_mb == pytest.approx(service.stages[0].output_mb_mean / service.input_mb_mean)
    assert np.mean(inputs) == pytest.approx(service.input_mb_mean, rel=.02)
    assert np.mean(computes) == pytest.approx(service.stages[0].compute_gcycles_mean, rel=.02)


def test_uniform_temporal_strata_do_not_move_clock_or_leave_stratum():
    env = tiny_replay_env()
    env.config.temporal_sampling_mode = "stratified-jitter"
    env.reset()
    for start in (0, 100, 200):
        env.current_time_minute = start / 60
        env.sample_interval_requests(100)
        assert env.current_time_minute == start / 60
        assert all(start <= request.arrival_minute * 60 < start + 100 for request in env.current_requests)


def test_paired_full_sampling_is_exact_replay():
    env = tiny_replay_env()
    env.reset()
    agent = repaired_agent(env)
    report = paired_sampling_audit(env.config, env.scenario, agent,
                                    seconds=60, samples=[6, 60], repeats=1, seed=14)
    full = next(row for row in report["raw"] if row["samples"] == 60)
    assert full["relative_latency_error"] == 0.0
    assert full["node_load_rmse"] == 0.0
    assert full["request_count_relative_error"] == 0.0
