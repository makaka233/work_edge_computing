from copy import deepcopy
from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from edge_drl.comparison.monolithic_training_env import (
    MONOLITHIC_REQUEST_GENERATOR,
    MonolithicTrainingEnv,
)
from edge_drl.comparison.types import ExperimentPoint
from edge_drl.env.environment import EdgeComputingEnv
from tests.comparison_helpers import tiny_config, tiny_scenario
from train_monolithic import _optimized_monolithic_args, _synchronized_episode


def assert_same_request_stream(proposed, monolithic):
    assert len(proposed.current_requests) == len(monolithic.current_requests)
    for original, actual in zip(proposed.current_requests, monolithic.current_requests):
        assert actual == replace(original,
                                 stage_compute_gcycles=(sum(original.stage_compute_gcycles),),
                                 stage_output_mb=(0.0,))
    assert proposed.rng.bit_generator.state == monolithic.rng.bit_generator.state
    assert proposed._sampling_rng.bit_generator.state == monolithic._sampling_rng.bit_generator.state
    assert proposed.request_counter == monolithic.request_counter
    assert proposed.current_time_minute == monolithic.current_time_minute


@pytest.mark.parametrize("mode", ["stratified-jitter", "left-hold"])
@pytest.mark.parametrize("sampled_seconds", [0, 6, 40])
@pytest.mark.parametrize("arrival_profile", ["stationary", "daily"])
def test_on_demand_stream_matches_proposed_across_load_boundaries(
    monkeypatch, mode, sampled_seconds, arrival_profile
):
    monkeypatch.setattr("edge_drl.env.environment.generate_realistic_scenario",
                        lambda **kwargs: tiny_scenario())
    config = tiny_config(episode_minutes=20)
    config.mean_requests_per_minute = 120
    config.temporal_sampling_mode = mode
    config.arrival_profile = arrival_profile
    config.demand_load_multiplier = 0.8
    config.demand_load_group = 0
    proposed = EdgeComputingEnv(deepcopy(config))
    mono = MonolithicTrainingEnv(deepcopy(config), ExperimentPoint("request_load", 1.0, "test"))
    proposed.reset()
    mono.reset()
    assert_same_request_stream(proposed, mono)
    for name in ["adjacency", "bandwidth_mb_s", "propagation_ms"]:
        np.testing.assert_array_equal(getattr(proposed.scenario, name), getattr(mono.scenario, name))
    assert proposed.scenario.nodes == mono.scenario.nodes
    assert proposed.scenario.users == mono.scenario.users
    for source, target in zip(proposed.scenario.services, mono.scenario.services):
        assert len(target.stages) == 1
        assert target.stages[0].memory_gb == sum(x.memory_gb for x in source.stages)
        assert target.stages[0].compute_gcycles_mean == sum(x.compute_gcycles_mean for x in source.stages)
    proposed.deployment[:] = True
    mono.deployment[:] = True
    hold = 1 if sampled_seconds == 0 else 600 / sampled_seconds
    for multiplier, group in [(0.8, 0), (1.35, 3)]:
        previous = proposed.config.demand_load_multiplier
        proposed.config.demand_load_multiplier = multiplier
        proposed.config.demand_load_group = group
        if not np.isclose(previous, multiplier):
            proposed.current_requests = proposed._generate_current_second_requests()
            proposed.current_request = proposed.current_requests[0] if proposed.current_requests else None
        mono.set_window_load(multiplier, group)
        assert mono.config.demand_load_group == group
        assert mono._arrival_rate_per_minute() == proposed._arrival_rate_per_minute()
        assert_same_request_stream(proposed, mono)
        for _ in range(600 if sampled_seconds == 0 else sampled_seconds):
            proposed.sample_interval_requests(hold)
            mono.sample_interval_requests(hold)
            assert_same_request_stream(proposed, mono)
            for env in (proposed, mono):
                env.step([[request.home_node] * len(request.stage_compute_gcycles)
                                for request in env.current_requests], represented_seconds=hold)
            assert_same_request_stream(proposed, mono)
    if sampled_seconds == 6:
        # 1 reset + 11 step prefetches + 1 changed boundary + 12 jitter draws.
        assert mono.generated_request_batches == (25 if mode == "stratified-jitter" else 13)
    first = deepcopy(mono.current_requests)
    mono.reset()
    proposed.reset()
    assert_same_request_stream(proposed, mono)
    assert mono.generated_request_batches == 1
    assert first == []  # End of the preceding complete episode.


def test_load_group_changes_without_extra_rng_draw_when_multiplier_is_unchanged(monkeypatch):
    monkeypatch.setattr("edge_drl.env.environment.generate_realistic_scenario",
                        lambda **kwargs: tiny_scenario())
    env = MonolithicTrainingEnv(tiny_config(), ExperimentPoint("request_load", 1.0, "test"))
    env.reset()
    state = deepcopy(env.rng.bit_generator.state)
    env.set_window_load(1.0, 3)
    assert env.config.demand_load_group == 3
    assert env.rng.bit_generator.state == state


def test_episode_factory_is_lazy_and_records_sampler_version(monkeypatch):
    monkeypatch.setattr("edge_drl.env.environment.generate_realistic_scenario",
                        lambda **kwargs: tiny_scenario())
    args = dict(vars(tiny_config(60)))
    args["mean_requests_per_minute"] = 120
    env = _synchronized_episode(base_args=args, physical_seed=10,
                                environment_seed=30, demand_seed=20,
                                episode_minutes=60,
                                load_assignments=[(0.8, 0), (1., 1), (1.1, 2), (1.35, 3), (1., 1), (.8, 0)],
                                windows_per_episode=6,
                                point=ExperimentPoint("request_load", 1.0, "test"))
    assert env.generated_request_batches == 1
    assert env.config.demand_load_multiplier == .8
    assert env.config.demand_load_group == 0
    assert not hasattr(env, "_comparison_trace")
    optimized = _optimized_monolithic_args(args, reservation_microbatch_size=1,
                                          allow_full_resource_use=False)
    assert optimized["monolithic_request_generator"] == MONOLITHIC_REQUEST_GENERATOR
    assert "monolithic_request_generator" not in args


def test_mso_entrypoint_all_groups_checkpoint_and_resume(tmp_path, monkeypatch):
    from train_dual_ppo import parse_args

    monkeypatch.setattr(sys, "argv", ["test", "--pressure-profile", "mec-latency200",
                                     "--training-repair", "batch-credit-v1",
                                     "--training-design", "trajectory-simultaneous",
                                     "--num-edge-nodes", "8", "--num-service-types", "2",
                                     "--num-users", "10000", "--mean-requests-per-minute", "120",
                                     "--episode-minutes", "40",
                                     "--slow-allocation-mode", "budget-continuous-v2",
                                     "--fast-chain-context", "chain-kkt-v3",
                                     "--load-sampling-mode", "cyclic",
                                     "--sampled-seconds-per-window", "6",
                                     "--sampling-max-interval-seconds", "100",
                                     "--slow-critic-k-epochs", "1", "--slow-k-epochs", "1",
                                     "--slow-stage-k-epochs", "1", "--slow-placement-k-epochs", "1",
                                     "--fast-k-epochs", "2", "--fast-minibatch-size", "32"])
    args = vars(parse_args())
    base = tmp_path / "base.pt"
    torch.save({"metadata": {"args": args}}, base)
    common = [sys.executable, "train_mso.py", "--base-checkpoint", str(base),
              "--updates", "1", "--episodes-per-update", "1", "--device", "cpu",
              "--run-root", str(tmp_path), "--progress-interval-seconds", "0"]
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    root = Path(__file__).resolve().parents[1]
    fresh = subprocess.run(common + ["--run-name", "fresh"], cwd=root, env=environment,
                           capture_output=True, text=True, timeout=120)
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    with (tmp_path / "fresh/logs/training.csv").open(newline="") as handle:
        row = list(csv.DictReader(handle))[-1]
    assert float(row["fast_group_count"]) == 4
    assert float(row["fast_min_group_seen_fraction"]) == 1
    assert float(row["fast_epochs_completed"]) == 2
    assert float(row["fast_kl_early_stop"]) == 0
    manifest = json.loads((tmp_path / "fresh/training_manifest.json").read_text())
    assert manifest["synchronization"]["trace_pre_generated"] is False
    episode = manifest["updates"][0]["episodes"][0]
    assert episode["load_groups"] == [0, 1, 2, 3]
    assert episode["sampled_settlement_steps"] == 24
    assert episode["generated_request_batches"] < 60
    checkpoint = tmp_path / "fresh/checkpoints/latest.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["metadata"]["args"]["monolithic_request_generator"] == MONOLITHIC_REQUEST_GENERATOR
    assert saved["metadata"]["args"]["network_constraint_mode"] == "shared-nic"
    resumed = subprocess.run(common + ["--run-name", "resumed", "--load-checkpoint", str(checkpoint)],
                             cwd=root, env=environment, capture_output=True, text=True, timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    metadata = json.loads((tmp_path / "resumed/metadata.json").read_text())
    assert metadata["completed_updates"] == 2
    assert metadata["completed_rollouts"] == 8
    # Legacy checkpoints remain evaluable, but may not silently resume with the
    # repaired request process and combine two incompatible training histories.
    del saved["metadata"]["args"]["monolithic_request_generator"]
    old = tmp_path / "legacy.pt"
    torch.save(saved, old)
    rejected = subprocess.run(common + ["--run-name", "rejected", "--load-checkpoint", str(old)],
                              cwd=root, env=environment, capture_output=True, text=True, timeout=120)
    assert rejected.returncode != 0
    assert "monolithic_request_generator" in rejected.stderr
    assert not (tmp_path / "rejected").exists()
