"""A resume must not silently remove shared NIC/workload/sampling settings."""
import json
from pathlib import Path

import pytest

from train_dual_ppo import checkpoint_run_args


@pytest.mark.parametrize("nested", [False, True])
def test_old_checkpoint_recovers_real_run_sidecar(tmp_path, nested):
    run = tmp_path / "run"
    directory = run / "checkpoints" / "snapshots" if nested else run / "checkpoints"
    directory.mkdir(parents=True)
    args = {"network_constraint_mode": "shared-nic", "service_calibration_mode": "audited-v1",
            "service_calibration_document": {"schema": 1}, "temporal_sampling_mode": "stratified-jitter",
            "sampling_max_interval_seconds": 100}
    (run / "metadata.json").write_text(json.dumps({"args": args}), encoding="utf-8")
    assert checkpoint_run_args(directory / "latest.pt", {"training_repair": "batch-credit-v1"}) == args


def test_embedded_configuration_is_portable_and_authoritative(tmp_path):
    expected = {"network_constraint_mode": "shared-nic", "temporal_sampling_mode": "stratified-jitter"}
    assert checkpoint_run_args(tmp_path / "moved.pt", {"metadata": {"args": expected}}) == expected


def test_missing_repaired_configuration_fails_instead_of_using_legacy(tmp_path):
    with pytest.raises(ValueError, match="cannot safely restore"):
        checkpoint_run_args(tmp_path / "moved.pt", {"training_repair": "batch-credit-v1"})
    assert checkpoint_run_args(tmp_path / "legacy.pt", {}) == {}
