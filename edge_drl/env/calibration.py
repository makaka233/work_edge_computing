"""Auditable workload budgets and optional measured-budget import.

Built-in compute/context budgets are engineering estimates, NOT measurements.
Gcycle is an effective-work unit: C=F_reference*t_reference, never GHz/TOPS.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from edge_drl.env import latency200


REFERENCES = {
    "runtime": "https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html",
    "speech": "https://github.com/openai/whisper",
    "detector": "https://docs.ultralytics.com/models/yolov8/",
    "medical": "https://physionet.org/content/ptb-xl/1.0.3/",
    "cloud": "https://pointclouds.org/documentation/tutorials/compression.html",
}


def builtin_budget_document():
    services = []
    # Declared FP16 artifact budget where parameter counts are documented;
    # otherwise a reserve, not a claim about an actual loaded model file.
    model_weights = [.148, .0064, .0224, .0234, .0224, .0224, .0, .020, .0224, .020]
    for service_id, profile in enumerate(latency200.SERVICE_PROFILES):
        stages = []
        for i, compute in enumerate(profile["stage_compute"]):
            weights = model_weights[service_id] if i == 1 else 0.0
            resident = .128 + weights if service_id == 0 and i == 1 else .064 + weights
            memory = float(profile["stage_memory"][i]) * 1.15
            # Two resident contexts; this is a memory reservation, not a
            # promise of twofold throughput or a per-request queue simulation.
            context = (memory - resident) / 2.0
            stages.append({"compute_ms": compute / 72.0 * 1000,
                           "resident_memory_gb": resident, "context_memory_gb": context,
                           "contexts": 2, "storage_gb": float(profile["stage_storage"][i]) * 1.5,
                           "output_mb": profile["stage_output"][i],
                           "weights_gb": weights, "status": "estimated"})
        services.append({"name": profile["name"], "input_mb": profile["input_mb"], "stages": stages})
    return {"schema": 1, "reference_compute_gcycles_s": 72.0,
            "units": {"data": "MB", "memory": "GB", "time": "ms", "compute": "Gcycle/s"},
            "provenance": {"status": "estimated", "references": REFERENCES,
                           "note": "effective compute and execution-context budgets require device profiling"},
            "services": services}


def load_budget_document(path=None):
    if path is None:
        return builtin_budget_document()
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    validate_budget_document(document)
    return document


def validate_budget_document(document):
    if document.get("schema") != 1 or document.get("units") != {
        "data": "MB", "memory": "GB", "time": "ms", "compute": "Gcycle/s"}:
        raise ValueError("calibration schema/units must be explicit and match schema 1")
    reference = float(document["reference_compute_gcycles_s"])
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("invalid effective reference compute rate")
    provenance = document.get("provenance", {})
    if provenance.get("status") not in {"estimated", "measured"}:
        raise ValueError("calibration provenance must say estimated or measured")
    if provenance["status"] == "measured":
        required = ("hardware", "power_mode", "precision", "runtime_version", "input_spec", "measurement_file", "sample_count")
        if any(not provenance.get(key) for key in required) or int(provenance["sample_count"]) < 30:
            raise ValueError("measured calibration requires profiling provenance and at least 30 samples")
    rows = document.get("services", [])
    if len(rows) != 10 or [r.get("name") for r in rows] != [r["name"] for r in latency200.SERVICE_PROFILES]:
        raise ValueError("calibration must contain the ordered ten-service catalogue")
    for row, profile in zip(rows, latency200.SERVICE_PROFILES):
        if not np.isfinite(row["input_mb"]) or row["input_mb"] <= 0 or len(row["stages"]) != len(profile["stage_compute"]):
            raise ValueError("invalid input/stage dimensions")
        for stage in row["stages"]:
            for field in ("compute_ms", "resident_memory_gb", "context_memory_gb", "storage_gb", "output_mb", "weights_gb"):
                if not np.isfinite(stage[field]) or stage[field] < 0:
                    raise ValueError("stage budgets must be finite and non-negative")
            contexts = stage["contexts"]
            if not isinstance(contexts, int) or contexts < 1 or stage["compute_ms"] <= 0:
                raise ValueError("invalid execution-context budget")
            if stage["weights_gb"] > min(stage["resident_memory_gb"], stage["storage_gb"]):
                raise ValueError("weights must fit both resident memory and incremental storage")


def apply_service_budgets(services, path=None, document=None):
    document = load_budget_document(path) if document is None else document
    validate_budget_document(document)
    reference = float(document["reference_compute_gcycles_s"])
    result = []
    for service in services:
        row = document["services"][service.service_id]
        stages = []
        for stage in service.stages:
            budget = row["stages"][stage.stage_id]
            stages.append(replace(stage,
                compute_gcycles_mean=reference * budget["compute_ms"] / 1000,
                output_mb_mean=budget["output_mb"],
                memory_gb=budget["resident_memory_gb"] + budget["contexts"] * budget["context_memory_gb"],
                storage_gb=budget["storage_gb"], resident_memory_gb=budget["resident_memory_gb"],
                context_memory_gb=budget["context_memory_gb"], reserved_contexts=budget["contexts"]))
        # Fixed input formats: 11s PCM, 1024² gray, 10s ECG. JPEG and point
        # count distributions remain variable; decoded fixed-size tensors do not.
        fixed_input = service.service_id in {0, 3, 7}
        fixed_outputs = ({0: {0}, 1: {0}, 2: {0}, 3: {0, 1}, 4: {0}, 5: {0},
                          7: {0, 1}, 8: {0}}).get(service.service_id, set())
        coupling = tuple(service.service_id in {6, 9} and i == 0 for i in range(len(stages)))
        result.append(replace(service, stages=tuple(stages), input_mb_mean=row["input_mb"],
                              coherent_sampling=True, input_sigma=0.0 if fixed_input else .35,
                              output_sigmas=tuple(0.0 if i in fixed_outputs else .35 for i in range(len(stages))),
                              output_input_coupling=coupling))
    return result
