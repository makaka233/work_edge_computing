"""Opt-in latency200 candidate parameters, not hardware measurements.

Keep the original catalogue schema and its 1.15 memory / 1.50 storage
allowances. Runtime/workspace and effective Gcycle budgets require profiling;
the hardware and payload references are documented in LATENCY200_SCENARIO.md.
These constants do not prescribe scheduling actions or truncate latency.
"""

NODE_PROFILE = "jetson-ai-edge"
SERVICE_PROFILE = "edge-ai-latency200"

NODE_TYPES = ("jetson-orin-nano-8gb", "jetson-orin-nx-16gb", "jetson-agx-orin-32gb")
NODE_MEMORY_GB = (8.0, 16.0, 32.0)
NODE_STORAGE_GB = (128.0, 256.0, 1024.0)
NODE_COMPUTE_GCYCLES_S = (48.0, 72.0, 144.0)
NODE_NIC_MB_S = (125.0, 125.0, 1250.0)

# stage_memory/storage are generator BASE values. Divide the agreed final
# reserves by the existing generator allowances to avoid counting them twice.
# Payloads and compute below are nominal catalogue values, not sampled means.
# Preserve the previous business deadlines; the target is physical latency,
# not a lower violation rate produced by relaxing deadlines.
SERVICE_PROFILES = [
    {
        "name": "speech-recognition",
        "stage_compute": [0.360, 21.600],
        "stage_memory": [0.20 / 1.15, 1.50 / 1.15],
        "stage_storage": [0.05 / 1.50, 0.30 / 1.50],
        "stage_output": [0.960, 0.004],
        "input_mb": 0.352,
        "deadline_s": 0.20,
    },
    {
        "name": "ar-overlay",
        "stage_compute": [0.288, 0.432, 0.216],
        "stage_memory": [0.18 / 1.15, 0.70 / 1.15, 0.12 / 1.15],
        "stage_storage": [0.05 / 1.50, 0.04 / 1.50, 0.02 / 1.50],
        "stage_output": [1.229, 0.0016, 0.002],
        "input_mb": 0.300,
        "deadline_s": 0.24,
    },
    {
        "name": "video-object-detection",
        "stage_compute": [0.288, 1.296, 0.288],
        "stage_memory": [0.18 / 1.15, 1.00 / 1.15, 0.18 / 1.15],
        "stage_storage": [0.05 / 1.50, 0.10 / 1.50, 0.03 / 1.50],
        "stage_output": [1.229, 0.0032, 0.0048],
        "input_mb": 0.152,
        "deadline_s": 0.38,
    },
    {
        "name": "industrial-inspection",
        "stage_compute": [0.432, 2.880, 0.216],
        "stage_memory": [0.25 / 1.15, 1.10 / 1.15, 0.18 / 1.15],
        "stage_storage": [0.05 / 1.50, 0.10 / 1.50, 0.03 / 1.50],
        "stage_output": [1.049, 0.1311, 0.004],
        "input_mb": 1.049,
        "deadline_s": 0.34,
    },
    {
        "name": "traffic-perception",
        "stage_compute": [0.288, 1.296, 0.432],
        "stage_memory": [0.18 / 1.15, 1.00 / 1.15, 0.24 / 1.15],
        "stage_storage": [0.05 / 1.50, 0.10 / 1.50, 0.04 / 1.50],
        "stage_output": [1.229, 0.0064, 0.002],
        "input_mb": 0.300,
        "deadline_s": 0.30,
    },
    {
        "name": "smart-retail-event",
        "stage_compute": [0.288, 1.296, 0.360],
        "stage_memory": [0.18 / 1.15, 1.00 / 1.15, 0.20 / 1.15],
        "stage_storage": [0.05 / 1.50, 0.10 / 1.50, 0.03 / 1.50],
        "stage_output": [1.229, 0.0032, 0.004],
        "input_mb": 0.250,
        "deadline_s": 0.26,
    },
    {
        "name": "robot-control",
        "stage_compute": [0.576, 1.080, 0.360],
        "stage_memory": [0.30 / 1.15, 0.50 / 1.15, 0.25 / 1.15],
        "stage_storage": [0.08 / 1.50, 0.10 / 1.50, 0.08 / 1.50],
        "stage_output": [3.084, 0.040, 0.002],
        "input_mb": 0.210,
        "deadline_s": 0.14,
    },
    {
        "name": "medical-vital-anomaly",
        "stage_compute": [0.288, 0.576, 0.144],
        "stage_memory": [0.12 / 1.15, 0.40 / 1.15, 0.10 / 1.15],
        "stage_storage": [0.03 / 1.50, 0.08 / 1.50, 0.02 / 1.50],
        "stage_output": [0.240, 0.000284, 0.001],
        "input_mb": 0.120,
        "deadline_s": 0.18,
    },
    {
        "name": "drone-inspection",
        "stage_compute": [0.576, 4.680, 0.432],
        "stage_memory": [0.30 / 1.15, 1.50 / 1.15, 0.25 / 1.15],
        "stage_storage": [0.05 / 1.50, 0.15 / 1.50, 0.05 / 1.50],
        "stage_output": [4.915, 0.0064, 0.008],
        "input_mb": 0.148,
        "deadline_s": 0.36,
    },
    {
        "name": "connected-vehicle-planning",
        "stage_compute": [0.360, 2.880, 0.720],
        "stage_memory": [0.25 / 1.15, 1.20 / 1.15, 0.35 / 1.15],
        "stage_storage": [0.08 / 1.50, 0.10 / 1.50, 0.08 / 1.50],
        "stage_output": [1.600, 0.0064, 0.002],
        "input_mb": 0.801,
        "deadline_s": 0.24,
    },
]
