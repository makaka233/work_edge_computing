# Hardware-constrained MEC scenario

`--pressure-profile mec-realistic-constrained` is an opt-in physical scenario.
It does not alter the legacy `baseline`, `mec-moderate`, or `mec-stress`
generators.

## Hardware anchors

The node catalogue is grounded in three deployed edge-computing classes:

| Simulated class | Hardware anchor | Mix | Memory | Provisioned storage | Effective compute | NIC cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Device / micro edge | Raspberry Pi 5 | 45% | 8 GB | 128 GB | 38.4 Gcycle/s | 125 MB/s |
| Accelerated edge | NVIDIA Jetson Orin Nano | 40% | 8 GB | 256 GB | 72 Gcycle/s | 125 MB/s |
| Regional edge | Dell PowerEdge XR4000 | 15% | 128 GB | 1,920 GB | 168 Gcycle/s | 1,250 MB/s |

Each generated node receives bounded multiplicative variation. Link capacity is
also capped by the slower endpoint NIC before the scenario bandwidth scale is
applied.

Primary hardware references:

- Raspberry Pi 5: quad-core 2.4 GHz Cortex-A76, up to 16 GB RAM, Gigabit
  Ethernet, and optional PCIe-attached storage:
  <https://www.raspberrypi.com/products/raspberry-pi-5/>
- NVIDIA Jetson Orin Nano Super: 6-core Arm CPU, 8 GB LPDDR5, 67 sparse INT8
  TOPS, and Gigabit Ethernet:
  <https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/>
- Dell PowerEdge XR4000: Xeon D with up to 20 cores, up to 512 GB memory,
  M.2 NVMe storage, and 10/25/40/50/100 GbE options:
  <https://i.dell.com/sites/csdocuments/Product_Docs/en/dell-poweredge-xr4000-technical-guide.pdf>

## Calibration boundary

`compute_gcycles_per_s` is an effective application-throughput abstraction. It
is deliberately not a direct conversion of CPU GHz or GPU TOPS. The values keep
the ordering and resource ratios of the hardware classes, then calibrate the
absolute scale so the default workload creates contention without permanent
saturation.

The profile uses 60% of physical memory/storage for managed services. Despite
the larger fraction than `mec-moderate`, the much smaller device-edge memory
reduces the aggregate deployment pool. A multi-stage service may therefore fail
to fit as one monolith on a device node while its individual stages remain
feasible, which is a real resource-fragmentation advantage of staged placement.

## Edge AI service demand

The same pressure profile selects `edge-ai-pipelines`. Unlike the legacy
catalogue, it does not draw unrelated memory and storage values for every
stage. Each service is a bounded multi-stage pipeline with explicit compute,
runtime memory, model/container storage, input payload, and intermediate tensor
or result payload.

Representative structures include:

- speech feature extraction followed by ASR inference;
- video decode/preprocessing, object inference, then tracking/postprocessing;
- perception followed by planning for robots, drones, and connected vehicles;
- preprocessing, anomaly inference, and result filtering for medical telemetry.

The workload families follow MLPerf Edge's object-detection, speech, language,
and medical inference workloads: <https://github.com/mlcommons/inference>.
Video pipelines follow NVIDIA DeepStream's decode/batch/inference/tracking
structure: <https://developer.nvidia.com/deepstream-sdk>. Speech runtime memory
is bounded around the published OpenAI Whisper small-model requirement rather
than a large cloud model: <https://github.com/openai/whisper>.

These references establish workload type and order of magnitude. Simulator
values remain calibrated effective demands; they are not presented as exact
measurements of one model build. The catalogue applies bounded implementation
variation, a 15% serving-runtime memory allowance, and a 50% retained
engine/container storage allowance. Every individual stage remains feasible on
regional nodes, while heavier inference stages may intentionally exceed the
service partition of a device-class node.

The default load strata are `0.65:0.80`, `0.80:1.00`, `1.00:1.20`, and
`1.20:1.45`, with probabilities `0.20,0.45,0.25,0.10`. Wired bandwidth uses a
0.75 scale: compute and deployment memory are intended bottlenecks, while the
network remains usable enough that cross-node placement is sometimes rational.

## Fair comparison rule

Train Proposed first with this profile. Its checkpoint embeds the complete
environment configuration. Train Monolithic from that Proposed checkpoint and
run comparison evaluation from the same checkpoint metadata. Do not compare a
legacy-environment checkpoint against a hardware-constrained checkpoint.
