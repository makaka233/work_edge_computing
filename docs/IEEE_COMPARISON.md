# IEEE formal scheme comparison

This framework is isolated under `edge_drl/comparison/`. It reuses the existing
physical simulator, radio model, multi-hop routing, contention model, and KKT
settlement without changing the training entry point or the Proposed policy.

The paper-facing schemes are exactly:

- **Proposed**: deterministic inference from an explicit checkpoint; no retraining.
- **Monolithic**: optimization-backed whole-service placement, adapted from [2];
  it is not presented as a reproduction of SD3.
- **DMDR**: native AES-JDR/RDMP adaptation following Peng et al., IEEE TSC 2024.
  Integer instance multiplicity is retained as a native diagnostic; the common
  physical simulator projects it to `N_integer > 0`, and memory/storage are
  enforced on that indicator.
- **SICP**: adapted JPS-CP chain placement with OR-Tools CP-SAT; TSN gate
  scheduling is outside this MEC simulator and is explicitly omitted.

Greedy remains available only as a development sanity check and is never added
to formal plots or the main result tables.

Every evaluation point and seed first creates a complete immutable one-second
request trace. Each scheme receives the identical trace and an independent copy
of the same transformed physical scenario. Formal episodes contain 3,600
logical and settlement steps (60 minutes); Phase 1 intentionally uses 600 steps
for correctness screening. No temporal sampling is used in comparison runs.

## Phases

1. Phase 1: nominal load, one seed, 10 minutes.
2. Phase 2: all five parameter sweeps, three paired seeds, 60 minutes.
3. Phase 3: all sweeps, twenty paired seeds, 60 minutes; DMDR routing is repeated
   three times and averaged within each seed before Student-t confidence intervals.
   The runner refuses Phase 3 unless `--phase2-validation-run` points to a complete,
   failure-free Phase 2 result directory.

Run Phase 1 from PowerShell:

```powershell
python run_comparison.py `
  --checkpoint "runs\placement_credit_episode60_large_s30_u320_20260826_123639\checkpoints\best.pt" `
  --phase 1 `
  --device cuda
```

Plot a completed run:

```powershell
python scripts\plot_comparison.py "results\comparison\<run_id>"
```
