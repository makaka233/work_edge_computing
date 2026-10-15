"""Recent workload correction and isolated post-update checkpoint probes."""
from copy import deepcopy
import random
import numpy as np
import torch
from scipy.optimize import lsq_linear


def recent_workload_score(history, window, reference_load, reference_compute, horizon=80):
    records = [r for r in history if all(np.isfinite(r.get(k, np.nan))
               for k in ("latency_s", "load_multiplier", "compute_gcycles"))][-max(window, horizon):]
    if not records:
        return float("inf"), 0.
    y = np.asarray([r["latency_s"] for r in records])
    load = np.asarray([r["load_multiplier"] for r in records])
    work = np.asarray([r["compute_gcycles"] for r in records])
    t = np.linspace(-1., 1., len(records))
    matrix = np.c_[np.ones(len(y)), load-load.mean(), work-work.mean(), t]
    # Positive workload/load effects; a free time trend separates ongoing
    # policy improvement from load response. This is a diagnostic, not an eval.
    ridge = np.diag([0., .01, .01, .01])
    fit = lsq_linear(np.r_[matrix, ridge], np.r_[y, np.zeros(4)],
                     bounds=([-np.inf, 0, 0, -np.inf], [np.inf]*4)).x
    corrected = y - fit[1]*(load-reference_load) - fit[2]*(work-reference_compute)
    return float(corrected[-window:].mean()), float(fit[1])


def checkpoint_probe(agent, config, *, seed, samples, hold_cap):
    """Evaluate the EXACT post-update policy without touching training state.

    One fixed nominal deployment window; not a five-scenario validation claim.
    The private trace/RNG does not enter actor training or critic replay.
    """
    from edge_drl.env.environment import EdgeComputingEnv
    from edge_drl.comparison.trace import generate_comparison_trace
    from edge_drl.comparison.replay_env import TraceReplayEnv
    from edge_drl.comparison.scheme import BaseComparisonScheme
    from edge_drl.comparison.proposed import ProposedScheme
    from edge_drl.comparison.metrics import evaluate_scheme_episode
    from edge_drl.comparison.types import ExperimentPoint
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        cfg = deepcopy(config)
        cfg.seed = seed + 2_000_003
        cfg.scenario_seed = seed
        cfg.episode_minutes = cfg.deployment_interval_minutes
        cfg.episode_hours = None
        cfg.demand_load_multiplier = 1.
        cfg.demand_load_group = None
        physical = EdgeComputingEnv(cfg)
        physical.reset()
        trace = generate_comparison_trace(
            scenario=physical.scenario, logical_steps=int(cfg.episode_minutes*60),
            requests_per_minute=physical._arrival_rate_per_minute(),
            physical_seed=cfg.physical_seed if cfg.physical_seed is not None else cfg.seed,
            demand_seed=seed, request_seed=seed+3_000_017,
            task_compute_scale=cfg.task_compute_scale, task_data_scale=cfg.task_data_scale)
        env = TraceReplayEnv(cfg, physical.scenario, trace)
        scheme = ProposedScheme.__new__(ProposedScheme)
        BaseComparisonScheme.__init__(scheme)
        scheme.agent = deepcopy(agent)
        scheme.fast_deterministic = True
        scheme._initialize_window_feedback()
        result = evaluate_scheme_episode(
            scheme=scheme, env=env, point=ExperimentPoint("request_load", 1., "checkpoint_probe"),
            eval_seed=seed, routing_repeat=0, sampled_seconds_per_window=samples,
            sampling_protocol="training-equivalent-v2", sampling_max_interval_seconds=hold_cap)
        if result.failed or result.invalid_action_rate:
            raise RuntimeError("checkpoint probe produced invalid actions")
        return {"mean_latency_s": result.mean_latency_ms/1000., "trace_hash":trace.trace_hash,
                "sampled_steps":result.settlement_steps, "request_count":result.request_count,
                "runtime_s":result.total_runtime_s, "minutes":cfg.episode_minutes,
                "selection_kind":"post_update_nominal_probe_v2"}
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
