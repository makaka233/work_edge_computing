"""Standalone integrity/import/training/resume check for the Proposed bundle."""
from argparse import ArgumentParser
import csv
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def physical_signature():
    import train_dual_ppo as training
    from edge_drl.env.calibration import builtin_budget_document
    previous = sys.argv
    try:
        sys.argv = ["train_dual_ppo.py", "--seed", "2026", "--pressure-profile", "mec-latency200"]
        args = training.parse_args()
    finally:
        sys.argv = previous
    args.service_calibration_document = builtin_budget_document()
    env = training.build_env(args)
    env.reset()
    scenario = env.scenario
    data = {"nodes": [asdict(n) for n in scenario.nodes],
            "services": [asdict(s) for s in scenario.services],
            "users": [asdict(u) for u in scenario.users],
            "adjacency": scenario.adjacency.tolist(),
            "bandwidth": scenario.bandwidth_mb_s.tolist(),
            "propagation": scenario.propagation_ms.tolist()}
    def scalar(value):
        return value.item() if hasattr(value, "item") else value.tolist()
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), default=scalar).encode()).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true", help="Verify a fresh policy without optimizer steps or training runs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--proposed-design", choices=["joint-credit-v3", "joint-credit-v4", "joint-credit-v5"], default="joint-credit-v3")
    parser.add_argument("--output", type=Path, default=ROOT / "verification_report.json")
    options = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    for module in ("train_dual_ppo", "edge_drl.agents.drl", "edge_drl.env.calibration",
                   "edge_drl.allocators.nic_kkt", "edge_drl.comparison.checkpoint",
                   "edge_drl.comparison.replay_env"):
        loaded = importlib.import_module(module)
        assert Path(loaded.__file__).resolve().is_relative_to(ROOT), (module, loaded.__file__)
    import torch
    import numpy as np
    torch.set_num_threads(1)
    signature = physical_signature()
    assert signature == manifest["physical_signature_seed2026"], "source/bundle physical environment differs"
    from edge_drl.allocators.nic_kkt import nic_link_rates
    nic_checks = []
    for fixture in sorted((ROOT / "validation/fixtures").glob("*.npz")):
        data = np.load(fixture)
        rates, cost, (u, v) = nic_link_rates(data["loads"], data["capacities"], data["nic"],
                                           initial_duals=(data["outgoing"], data["incoming"]))
        active = data["loads"] > 0
        raw = np.minimum(np.where(active, data["capacities"], 0),
                         data["loads"] / np.sqrt(np.maximum(u[:, None] + v, 1e-30)))
        dual = np.sum(np.divide(data["loads"]**2, raw, out=np.zeros_like(raw), where=active))
        dual += np.dot(u, raw.sum(axis=1) - data["nic"]) + np.dot(v, raw.sum(axis=0) - data["nic"])
        gap = float((cost - dual) / cost)
        assert -1e-10 <= gap <= 1e-7
        assert np.all(rates.sum(axis=0) <= data["nic"] + 1e-9)
        assert np.all(rates.sum(axis=1) <= data["nic"] + 1e-9)
        nic_checks.append({"fixture": fixture.name, "relative_gap": gap})
    report = {"integrity": True, "isolated_imports": True, "physical_signature": signature,
              "nic_certificates": nic_checks, "device": options.device, "torch": torch.__version__,
              "training_resume_verified": False}
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    environment.pop("PYTHONPATH", None)
    environment["MPLCONFIGDIR"] = str(ROOT / ".matplotlib-cache")
    def run(arguments):
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError(result.stdout + "\n" + result.stderr)
        return result.stdout
    if options.inference_only:
        import train_dual_ppo as training
        from edge_drl.agents.drl import HierarchicalPPOAgent
        previous = sys.argv
        try:
            sys.argv = ['train_dual_ppo.py','--seed','2026','--pressure-profile','mec-latency200',
                        '--proposed-design',options.proposed_design,'--device',options.device]
            args = training.apply_proposed_design(training.parse_args())
        finally:
            sys.argv = previous
        env = training.build_env(args); env.reset()
        kwargs = {name:getattr(args,name) for name in inspect.signature(HierarchicalPPOAgent.from_env).parameters
                  if name not in {'env','cls'} and hasattr(args,name) and getattr(args,name) is not None}
        torch.manual_seed(2026)
        agent = HierarchicalPPOAgent.from_env(env,**kwargs)
        deployment = agent.slow_agent.plan_deployment(env,deterministic=True,record=False)
        assert env.check_deployment_feasible(deployment)[0]
        env.apply_deployment(deployment)
        agent.fast_agent.schedule_batch(env,env.current_requests[:1],deterministic=True,record=False)
        for ppo in [agent.slow_agent.count_ppo,agent.slow_agent.stage_ppo,agent.slow_agent.placement_ppo,agent.fast_agent.ppo]:
            assert not ppo.optimizer.state and len(ppo.buffer) == 0
        report['proposed_design'] = args.proposed_design
        report['inference_smoke_verified'] = True
        report['optimizer_steps'] = 0
    if not options.imports_only and not options.inference_only:
        common = ["train_dual_ppo.py", "--pressure-profile", "mec-latency200", "--proposed-design", options.proposed_design,
                  "--train-mode", "joint", "--rollout-unit", "window", "--replicas-per-stage", "0",
                  "--num-users", "10000", "--num-edge-nodes", "8", "--num-service-types", "2",
                  "--mean-requests-per-minute", "600",
                  "--episode-minutes", "60", "--fast-windows-per-update", "6", "--slow-windows-per-update", "6",
                  "--sampled-seconds-per-window", "1", "--sampling-max-interval-seconds", "600",
                  "--slow-critic-k-epochs", "2", "--slow-k-epochs", "1", "--slow-stage-k-epochs", "1",
                  "--slow-placement-k-epochs", "1", "--fast-k-epochs", "2", "--fast-minibatch-size", "64",
                  "--updates", "1", "--eval-interval", "0", "--device", options.device,
                  "--run-root", "_verification_runs", "--run-name", "smoke", "--progress-interval-seconds", "0"]
        target = ROOT / "_verification_runs/smoke"
        if target.exists():
            raise FileExistsError("verification run already exists; retain it and verify in a fresh extraction")
        run(common)
        checkpoint = target / "checkpoints/latest.pt"
        run(common + ["--load-checkpoint", str(checkpoint), "--append-log"])
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert data["metadata"]["completed_updates"] == 2
        args = data["metadata"]["args"]
        assert args["network_constraint_mode"] == "shared-nic"
        assert args["service_calibration_mode"] == "audited-v1"
        assert args["temporal_sampling_mode"] == "stratified-jitter"
        assert args["proposed_design"] == options.proposed_design
        assert args["slow_allocation_mode"] == {"joint-credit-v3":"stage-budget-v4", "joint-credit-v4":"stage-budget-v5", "joint-credit-v5":"stage-budget-v6"}[options.proposed_design]
        assert data["slow_quota_decode"] == ("categorical-mode" if options.proposed_design == "joint-credit-v3" else "categorical-mean")
        assert args["slow_learning_repair"] == ("prefix-structured-v3" if options.proposed_design == "joint-credit-v5" else "prefix-credit-v2")
        if options.proposed_design == "joint-credit-v5":
            assert args["slow_deployment_context"] == "deployment-chain-v3"
            assert (target/'logs/slow_update_metrics_prefix_structured_v3.csv').is_file()
        assert data["fast_independent_critic"]
        assert data["fast_update_chunks"] == 4
        best = torch.load(target / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        assert best["metadata"]["selection_kind"] == "post_update_nominal_probe_v2"
        with (target / "logs/training.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["update"]) for row in rows] == [1, 2]
        assert all(float(row["invalid_actions"]) == 0 for row in rows)
        assert all(float(row["slow_updated"]) == 1 and float(row["fast_optimizer_steps"]) > 0 for row in rows)
        assert all(float(row["sampled_request_events"]) < float(row["request_events"]) for row in rows)
        with (target / "logs/episode_metrics.csv").open(encoding="utf-8", newline="") as handle:
            episodes = list(csv.DictReader(handle))
        assert len(episodes) == 2 and all(float(row["episode_complete"]) == 1 for row in episodes)
        report["training_resume_verified"] = True
        for script in ("scripts/plot_training_run.py", "scripts/plot_rollout_training.py"):
            arguments = [script, str(target / "logs/training.csv") if "plot_training_run" in script else str(target)]
            run(arguments)
        report["plotting_verified"] = bool(list((target / "plots").glob("*.png")))
        run(["scripts/analyze_convergence.py", str(target / "logs/training.csv")])
        run(["scripts/calibrate_environment.py", "--output", "_verification_runs/resource_report.json"])
        # Fast-only initialization exercises a lazy import not used by fresh training.
        run(common + ["--run-name", "fast_init", "--init-fast-checkpoint", str(checkpoint), "--init-fast-actor-only"])
        report["fast_initialization_verified"] = True
        run(common + ["--run-name", "slow_init", "--init-fast-checkpoint", str(checkpoint),
                      "--init-fast-actor-only", "--init-slow-checkpoint", str(checkpoint)])
        initial = torch.load(ROOT / "_verification_runs/slow_init/checkpoints/initial.pt", map_location="cpu", weights_only=False)
        assert initial["metadata"]["completed_updates"] == 0
        assert initial["metadata"]["args"]["slow_initialization"]["source_update"] == 2
        report["slow_actor_initialization_verified"] = True
        report["initial_checkpoint_verified"] = True
        report["proposed_design"] = args["proposed_design"]
        report["post_update_probe_selection_verified"] = True
        from edge_drl.comparison.checkpoint import build_proposed_agent
        from train_dual_ppo import build_env
        from argparse import Namespace
        env = build_env(Namespace(**args)); env.reset()
        restored, _ = build_proposed_agent(env, checkpoint, args, device="cpu")
        restored.fast_agent.schedule_batch(env, env.current_requests[:1], deterministic=True, record=False)
        report["checkpoint_inference_verified"] = True
    for module, loaded in list(sys.modules.items()):
        if module.startswith("edge_drl") and getattr(loaded, "__file__", None):
            assert Path(loaded.__file__).resolve().is_relative_to(ROOT), module
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
