"""Export the exact working-tree Proposed dependency closure as a standalone ZIP."""
from argparse import ArgumentParser
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dependency_closure(entries):
    files, pending = set(), list(entries)
    def add_module(module):
        base = ROOT.joinpath(*module.split("."))
        path = base.with_suffix(".py")
        if not path.is_file():
            path = base / "__init__.py"
        if path.is_file():
            pending.append(path.relative_to(ROOT).as_posix())
    while pending:
        relative = pending.pop()
        if relative in files:
            continue
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        files.add(relative)
        for parent in source.relative_to(ROOT).parents:
            initializer = ROOT / parent / "__init__.py"
            if parent != Path(".") and initializer.is_file():
                pending.append(initializer.relative_to(ROOT).as_posix())
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
            modules = [name.name for name in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if node.level:
                    package = source.relative_to(ROOT).parent.parts
                    module = ".".join((*package[:len(package) - node.level + 1], module))
                modules += [module, *(module + "." + name.name for name in node.names)]
            for module in modules:
                if module == "train_dual_ppo" or module.startswith("edge_drl"):
                    add_module(module)
    return sorted(files)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    created = datetime.now(timezone(timedelta(hours=8)))
    output = options.output or ROOT / "dist" / ("proposed_training_" + created.strftime("%Y%m%d_%H%M%S") + ".zip")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    entries = ["train_dual_ppo.py", "scripts/plot_rollout_training.py", "scripts/plot_training_run.py",
               "scripts/analyze_convergence.py", "scripts/calibrate_environment.py"]
    closure = dependency_closure(entries)
    # Keep only modules actually reached from Proposed and the selected tools.
    forbidden = {"monolithic", "unified", "algorithmic", "classic", "runner", "heuristic", "kkt_ablation"}
    assert not any(Path(path).stem in forbidden for path in closure), closure
    payload = {relative: (ROOT / relative).read_bytes() for relative in closure}
    source_mapping = {relative: relative for relative in closure}
    for name in ["LATENCY200_SCENARIO.md", "LATENCY200_TRAINING_REPAIR.md",
                 "SLOW_DEPLOYMENT_CONTEXT_REPAIR.md", "SHARED_NIC_PRECISION_REPAIR.md"]:
        relative = "docs/" + name
        payload[relative] = (ROOT / relative).read_bytes()
        source_mapping[relative] = relative
    for name in ["README.md", "verify_bundle.py", "train_proposed.ps1"]:
        source = "packaging/proposed/" + name
        payload[name] = (ROOT / source).read_bytes()
        source_mapping[name] = source
    for name in ["shared_nic_update50_failure.npz", "shared_nic_wide_scale_stress.npz"]:
        relative = "validation/fixtures/" + name
        source = "tests/fixtures/" + name
        payload[relative] = (ROOT / source).read_bytes()
        source_mapping[relative] = source
    from edge_drl.env.calibration import builtin_budget_document
    payload["config/service_budget.json"] = json.dumps(builtin_budget_document(), indent=2, ensure_ascii=False).encode("utf-8")
    dependencies = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "torch", "tqdm", "pandas", "matplotlib"]}
    requirements = "# Versions used to verify this bundle; CUDA build selection depends on target hardware.\n"
    requirements += "\n".join(name + "==" + version.split("+")[0] for name, version in dependencies.items()) + "\n"
    payload["requirements.txt"] = requirements.encode("utf-8")
    signature = runpy.run_path(str(ROOT / "packaging/proposed/verify_bundle.py"), run_name="bundle_helpers")["physical_signature"]()
    manifest = {
        "created_at": created.isoformat(), "scope": "Proposed training and its original evaluation/checkpoint/plot/calibration dependencies",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_is_working_tree_not_git_head": True,
        "source_status": subprocess.check_output(["git", "status", "--short", "--", *source_mapping.values()], cwd=ROOT, text=True).splitlines(),
        "python_verified": platform.python_version(), "dependencies_verified": dependencies,
        "physical_signature_seed2026": signature,
        "source_mapping": source_mapping,
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payload.items())},
        "excluded": [".git", "runs", "runs_tmp", "results", "cached files", "external model checkpoints",
                     "MSO/DPMS/HEFT/DARP/Unified scheme implementations"],
        "runtime_note": "Third-party packages, Python and GPU drivers must be installed on the target machine; no convergence guarantee.",
    }
    payload["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr("proposed_training/" + name, data)
    # Detect edits during export rather than silently mixing different revisions.
    for destination, source in source_mapping.items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == manifest["sha256"][destination], source
    print(json.dumps({"zip": str(output), "files": len(payload), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "python_modules": closure}, indent=2))


if __name__ == "__main__":
    main()
