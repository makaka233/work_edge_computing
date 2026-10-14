"""Replay one update in an isolated run and capture a NIC failure as test data."""
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edge_drl.allocators.nic_kkt as solver
import edge_drl.agents.drl as agents

original = solver.nic_link_rates
repaired = original
arguments = list(sys.argv[1:])
if "--reference-commit" in arguments:
    index = arguments.index("--reference-commit")
    revision = arguments[index + 1]
    del arguments[index:index + 2]
    namespace = {"__name__": "reference_nic_solver"}
    exec(subprocess.check_output(["git", "show", revision + ":edge_drl/allocators/nic_kkt.py"], text=True), namespace)
    original = namespace["nic_link_rates"]


def capture(loads, capacities, nic, **kwargs):
    try:
        return original(loads, capacities, nic, **kwargs)
    except RuntimeError:
        target = Path("runs_tmp/nic_update50_failure.npz")
        target.parent.mkdir(exist_ok=True)
        warm = kwargs.get("initial_duals") or (np.zeros(len(nic)), np.zeros(len(nic)))
        np.savez_compressed(target, loads=loads, capacities=capacities, nic=nic,
                            outgoing=warm[0], incoming=warm[1])
        print("NIC numerical fixture:", target.resolve(), flush=True)
        if original is repaired:
            raise
        bandwidth, cost, duals = repaired(loads, capacities, nic, **kwargs)
        u, v = duals
        active = np.asarray(loads) > 0
        raw = np.minimum(np.where(active, capacities, 0), loads / np.sqrt(np.maximum(u[:, None] + v, 1e-30)))
        dual = np.sum(np.divide(loads ** 2, raw, out=np.zeros_like(raw), where=active))
        dual += np.dot(u, raw.sum(axis=1) - nic) + np.dot(v, raw.sum(axis=0) - nic)
        print("Repaired NIC certificate:", (cost - dual) / cost,
              "capacity ratio:", max(np.max(bandwidth.sum(axis=0) / nic), np.max(bandwidth.sum(axis=1) / nic)), flush=True)
        return bandwidth, cost, duals


solver.nic_link_rates = agents.nic_link_rates = capture
sys.argv = ["train_dual_ppo.py", *arguments]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "train_dual_ppo.py"), run_name="__main__")
