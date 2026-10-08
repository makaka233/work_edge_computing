import csv
import json

from edge_drl.comparison.names import canonical_scheme_id, display_scheme_name
from scripts.rename_final_comparison_schemes import relabel_run


def test_current_and_legacy_scheme_names() -> None:
    expected = {
        "Monolithic": ("MSO", "Monolithic Service Orchestration (MSO)"),
        "Algorithmic": ("DPMS", "Demand-Aware Placement and Marginal-Cost Scheduling (DPMS)"),
        "HEFTFast": ("HEFT", "Heterogeneous Earliest Finish Time Scheduling (HEFT)"),
        "PopularityDeploy": ("DARP", "Demand-Aware Replica Placement (DARP)"),
    }
    for legacy, (identifier, label) in expected.items():
        assert canonical_scheme_id(legacy) == identifier
        assert canonical_scheme_id(label) == identifier
        assert display_scheme_name(legacy) == label
        assert display_scheme_name(identifier) == label
        assert display_scheme_name(label) == label


def test_final_result_relabel_is_idempotent_and_preserves_metrics(tmp_path) -> None:
    run = tmp_path / "comparison"
    (run / "summary").mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps({"formal_schemes": ["Proposed", "Monolithic", "Algorithmic"], "labels": {"Monolithic": "PPO"}}),
        encoding="utf-8",
    )
    summary = run / "summary" / "summary.csv"
    summary.write_text(
        "scheme,metric,mean\nMonolithic,mean_latency_ms,123.456\nAlgorithmic,mean_latency_ms,234.567\n",
        encoding="utf-8",
    )
    assert relabel_run(run) == {"summary/summary.csv": 2}
    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mean"] for row in rows] == ["123.456", "234.567"]
    assert [row["scheme"] for row in rows] == [
        "Monolithic Service Orchestration (MSO)",
        "Demand-Aware Placement and Marginal-Cost Scheduling (DPMS)",
    ]
    assert relabel_run(run) == {}
