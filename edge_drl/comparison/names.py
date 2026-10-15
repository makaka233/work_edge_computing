"""Stable comparison identifiers and publication-facing scheme names.

Legacy identifiers remain accepted so archived commands and results can still
be read. DARP/DPMS implementation versions are selected separately by the
comparison runner; name canonicalization never rewrites archived results.
"""

SCHEME_LABELS = {
    "MSO": "Monolithic Service Orchestration (MSO)",
    "DPMS": "Demand-Aware Placement and Marginal-Cost Scheduling (DPMS)",
    "HEFT": "Heterogeneous Earliest Finish Time Scheduling (HEFT)",
    "DARP": "Demand-Aware Replica Placement (DARP)",
}

LEGACY_SCHEME_IDS = {
    "Monolithic": "MSO",
    "Algorithmic": "DPMS",
    "HEFTFast": "HEFT",
    "PopularityDeploy": "DARP",
}


def canonical_scheme_id(name: str) -> str:
    """Return the current short identifier for an old ID or full label."""
    if name in SCHEME_LABELS.values():
        return next(key for key, label in SCHEME_LABELS.items() if label == name)
    return LEGACY_SCHEME_IDS.get(name, name)


def display_scheme_name(name: str) -> str:
    """Return the full publication label, leaving unrelated schemes alone."""
    identifier = canonical_scheme_id(name)
    return SCHEME_LABELS.get(identifier, name)
