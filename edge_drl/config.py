from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    return data


def service_name_to_id(config: dict[str, Any]) -> dict[str, int]:
    return {svc["name"]: idx for idx, svc in enumerate(config["services"])}


def node_name_to_id(config: dict[str, Any]) -> dict[str, int]:
    return {node["name"]: idx for idx, node in enumerate(config["nodes"])}

