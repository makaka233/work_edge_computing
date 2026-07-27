from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def timestamped_run_dir(root: str | Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

