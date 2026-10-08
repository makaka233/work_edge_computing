"""Update publication scheme labels in one copied comparison result directory.

Only scheme-name fields are rewritten. The original comparison run, checkpoint
directories, trace identifiers, measurements, and scenario values are untouched.
The operation is idempotent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_drl.comparison.names import (
    LEGACY_SCHEME_IDS,
    SCHEME_LABELS,
    canonical_scheme_id,
    display_scheme_name,
)


NAME_COLUMNS = ("scheme", "baseline")


def relabel_run(run_dir: Path) -> dict[str, int]:
    if not (run_dir / "metadata.json").is_file() or not (run_dir / "summary" / "summary.csv").is_file():
        raise ValueError(f"not a comparison result directory: {run_dir}")

    changed: dict[str, int] = {}
    for path in sorted(run_dir.rglob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if fields is None:
                continue
            rows = list(reader)
        columns = [name for name in NAME_COLUMNS if name in fields]
        count = 0
        for row in rows:
            for column in columns:
                old = row[column]
                new = display_scheme_name(old)
                if new != old:
                    row[column] = new
                    count += 1
        if count:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            changed[path.relative_to(run_dir).as_posix()] = count

    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["formal_schemes"] = [canonical_scheme_id(name) for name in metadata["formal_schemes"]]
    metadata["scheme_display_names"] = {
        name: SCHEME_LABELS.get(name, name) for name in metadata["formal_schemes"]
    }
    metadata["labels"] = {
        canonical_scheme_id(name): detail for name, detail in metadata.get("labels", {}).items()
    }
    metadata["legacy_scheme_ids"] = LEGACY_SCHEME_IDS
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for file_name, count in relabel_run(args.run_dir).items():
        print(f"{file_name}: {count} scheme fields relabeled")


if __name__ == "__main__":
    main()
