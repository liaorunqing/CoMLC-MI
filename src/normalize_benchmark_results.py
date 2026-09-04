"""Create the neutral benchmark result tree from the verified analysis outputs.

This is a metadata/name migration only: numeric arrays and estimates are not
recomputed.  The historical result tree remains untouched for audit purposes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPLACEMENTS = {
    "TabPFN v2": "TabPFN-3",
    "RAkEL-RF": "RAkELd-RF",
    "ML-KNN": "BR-kNN (distance-weighted, k=10)",
    "Revision-stage specification": "Specification",
}


def _replace_text(value: str) -> str:
    for old, new in REPLACEMENTS.items():
        value = value.replace(old, new)
    return value


def _normalize_csv(path: Path) -> None:
    frame = pd.read_csv(path)
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].map(lambda value: _replace_text(value) if isinstance(value, str) else value)
    frame.to_csv(path, index=False)


def _normalize_json(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    rendered = _replace_text(json.dumps(value, indent=2, ensure_ascii=False))
    path.write_text(rendered, encoding="utf-8")


def _normalize_npz(path: Path) -> None:
    stored = np.load(path)
    payload = {key: stored[key] for key in stored.files}
    if "rakel_rf" in payload and "rakeld_rf" not in payload:
        payload["rakeld_rf"] = payload.pop("rakel_rf")
    np.savez_compressed(path, **payload)


def run(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")
    shutil.copytree(source, destination)
    for path in destination.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".csv":
            _normalize_csv(path)
        elif path.suffix.lower() == ".json":
            _normalize_json(path)
        elif path.name == "internal_oof_predictions.npz" or path.parent.name == "checkpoints" and path.suffix == ".npz":
            _normalize_npz(path)
    obsolete = destination / "supplement_zip_README.txt"
    if obsolete.exists():
        obsolete.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/revision_r1"))
    parser.add_argument("--destination", type=Path, default=Path("output/benchmark"))
    args = parser.parse_args()
    run(args.source, args.destination)


if __name__ == "__main__":
    main()
