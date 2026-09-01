"""Build the IEEE Access R1 upload package from explicit safe whitelists."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "revision_r1"
DESTINATION = ROOT / "output" / "submission_r1_final"


SUPPLEMENT_FILES = [
    (ROOT / "paper" / "supplementary_r1.pdf", "supplementary_r1.pdf"),
    (RESULTS / "feature_contract.json", "machine_readable/feature_contract.json"),
    (RESULTS / "outcome_definitions.csv", "machine_readable/outcome_definitions.csv"),
    (RESULTS / "outer_fold_assignments.csv", "machine_readable/outer_fold_assignments.csv"),
    (RESULTS / "internal_model_metrics.csv", "machine_readable/internal_model_metrics.csv"),
    (RESULTS / "internal_repeat_metrics.csv", "machine_readable/internal_repeat_metrics.csv"),
    (RESULTS / "internal_bootstrap_summary.csv", "machine_readable/internal_bootstrap_summary.csv"),
    (RESULTS / "internal_per_label_metrics.csv", "machine_readable/internal_per_label_metrics.csv"),
    (RESULTS / "internal_calibration_bins.csv", "machine_readable/internal_calibration_bins.csv"),
    (RESULTS / "primary_paired_difference.csv", "machine_readable/primary_paired_difference.csv"),
    (RESULTS / "primary_per_label_tests.csv", "machine_readable/primary_per_label_tests.csv"),
    (RESULTS / "primary_per_label_bootstrap.csv", "machine_readable/primary_per_label_bootstrap.csv"),
    (RESULTS / "primary_calibration_intervals.csv", "machine_readable/primary_calibration_intervals.csv"),
    (RESULTS / "gcn_ablation" / "gcn_ablation_10_seed_full.csv", "machine_readable/gcn_ablation_10_seed_full.csv"),
    (RESULTS / "gcn_ablation" / "gcn_ablation_paired_differences.csv", "machine_readable/gcn_ablation_paired_differences.csv"),
    (RESULTS / "gcn_ablation" / "gcn_ablation_metadata.json", "machine_readable/gcn_ablation_metadata.json"),
    (RESULTS / "gcn_ablation" / "gcn_checkpoint_specification.json", "machine_readable/gcn_checkpoint_specification.json"),
    (RESULTS / "temporal" / "temporal_model_metrics.csv", "machine_readable/temporal_model_metrics.csv"),
    (RESULTS / "temporal" / "temporal_per_label_metrics.csv", "machine_readable/temporal_per_label_metrics.csv"),
    (RESULTS / "temporal" / "temporal_metadata.json", "machine_readable/temporal_metadata.json"),
    (RESULTS / "temporal" / "temporal_checkpoint_specification.json", "machine_readable/temporal_checkpoint_specification.json"),
    (RESULTS / "synthetic" / "controlled_dependency_raw.csv", "machine_readable/controlled_dependency_raw.csv"),
    (RESULTS / "synthetic" / "controlled_dependency_summary.csv", "machine_readable/controlled_dependency_summary.csv"),
    (RESULTS / "synthetic" / "controlled_dependency_metadata.json", "machine_readable/controlled_dependency_metadata.json"),
    (RESULTS / "synthetic" / "controlled_dependency_checkpoint_specification.json", "machine_readable/controlled_dependency_checkpoint_specification.json"),
    (RESULTS / "external_transportability" / "external_transportability_results.csv", "machine_readable/external_transportability_results.csv"),
    (RESULTS / "external_transportability" / "external_transportability_paired_differences.csv", "machine_readable/external_transportability_paired_differences.csv"),
    (RESULTS / "external_transportability" / "external_transportability_calibration_bins.csv", "machine_readable/external_transportability_calibration_bins.csv"),
    (RESULTS / "external_transportability" / "external_transportability_cohort_summary.csv", "machine_readable/external_transportability_cohort_summary.csv"),
    (RESULTS / "external_transportability" / "external_transportability_metadata.json", "machine_readable/external_transportability_metadata.json"),
    (RESULTS / "runtime_environment.json", "machine_readable/runtime_environment.json"),
    (RESULTS / "SHA256SUMS.json", "machine_readable/SHA256SUMS.json"),
]


SOURCE_FILES = [
    ROOT / "paper" / "gai_revised_clean.tex",
    ROOT / "paper" / "ieeeaccess.cls",
    ROOT / "paper" / "bullet.png",
    ROOT / "paper" / "logo.png",
    ROOT / "paper" / "notaglinelogo.png",
]

SOURCE_GENERATED_FILES = [
    "table_synthetic_r1.tex",
    "table_internal_models.tex",
    "table_gcn_paired.tex",
    "table_primary_labels_r1.tex",
    "table_temporal_r1.tex",
    "table_calibration_r1.tex",
    "table_external_r1.tex",
]


def _require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required artifact(s) missing:\n" + "\n".join(missing))


def _write_zip(path: Path, entries: list[tuple[Path, str]]) -> None:
    forbidden = ("prediction", "bootstrap.npz", ".xlsx")
    for source, archive_name in entries:
        lower = archive_name.lower()
        if any(token in lower for token in forbidden):
            raise ValueError(f"Unsafe supplemental archive member: {archive_name}")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, archive_name in entries:
            archive.write(source, archive_name)


def _supplement_readme() -> Path:
    path = RESULTS / "supplement_zip_README.txt"
    text = """IEEE Access manuscript Access-2026-35668: supplementary machine-readable results

The archive contains the supplementary PDF, aggregate result tables, fold
assignments for the 1,700-row public Krasnoyarsk benchmark, configuration
metadata, and reproducibility checksums. It deliberately excludes the
Hungarian registry workbook, every patient-level prediction array, and all
bootstrap resampling arrays. The Hungarian data remain available from their
original Mendeley Data record under the provider's terms.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _source_entries() -> list[tuple[Path, str]]:
    entries = [(path, path.relative_to(ROOT / "paper").as_posix()) for path in SOURCE_FILES]
    generated = ROOT / "paper" / "generated_r1"
    for name in SOURCE_GENERATED_FILES:
        path = generated / name
        entries.append((path, f"paper/generated_r1/{name}"))
    photos = ROOT / "paper" / "author_photos"
    for path in sorted(photos.rglob("*")):
        if path.is_file():
            entries.append((path, f"author_photos/{path.relative_to(photos).as_posix()}"))
    for path in sorted((ROOT / "figures" / "r1").glob("*.pdf")):
        entries.append((path, f"figures/r1/{path.name}"))
    for path in sorted((ROOT / "paper").glob("t1-*")):
        if path.is_file():
            entries.append((path, path.name))
    for path in sorted((ROOT / "paper").glob("t1*.fd")):
        if path.is_file():
            entries.append((path, path.name))
    return entries


def _manifest(folder: Path) -> None:
    rows = {}
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            rows[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "SHA256SUMS.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def package(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    direct = [
        (ROOT / "revision" / "response_to_reviewers.docx", "Access-2026-35668_Author_Response.docx"),
        (ROOT / "output" / "pdf" / "gai_revised_highlighted.pdf", "Access-2026-35668_Highlighted.pdf"),
        (ROOT / "paper" / "gai_revised_clean.pdf", "Access-2026-35668_Clean.pdf"),
    ]
    _require([source for source, _ in direct])
    for source, name in direct:
        shutil.copy2(source, destination / name)

    readme = _supplement_readme()
    supplement_entries = SUPPLEMENT_FILES + [(readme, "README.txt")]
    _require([source for source, _ in supplement_entries])
    _write_zip(destination / "Access-2026-35668_Supplementary_Materials.zip", supplement_entries)

    source_entries = _source_entries()
    _require([source for source, _ in source_entries])
    _write_zip(destination / "Access-2026-35668_Clean_LaTeX_Source.zip", source_entries)
    _manifest(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package(args.destination)
    print(f"Submission package written to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
