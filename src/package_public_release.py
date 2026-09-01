"""Build a privacy-audited public release candidate for the R1 fixed tag."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "revision_r1"
OUTPUT = ROOT / "output" / "public_release_candidate_access-2026-35668-r1.zip"
AUDIT = ROOT / "output" / "public_release_candidate_access-2026-35668-r1_audit.json"
PREFIX = "CoMLC-MI-access-2026-35668-r1"

TOP_LEVEL = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pytest.ini",
    ".gitignore",
]

FORMAL_SOURCE = [
    "__init__.py",
    "run_revision.py",
    "revision_features.py",
    "revision_validation.py",
    "revision_models.py",
    "revision_metrics.py",
    "revision_losses.py",
    "revision_gcn.py",
    "revision_temporal.py",
    "statistical_utils.py",
    "synthetic_experiments.py",
    "external_transportability.py",
    "label_metrics.py",
    "generate_r1_figures.py",
    "generate_external_transportability_figure.py",
    "generate_r1_latex.py",
    "generate_r1_supplement.py",
    "generate_highlighted_pdf.py",
    "generate_response_docx.py",
    "package_r1_submission.py",
    "package_public_release.py",
    "revision_r1_qa.py",
    "verify_r1_references.py",
]

DATASET_FILES = [
    "Myocardial infarction complications Database.csv",
    "Myocardial infarction complications Database description.pdf",
    "Descriptive statistics.pdf",
]

FORBIDDEN_ARCHIVE_TOKENS = (
    "hungarian myocardial infarction registry extracted database.xlsx",
    "external_transportability_predictions.npz",
    "external_transportability_bootstrap.npz",
)
FORBIDDEN_TEXT_TOKENS = (
    "c:" + "\\users\\" + "liaoq",
    "c:" + "/users/" + "liaoq",
)


def add(entries: dict[str, Path], source: Path, archive_name: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    normalized = archive_name.replace("\\", "/")
    if any(token in normalized.casefold() for token in FORBIDDEN_ARCHIVE_TOKENS):
        raise ValueError(f"Forbidden public-release member: {normalized}")
    entries[f"{PREFIX}/{normalized}"] = source


def collect() -> tuple[dict[str, Path], list[str]]:
    entries: dict[str, Path] = {}
    exclusions: list[str] = []
    for name in TOP_LEVEL:
        add(entries, ROOT / name, name)
    add(entries, ROOT / "configs" / "access_2026_35668_r1.json", "configs/access_2026_35668_r1.json")
    for name in FORMAL_SOURCE:
        add(entries, ROOT / "src" / name, f"src/{name}")
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        add(entries, path, f"tests/{path.name}")
    for name in DATASET_FILES:
        add(entries, ROOT / "dataset" / name, f"dataset/{name}")
    for path in sorted((ROOT / "figures" / "r1").glob("*")):
        # PDF/SVG provide editable publication artwork and PNG provides a
        # lightweight preview. Redundant 600-dpi TIFF exports remain in the
        # manuscript workspace but are omitted from the public Git release to
        # keep the fixed reproducibility snapshot practical to clone.
        if path.is_file() and path.suffix.casefold() != ".tiff":
            add(entries, path, f"figures/r1/{path.name}")

    for path in sorted(RESULTS.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(RESULTS).as_posix()
        lower = relative.casefold()
        if relative == "SHA256SUMS.json":
            exclusions.append(relative + " (replaced by archive-specific manifest)")
            continue
        if lower.startswith("external_transportability/") and path.suffix.casefold() == ".npz":
            exclusions.append(relative + " (patient-level external predictions/bootstrap)")
            continue
        add(entries, path, f"output/revision_r1/{relative}")
    return entries, exclusions


def audit_text(entries: dict[str, Path]) -> None:
    text_suffixes = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".yaml", ".yml"}
    for archive_name, source in entries.items():
        if source.suffix.casefold() not in text_suffixes:
            continue
        content = source.read_text(encoding="utf-8", errors="ignore").casefold()
        for token in FORBIDDEN_TEXT_TOKENS:
            if token in content:
                raise ValueError(f"Local user path found in {archive_name}: {token}")


def write() -> None:
    entries, exclusions = collect()
    audit_text(entries)
    manifest = {
        archive_name: hashlib.sha256(source.read_bytes()).hexdigest()
        for archive_name, source in sorted(entries.items())
    }
    manifest_name = f"{PREFIX}/PUBLIC_RELEASE_SHA256SUMS.json"
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_name, source in sorted(entries.items()):
            archive.write(source, archive_name)
        archive.writestr(manifest_name, manifest_bytes)

    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if any(token in name.casefold() for token in FORBIDDEN_ARCHIVE_TOKENS for name in names):
            raise AssertionError("Forbidden external patient-level artifact entered the release archive")
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise AssertionError(f"CRC failure in {bad_crc}")

    report = {
        "archive": str(OUTPUT.resolve()),
        "tag_target": "access-2026-35668-r1",
        "files": len(entries) + 1,
        "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "external_patient_level_data_included": False,
        "excluded": exclusions,
        "privacy_scan": "passed: no user-specific absolute path; no Hungarian workbook; no external prediction/bootstrap NPZ",
        "publication_status": "local release candidate only; public GitHub tag not created",
    }
    AUDIT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    write()
