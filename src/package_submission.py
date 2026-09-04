"""Build the neutral IEEE Access submission and code-result delivery folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "output" / "final_delivery"
RESULTS = ROOT / "output" / "benchmark"

FORBIDDEN_PUBLIC_PARTS = (
    "hungarian myocardial infarction registry",
    "external_transportability_predictions",
    "external_transportability_bootstrap",
    ".ckpt",
    ".pt",
    ".pth",
)


def _is_safe_public(path: Path, archive_name: str) -> bool:
    value = f"{path.name}|{archive_name}".lower()
    if any(token in value for token in FORBIDDEN_PUBLIC_PARTS):
        return False
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        return False
    return True


def _write_zip(path: Path, entries: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [str(source) for source, _ in entries if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing package inputs:\n" + "\n".join(missing))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in sorted(entries, key=lambda item: item[1]):
            if not _is_safe_public(source, name):
                raise ValueError(f"Unsafe archive member: {name}")
            archive.write(source, name)


def _paper_entries(main_tex: str) -> list[tuple[Path, str]]:
    required = {
        "gai_revised_clean.tex",
        main_tex,
        "appendices.tex",
        "ieeeaccess.cls",
        "IEEEtran.cls",
        "IEEEtran.bst",
        "logo.png",
        "bullet.png",
        "notaglinelogo.png",
    }
    entries: list[tuple[Path, str]] = []
    for name in sorted(required):
        source = ROOT / "paper" / name
        entries.append((source, f"paper/{name}"))
    for pattern in ("t1-*", "t1*.fd"):
        for source in sorted((ROOT / "paper").glob(pattern)):
            if source.is_file():
                entries.append((source, f"paper/{source.name}"))
    for source in sorted((ROOT / "paper" / "generated_r1").glob("*.tex")):
        if source.name != "supp_external.tex":
            entries.append((source, f"paper/generated_r1/{source.name}"))
    for source in sorted((ROOT / "paper" / "author_photos").glob("*")):
        if source.is_file():
            entries.append((source, f"paper/author_photos/{source.name}"))
    for source in sorted((ROOT / "figures" / "r1").glob("*.pdf")):
        entries.append((source, f"figures/r1/{source.name}"))
    return entries


def _code_entries() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    roots = ["src", "tests", "configs", "dataset"]
    for folder in roots:
        for source in sorted((ROOT / folder).rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            name = source.relative_to(ROOT).as_posix()
            if _is_safe_public(source, name):
                entries.append((source, name))
    for source in (ROOT / "README.md", ROOT / "requirements.txt"):
        entries.append((source, source.name))
    for source in sorted(RESULTS.rglob("*")):
        if not source.is_file() or any(part in {"logs", "checkpoints"} for part in source.parts):
            continue
        name = source.relative_to(ROOT).as_posix()
        if _is_safe_public(source, name):
            entries.append((source, name))
    return entries


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _materialize(entries: list[tuple[Path, str]], destination: Path) -> None:
    for source, name in entries:
        _copy(source, destination / name)


def _manifest(folder: Path) -> Path:
    rows: dict[str, str] = {}
    for source in sorted(folder.rglob("*")):
        if source.is_file() and source.name != "SHA256SUMS.json":
            rows[source.relative_to(folder).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    path = folder / "SHA256SUMS.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def package(destination: Path) -> Path:
    destination = destination.resolve()
    output_root = (ROOT / "output").resolve()
    try:
        destination.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Package destination must stay within {output_root}") from exc
    if destination == output_root:
        raise ValueError("Refusing to replace the workspace output root")
    if destination.exists():
        shutil.rmtree(destination)
    portal = destination / "01_Portal_Upload"
    editable = destination / "02_Editable_Sources"
    code = destination / "03_Code_and_Results"
    qa = destination / "04_QA_and_Audits"
    for folder in (portal, editable, code, qa):
        folder.mkdir(parents=True, exist_ok=True)

    _copy(ROOT / "revision" / "response_to_reviewers.docx", portal / "Author_Response.docx")
    _copy(ROOT / "output" / "pdf" / "gai_revised_highlighted.pdf", portal / "Highlighted_Manuscript.pdf")
    _copy(ROOT / "paper" / "gai_revised_clean.pdf", portal / "Main_Manuscript.pdf")

    clean_zip = portal / "Clean_LaTeX_Source.zip"
    highlighted_zip = editable / "Highlighted_LaTeX_Source.zip"
    clean_entries = _paper_entries("gai_revised_clean.tex")
    highlighted_entries = _paper_entries("gai_revised_highlighted.tex")
    _write_zip(clean_zip, clean_entries)
    _write_zip(highlighted_zip, highlighted_entries)
    _materialize(clean_entries, editable / "Clean_TeX_Project")
    _materialize(highlighted_entries, editable / "Highlighted_TeX_Project")
    _copy(ROOT / "revision" / "response_to_reviewers.md", editable / "response_to_reviewers.md")
    _copy(ROOT / "revision" / "response_to_reviewers.docx", editable / "response_to_reviewers.docx")

    _write_zip(code / "CoMLC_MI_Code_and_Results.zip", _code_entries())

    audit_candidates = [
        RESULTS / "reference_crossref_audit.csv",
        ROOT / "revision" / "reference_verification_report.md",
        ROOT / "revision" / "resubmission_checklist_completed.docx",
        ROOT / "revision" / "SUBMISSION_READINESS.md",
        ROOT / "output" / "pdf" / "gai_revised_highlighted_audit.json",
        ROOT / "output" / "privacy_audit.json",
        ROOT / "output" / "github_release_audit.json",
        ROOT / "output" / "pdf_visual_qa.md",
        ROOT / "output" / "submission_qa.json",
    ]
    for source in audit_candidates:
        if source.is_file():
            _copy(source, qa / source.name)
    _manifest(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    print(package(args.destination.resolve()))


if __name__ == "__main__":
    main()
