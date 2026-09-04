"""Compatibility entry point for the neutral code-and-results package."""

from pathlib import Path

from src.package_submission import DESTINATION, _code_entries, _write_zip


def write() -> Path:
    output = DESTINATION / "03_Code_and_Results" / "CoMLC_MI_Code_and_Results.zip"
    _write_zip(output, _code_entries())
    print(output.resolve())
    return output


if __name__ == "__main__":
    write()
