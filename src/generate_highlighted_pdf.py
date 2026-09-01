"""Create a yellow-highlighted revision PDF against the original submission.

The comparison is performed on word tokens extracted from the original and
clean revised PDFs. Revised tokens classified as inserted or replaced are
highlighted; deleted text is not reproduced. Replaced/new figure captions and
new biography photographs also receive a visible yellow outline. The clean PDF
itself remains unchanged.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = ROOT / "paper" / "gai_rewrite.pdf"
DEFAULT_REVISED = ROOT / "output" / "pdf" / "gai_revised_clean.pdf"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "gai_revised_highlighted.pdf"
DEFAULT_AUDIT = ROOT / "output" / "pdf" / "gai_revised_highlighted_audit.json"
FIGURES_TO_MARK = {1, 3, 4, 5, 6, 7}


@dataclass(frozen=True)
class PdfWord:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block: int
    line: int
    ordinal: int


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    value = re.sub(r"\s+", "", value)
    return value


def extract_words(document: fitz.Document) -> list[PdfWord]:
    words: list[PdfWord] = []
    for page_index, page in enumerate(document):
        for ordinal, item in enumerate(page.get_text("words", sort=True)):
            x0, y0, x1, y1, text, block, line, _ = item
            token = normalize(text)
            if not token:
                continue
            # Ignore running headers and page numbers; these change through reflow.
            if y0 < 42 or y1 > page.rect.height - 42:
                continue
            words.append(PdfWord(page_index, x0, y0, x1, y1, text, block, line, ordinal))
    return words


def changed_revised_indices(original: list[PdfWord], revised: list[PdfWord]) -> set[int]:
    matcher = difflib.SequenceMatcher(
        None,
        [normalize(word.text) for word in original],
        [normalize(word.text) for word in revised],
        autojunk=False,
    )
    changed: set[int] = set()
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "insert"}:
            changed.update(range(j1, j2))
    return changed


def group_line_rectangles(words: list[PdfWord], changed: set[int]) -> dict[int, list[fitz.Rect]]:
    grouped: dict[tuple[int, int, int], list[tuple[int, PdfWord]]] = defaultdict(list)
    for index in sorted(changed):
        word = words[index]
        grouped[(word.page, word.block, word.line)].append((index, word))

    by_page: dict[int, list[fitz.Rect]] = defaultdict(list)
    for (page_index, _, _), line_words in grouped.items():
        line_words.sort(key=lambda item: item[1].ordinal)
        run: list[PdfWord] = []
        previous_ordinal: int | None = None
        for _, word in line_words:
            if previous_ordinal is not None and word.ordinal > previous_ordinal + 1:
                by_page[page_index].append(union_rect(run))
                run = []
            run.append(word)
            previous_ordinal = word.ordinal
        if run:
            by_page[page_index].append(union_rect(run))
    return by_page


def union_rect(words: list[PdfWord]) -> fitz.Rect:
    rect = fitz.Rect(words[0].x0, words[0].y0, words[0].x1, words[0].y1)
    for word in words[1:]:
        rect.include_rect(fitz.Rect(word.x0, word.y0, word.x1, word.y1))
    rect.x0 -= 0.7
    rect.x1 += 0.7
    rect.y0 -= 0.5
    rect.y1 += 0.5
    return rect


def mark_figure_captions(document: fitz.Document) -> list[dict[str, object]]:
    marked: list[dict[str, object]] = []
    pattern = re.compile(r"FIGURE\s+(\d+)\.", re.IGNORECASE)
    for page_index, page in enumerate(document):
        blocks = page.get_text("blocks", sort=True)
        for x0, y0, x1, y1, text, *_ in blocks:
            matches = pattern.findall(text)
            if not matches:
                continue
            figure_numbers = {int(number) for number in matches}
            if not figure_numbers.intersection(FIGURES_TO_MARK):
                continue
            rect = fitz.Rect(max(24, x0 - 3), max(24, y0 - 3), min(page.rect.width - 24, x1 + 3), min(page.rect.height - 24, y1 + 3))
            page.draw_rect(rect, color=(1.0, 0.72, 0.0), width=1.6, overlay=True)
            marked.append({"page": page_index + 1, "figures": sorted(figure_numbers), "rect": list(rect)})
    return marked


def mark_biography_photos(document: fitz.Document) -> list[dict[str, object]]:
    """Outline author photographs while excluding page logos and figure assets."""
    marked: list[dict[str, object]] = []
    author_names = ("RUNQING LIAO", "XIAOPENG YANG", "ZHINING WANG")
    for page_index, page in enumerate(document):
        page_text = page.get_text().upper()
        if not any(name in page_text for name in author_names):
            continue
        for info in page.get_image_info():
            rect = fitz.Rect(info["bbox"])
            if rect.y0 < 50 or rect.width < 45 or rect.height < 60:
                continue
            outline = fitz.Rect(rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2)
            page.draw_rect(outline, color=(1.0, 0.72, 0.0), width=1.6, overlay=True)
            marked.append({"page": page_index + 1, "rect": list(outline)})
    return marked


def generate(original_path: Path, revised_path: Path, output_path: Path, audit_path: Path) -> None:
    original = fitz.open(original_path)
    revised = fitz.open(revised_path)
    original_words = extract_words(original)
    revised_words = extract_words(revised)
    changed = changed_revised_indices(original_words, revised_words)
    rectangles = group_line_rectangles(revised_words, changed)

    page_counts: dict[int, int] = {}
    for page_index, page_rectangles in rectangles.items():
        page = revised[page_index]
        for rect in page_rectangles:
            annotation = page.add_highlight_annot(rect)
            annotation.set_colors(stroke=(1.0, 0.86, 0.0))
            annotation.set_opacity(0.38)
            annotation.update()
        page_counts[page_index + 1] = len(page_rectangles)

    figure_marks = mark_figure_captions(revised)
    biography_photo_marks = mark_biography_photos(revised)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    revised.save(output_path, garbage=4, deflate=True, clean=True)
    original.close()
    revised.close()

    audit = {
        "comparison_baseline": str(original_path),
        "revised_clean_pdf": str(revised_path),
        "highlighted_pdf": str(output_path),
        "original_tokens": len(original_words),
        "revised_tokens": len(revised_words),
        "highlighted_revised_tokens": len(changed),
        "highlighted_token_fraction": len(changed) / len(revised_words),
        "highlight_rectangles_by_page": page_counts,
        "figure_caption_outlines": figure_marks,
        "biography_photo_outlines": biography_photo_marks,
        "note": "Yellow highlights mark revised words classified as inserted/replaced by global token-sequence comparison; yellow outlines mark revised figures and newly inserted biography photographs; deleted text is omitted.",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--revised", type=Path, default=DEFAULT_REVISED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    generate(args.original.resolve(), args.revised.resolve(), args.output.resolve(), args.audit.resolve())


if __name__ == "__main__":
    main()
