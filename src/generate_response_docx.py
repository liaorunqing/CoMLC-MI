"""Generate the submission-ready IEEE Access response letter from Markdown."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "revision" / "response_to_reviewers.md"
OUTPUT = ROOT / "revision" / "response_to_reviewers.docx"

BLUE = "1F4E79"
PALE_BLUE = "D9EAF7"
PALE_YELLOW = "FFF2CC"
GRAY = "666666"


def set_cellless_paragraph_shading(paragraph, fill: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shd = p_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        p_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Page ")
    run.font.size = Pt(9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def add_inline(paragraph, text: str) -> None:
    """Render the small Markdown subset used by the response source."""
    token_re = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")
    pos = 0
    for match in token_re.finditer(text):
        if match.start() > pos:
            paragraph.add_run(text[pos : match.start()])
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        else:
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        pos = match.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.70)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    section.header_distance = Inches(0.30)
    section.footer_distance = Inches(0.30)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08

    for name, size, color in [
        ("Title", 18, BLUE),
        ("Heading 1", 14, BLUE),
        ("Heading 2", 11, BLUE),
    ]:
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        style.paragraph_format.keep_with_next = True

    doc.styles["Heading 1"].paragraph_format.space_before = Pt(4)
    doc.styles["Heading 1"].paragraph_format.space_after = Pt(8)
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(8)
    doc.styles["Heading 2"].paragraph_format.space_after = Pt(4)

    # Keep the closing with Reviewer 2 instead of creating a nearly empty
    # signature page after the compact reproducibility-action list.
    list_bullet = doc.styles["List Bullet"]
    list_bullet.font.name = "Times New Roman"
    list_bullet.font.size = Pt(10)
    list_bullet._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    list_bullet.paragraph_format.space_after = Pt(1)
    list_bullet.paragraph_format.line_spacing = 1.0

    if "Reviewer block" not in doc.styles:
        style = doc.styles.add_style("Reviewer block", WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = doc.styles["Normal"]
        style.paragraph_format.left_indent = Inches(0.12)
        style.paragraph_format.right_indent = Inches(0.06)
        style.paragraph_format.space_after = Pt(5)

    header = section.header.paragraphs[0]
    header.text = "IEEE Access — Response to Reviewers | Access-2026-35668"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.name = "Arial"
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor.from_string(GRAY)
    add_page_number(section.footer.paragraphs[0])


def generate() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    doc = Document()
    configure_document(doc)

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("# "):
            paragraph = doc.add_paragraph(style="Title")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(paragraph, line[2:])
            subtitle = doc.add_paragraph()
            subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = subtitle.add_run("Original submission and revised-manuscript correspondence")
            run.italic = True
            run.font.color.rgb = RGBColor.from_string(GRAY)
            continue

        if line.startswith("## "):
            heading = line[3:]
            if heading.startswith("Reviewer"):
                doc.add_page_break()
            paragraph = doc.add_paragraph(style="Heading 1")
            add_inline(paragraph, heading)
            set_cellless_paragraph_shading(paragraph, PALE_BLUE)
            continue

        if line.startswith("### "):
            paragraph = doc.add_paragraph(style="Heading 2")
            add_inline(paragraph, line[4:])
            continue

        if line.startswith("- "):
            paragraph = doc.add_paragraph(style="List Bullet")
            add_inline(paragraph, line[2:])
            continue

        paragraph = doc.add_paragraph(style="Reviewer block" if line.startswith("**") else "Normal")
        add_inline(paragraph, line)
        if line.startswith("**Concern."):
            set_cellless_paragraph_shading(paragraph, "F2F2F2")
        elif line.startswith("**Response."):
            set_cellless_paragraph_shading(paragraph, "EAF2F8")
        elif line.startswith("**Action."):
            set_cellless_paragraph_shading(paragraph, PALE_YELLOW)
        if line.startswith("**Manuscript ID:") or line.startswith("**Original title:") or line.startswith("**Revised title:"):
            paragraph.paragraph_format.space_after = Pt(2)

    closing = doc.add_paragraph()
    closing.paragraph_format.space_before = Pt(6)
    closing.add_run("Sincerely,\nThe Authors")

    # Avoid widows/orphans and preserve adjacent comment blocks where possible.
    for paragraph in doc.paragraphs:
        p_pr = paragraph._p.get_or_add_pPr()
        if p_pr.find(qn("w:widowControl")) is None:
            p_pr.append(OxmlElement("w:widowControl"))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    generate()
