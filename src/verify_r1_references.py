"""Crossref-first metadata audit for the manuscript bibliography.

This script is intentionally read-only with respect to the manuscript.  It
writes a machine-readable report that can be reviewed before any citation is
changed.
"""

from __future__ import annotations

import argparse
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests


def _plain(value: str) -> str:
    value = value.replace("\\_", "_").replace("--", "-")
    value = re.sub(r"\\emph\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    value = value.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _year(message: dict) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = message.get(key, {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0] is not None:
            return int(parts[0][0])
    return None


def parse_bibliography(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    records = []
    for match in re.finditer(
        r"\\bibitem\{(?P<key>[^}]+)\}\s*(?P<body>.*?)(?=\n\s*\\bibitem|\n\s*\\end\{thebibliography\})",
        text,
        re.S,
    ):
        body = " ".join(match.group("body").split())
        title_match = re.search(r"``(.*?)''", body)
        href_doi = re.search(r"\\href\{https?://doi\.org/([^}]+)\}", body, re.I)
        text_doi = re.search(r"doi:\s*([^\s]+)", body, re.I)
        doi_match = href_doi or text_doi
        # Exclude the DOI because its suffix frequently contains an earlier
        # registration year that is not the bibliographic publication year.
        bibliographic_body = body[: doi_match.start()] if doi_match else body
        year_matches = re.findall(r"\b(?:19|20)\d{2}\b", bibliographic_body)
        doi = doi_match.group(1).rstrip(".,}") if doi_match else None
        if doi:
            doi = doi.replace("\\_", "_")
            doi = doi.replace("\\%", "%")
            doi = doi.replace("$<$", "<").replace("$>$", ">")
            doi = unquote(doi)
        records.append(
            {
                "key": match.group("key"),
                "manuscript_title": title_match.group(1) if title_match else None,
                "manuscript_year": int(year_matches[-1]) if year_matches else None,
                "doi": doi,
            }
        )
    return records


def audit(record: dict, session: requests.Session) -> dict:
    row = dict(record)
    row.update(
        {
            "crossref_status": "no DOI",
            "crossref_title": None,
            "crossref_year": None,
            "crossref_first_author": None,
            "doi_resolves": None,
            "title_similarity": None,
            "year_match": None,
        }
    )
    if not record["doi"]:
        return row
    try:
        resolved = session.get(
            f"https://doi.org/{record['doi']}",
            timeout=30,
            allow_redirects=True,
            stream=True,
        )
        row["doi_resolves"] = resolved.status_code < 400
        resolved.close()
    except requests.RequestException:
        row["doi_resolves"] = False
    url = f"https://api.crossref.org/works/{requests.utils.quote(record['doi'], safe='')}"
    try:
        response = session.get(url, timeout=30)
        row["crossref_status"] = str(response.status_code)
        if response.status_code != 200:
            return row
        message = response.json()["message"]
        title = (message.get("title") or [None])[0]
        authors = message.get("author") or []
        row["crossref_title"] = title
        row["crossref_year"] = _year(message)
        row["crossref_first_author"] = authors[0].get("family") if authors else None
        if record["manuscript_title"] and title:
            manuscript_title = _plain(record["manuscript_title"])
            crossref_title = _plain(title)
            if min(len(manuscript_title), len(crossref_title)) >= 5 and (
                manuscript_title in crossref_title or crossref_title in manuscript_title
            ):
                row["title_similarity"] = 1.0
            else:
                row["title_similarity"] = SequenceMatcher(
                    None, manuscript_title, crossref_title
                ).ratio()
        if record["manuscript_year"] and row["crossref_year"]:
            row["year_match"] = record["manuscript_year"] == row["crossref_year"]
    except requests.RequestException as exc:
        row["crossref_status"] = f"request error: {type(exc).__name__}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default="paper/gai_revised_clean.tex")
    parser.add_argument("--output", default="output/benchmark/reference_crossref_audit.csv")
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "CoMLC-MI-reference-audit/1.0 (mailto:corresponding-author@example.invalid)"
    rows = []
    for record in parse_bibliography(Path(args.tex)):
        rows.append(audit(record, session))
        time.sleep(0.05)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    has_titles = frame["manuscript_title"].notna() & frame["crossref_title"].notna()
    critical = frame[
        frame["doi"].notna()
        & (
            ((frame["crossref_status"] != "200") & ~frame["doi_resolves"].eq(True))
            | (has_titles & (frame["title_similarity"].fillna(0) < 0.80))
        )
    ]
    print(f"Audited {len(frame)} references; {len(critical)} require manual follow-up.")
    if len(critical):
        print(critical[["key", "doi", "crossref_status", "title_similarity", "crossref_title"]].to_string(index=False))


if __name__ == "__main__":
    main()
