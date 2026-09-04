"""Apply verified DOI/proceedings links to the manuscript bibliography."""

from __future__ import annotations

import re
from pathlib import Path


DOIS = {
    "elashmawi2024": "10.1109/MIUCC62295.2024.10783524",
    "makhmudov2025": "10.3390/bioengineering12050520",
    "christodoulou2019": "10.1016/j.jclinepi.2019.02.004",
    "futoma2020": "10.1016/S2589-7500(20)30186-2",
    "chen2019mlgcn": "10.1109/CVPR.2019.00532",
    "wang2020": "10.1609/aaai.v34i07.6909",
    "vancalster2019": "10.1186/s12916-019-1466-7",
    "zhang2014": "10.1109/TKDE.2013.39",
    "gibaja2015": "10.1145/2716262",
    "tsoumakas2007b": "10.1007/978-0-387-09823-4_34",
    "read2011": "10.1007/s10994-011-5256-5",
    "tsoumakas2007": "10.1007/978-3-540-74958-5_38",
    "jia2025": "10.3390/make7030092",
    "hollmann2025": "10.1038/s41586-024-08328-6",
    "tabpfn3report": "10.48550/arXiv.2605.13986",
    "uci2020": "10.24432/C53P5M",
    "almurshedi2026data": "10.17632/2v7n2r3xch.1",
    "balint2026": "10.3390/ijerph23081010",
    "stekhoven2012": "10.1093/bioinformatics/btr597",
    "sechidis2011": "10.1007/978-3-642-23808-6_10",
    "charte2015": "10.1016/j.knosys.2015.07.019",
    "chen2016": "10.1145/2939672.2939785",
    "ridnik2021": "10.1109/ICCV48922.2021.00015",
    "delong1988": "10.2307/2531595",
    "good2000": "10.1007/978-1-4757-3235-1",
    "benjamini1995": "10.1111/j.2517-6161.1995.tb02031.x",
    "efron1994": "10.1201/9780429246593",
    "brier1950": "10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2",
}

PROCEEDINGS = {
    "ke2017": "https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    "prokhorenkova2018": "https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html",
}


def _doi_link(doi: str) -> str:
    url = doi.replace("<", r"\%3C").replace(">", r"\%3E")
    return rf"\href{{https://doi.org/{url}}}{{\nolinkurl{{https://doi.org/{doi}}}}}"


def run(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "\\bibitem{tabpfn3report}" not in text:
        marker = "\\bibitem{uci2020}"
        entry = (
            "\\bibitem{tabpfn3report}\n"
            r"L.~Grinsztajn \emph{et al.}, ``TabPFN-3: Technical report,'' arXiv:2605.13986, 2026." "\n\n"
        )
        text = text.replace(marker, entry + marker)

    pattern = re.compile(r"(\\bibitem\{([^}]+)\}\n)(.*?)(?=\n\n\\bibitem|\n\n\\end\{thebibliography\})", re.S)

    def update(match: re.Match[str]) -> str:
        prefix, key, body = match.group(1), match.group(2), match.group(3).strip()
        body = re.sub(r", doi:.*(?=\.$)", "", body, flags=re.S)
        body = re.sub(r", \\href\{https://doi\.org/.*?(?=\.$)", "", body, flags=re.S)
        if body.endswith("."):
            body = body[:-1]
        if key in DOIS:
            body += ", " + _doi_link(DOIS[key])
        elif key in PROCEEDINGS:
            body += ", " + rf"\href{{{PROCEEDINGS[key]}}}{{official NeurIPS proceedings page}}"
        return prefix + body + "."

    updated = pattern.sub(update, text)
    path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    run(Path("paper/gai_revised_clean.tex"))
