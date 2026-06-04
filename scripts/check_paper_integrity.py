#!/usr/bin/env python3
"""Lightweight integrity checks for the ASAP paper draft.

The check is intentionally narrow: it audits submission-facing paper files,
not experiment logs or internal planning notes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper"

SUBMISSION_FILES = [
    PAPER_DIR / "00_abstract.md",
    PAPER_DIR / "01_introduction.md",
    PAPER_DIR / "02_related_work.md",
    PAPER_DIR / "03_method.md",
    PAPER_DIR / "04_experiments.md",
    PAPER_DIR / "05_conclusion.md",
    PAPER_DIR / "fig_asap_pipeline.tex",
    PAPER_DIR / "tables_icassp.tex",
    PAPER_DIR / "main_icassp.tex",
]

BIB_FILE = PAPER_DIR / "references.bib"

FORBIDDEN = [
    r"<!--",
    r"-->",
    r"\bTODO\b",
    r"\bTBD\b",
    r"placeholder",
    r"UNVERIFIED",
    r"cite-",
    r"figure to be added",
    r"\[XX",
    r"XX\.X",
]


def collect_citations(text: str) -> set[str]:
    """Collect Pandoc-style citation keys from Markdown prose."""
    keys: set[str] = set()
    for bracket in re.findall(r"\[([^\]]*@[^]]+)\]", text):
        for key in re.findall(r"@([A-Za-z0-9_:\-]+)", bracket):
            keys.add(key)
    return keys


def collect_latex_citations(text: str) -> set[str]:
    keys: set[str] = set()
    for body in re.findall(r"\\cite\{([^}]+)\}", text):
        keys.update(k.strip() for k in body.split(",") if k.strip())
    return keys


def collect_bib_keys(text: str) -> set[str]:
    return set(re.findall(r"^\s*@\w+\s*\{\s*([^,\s]+)", text, flags=re.MULTILINE))


def main() -> int:
    failures: list[str] = []

    for path in SUBMISSION_FILES:
        if not path.exists():
            failures.append(f"missing file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            if re.search(pattern, text, flags=re.IGNORECASE):
                failures.append(f"{path.relative_to(ROOT)} matches forbidden pattern: {pattern}")

    if not BIB_FILE.exists():
        failures.append(f"missing file: {BIB_FILE.relative_to(ROOT)}")
    else:
        bib_text = BIB_FILE.read_text(encoding="utf-8")
        for pattern in [r"UNVERIFIED", r"placeholder", r"\bTBD\b"]:
            if re.search(pattern, bib_text, flags=re.IGNORECASE):
                failures.append(f"{BIB_FILE.relative_to(ROOT)} matches forbidden pattern: {pattern}")

        md_text = "\n".join(
            p.read_text(encoding="utf-8")
            for p in SUBMISSION_FILES
            if p.suffix == ".md" and p.exists()
        )
        tex_text = "\n".join(
            p.read_text(encoding="utf-8")
            for p in SUBMISSION_FILES
            if p.suffix == ".tex" and p.exists()
        )
        cite_keys = collect_citations(md_text) | collect_latex_citations(tex_text)
        bib_keys = collect_bib_keys(bib_text)
        missing = sorted(cite_keys - bib_keys)
        if missing:
            failures.append("missing BibTeX keys: " + ", ".join(missing))

    if failures:
        print("Paper integrity check FAILED:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("Paper integrity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
