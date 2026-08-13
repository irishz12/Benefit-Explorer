"""Heuristic section and clause detection for insurance brochures."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import ExtractedPage, SectionSpan


SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Exclusions", re.compile(r"\b(exclusions?|what (?:is|are) not covered|not covered)\b", re.I)),
    ("Waiting Period", re.compile(r"\b(waiting periods?|survival periods?)\b", re.I)),
    ("Eligibility", re.compile(r"\b(eligibility|entry age|maturity age|who can (?:buy|apply)|age at entry)\b", re.I)),
    ("Premiums", re.compile(r"\b(premiums?|premium payment|payment frequency|modal premium)\b", re.I)),
    ("Benefits", re.compile(r"\b(benefits?|death benefit|maturity benefit|survival benefit|rider benefit)\b", re.I)),
    ("Policy Terms", re.compile(r"\b(policy terms?|terms and conditions|policy tenure|policy term)\b", re.I)),
    ("Claims", re.compile(r"\b(claims?|claim procedure|how to claim|claim settlement)\b", re.I)),
    ("Surrender", re.compile(r"\b(surrender|discontinuance|foreclosure)\b", re.I)),
    ("Charges", re.compile(r"\b(charges?|fees?|deductions?)\b", re.I)),
    ("Tax Benefits", re.compile(r"\b(tax benefits?|income tax|taxation)\b", re.I)),
    ("Definitions", re.compile(r"\b(definitions?|meaning of terms)\b", re.I)),
    ("Features", re.compile(r"\b(key features?|product features?|plan at a glance|highlights?)\b", re.I)),
    ("Riders", re.compile(r"\b(optional riders?|rider options?|add[- ]on covers?)\b", re.I)),
    ("Grievance", re.compile(r"\b(grievance|complaints?|ombudsman)\b", re.I)),
)


def classify_section(line: str) -> str | None:
    """Classify a probable heading into a normalized insurance section."""

    cleaned = re.sub(r"^[\d.()\-–—\s]+", "", line).strip(" :.-")
    if not cleaned or len(cleaned) > 140:
        return None
    for section_type, pattern in SECTION_PATTERNS:
        if pattern.search(cleaned):
            return section_type
    return None


def _is_heading(line: str, next_line: str = "") -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 140:
        return False
    word_count = len(stripped.split())
    numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+", stripped))
    typographic = stripped.isupper() and 1 <= word_count <= 14
    known = classify_section(stripped) is not None
    colon_heading = stripped.endswith(":") and word_count <= 12
    question_heading = stripped.endswith("?") and word_count <= 12
    compact_heading = (
        word_count <= 8
        and re.search(r"[.;,]", stripped) is None
        and re.search(r"\b(?:shall|will|would|provides?|helps?|allows?|called)\b", stripped, re.I) is None
    )
    return known and (
        typographic
        or colon_heading
        or question_heading
        or compact_heading
        or (numbered and word_count <= 12 and bool(next_line))
    )


def detect_sections(pages: Iterable[ExtractedPage]) -> list[SectionSpan]:
    """Split page text at section headings while carrying section context across pages."""

    spans: list[SectionSpan] = []
    current_section = "General"
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines()]
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                spans.append(SectionSpan(page.page_number, current_section, text))
            buffer.clear()

        for index, line in enumerate(lines):
            if not line:
                if buffer and buffer[-1] != "":
                    buffer.append("")
                continue
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            detected = classify_section(line) if _is_heading(line, next_line) else None
            if detected is not None:
                flush()
                current_section = detected
            buffer.append(line)
        flush()
    return spans
