"""Hybrid PDF text extraction using PyMuPDF and pdfplumber."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf
import pdfplumber

from .models import ExtractedDocument, ExtractedPage


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be extracted safely."""


def _normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pymupdf_text(page: pymupdf.Page) -> str:
    blocks = page.get_text("blocks", sort=True)
    text_blocks = [str(block[4]).strip() for block in blocks if len(block) > 6 and block[6] == 0]
    return _normalize_text("\n\n".join(block for block in text_blocks if block))


def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(character.isprintable() or character == "\n" for character in text)
    alphanumeric = sum(character.isalnum() for character in text)
    replacement_penalty = text.count("\ufffd") * 20
    return (printable / len(text)) + min(alphanumeric / 500.0, 4.0) - replacement_penalty


def _table_to_markdown(table: list[list[Any]]) -> str:
    rows: list[list[str]] = []
    for row in table:
        cleaned = [_normalize_text(str(cell or "")).replace("\n", " ") for cell in row]
        if any(cleaned):
            rows.append(cleaned)
    if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _recover_tables(page: pdfplumber.page.Page, base_text: str) -> list[str]:
    recovered: list[str] = []
    normalized_base = re.sub(r"\W+", " ", base_text).casefold()
    try:
        tables = page.extract_tables() or []
    except Exception:
        return recovered
    for table in tables:
        meaningful_cells = [
            _normalize_text(str(cell or ""))
            for row in table
            for cell in row
            if len(_normalize_text(str(cell or ""))) >= 3
        ]
        if not meaningful_cells:
            continue
        present_ratio = sum(
            re.sub(r"\W+", " ", cell).casefold() in normalized_base for cell in meaningful_cells
        ) / len(meaningful_cells)
        if present_ratio >= 0.85:
            continue
        markdown = _table_to_markdown(table)
        if markdown:
            recovered.append(markdown)
    return recovered


def _filename_product_name(path: Path) -> str:
    name = re.sub(r"[_\-]+", " ", path.stem)
    name = re.sub(r"(?i)\b(brochure|policy|document|final|web)\b", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or path.stem


def _product_name(path: Path, first_pages: list[str]) -> str:
    fallback = _filename_product_name(path)
    filename_terms = {term.casefold() for term in fallback.split() if len(term) > 2}
    candidates: list[tuple[int, str]] = []
    for text in first_pages[:2]:
        for line in text.splitlines()[:35]:
            line = re.sub(r"\s+", " ", line).strip(" |:-")
            line = re.sub(r"(?i)\s+brochure\b", "", line).strip()
            words = line.split()
            if not 5 <= len(line) <= 70 or len(words) > 8:
                continue
            lowered = line.casefold()
            if not lowered.startswith("kotak ") or re.search(r"[.!?;,]", line):
                continue
            overlap = len(filename_terms.intersection(lowered.split()))
            product_hint = int(
                any(word in lowered for word in ("plan", "protect", "pension", "assured", "gain", "tulip", "edge", "maximiser"))
            )
            noise = int(
                any(word in lowered for word in ("insurance is", "beware", "uin", "www.", "customer", "presents"))
            )
            score = overlap * 4 + product_hint * 2 - noise * 8 - len(words)
            if score >= 2:
                candidates.append((score, line))
    if not candidates:
        return fallback
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][1]


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract a PDF into normalized, one-indexed pages."""

    try:
        pymupdf_document = pymupdf.open(path)
        plumber_document = pdfplumber.open(path)
    except Exception as exc:
        raise PDFExtractionError(f"Could not open {path}: {exc}") from exc

    pages: list[ExtractedPage] = []
    try:
        if len(pymupdf_document) != len(plumber_document.pages):
            raise PDFExtractionError(
                f"Extractor page-count mismatch for {path.name}: "
                f"{len(pymupdf_document)} vs {len(plumber_document.pages)}"
            )
        for index, (fitz_page, plumber_page) in enumerate(
            zip(pymupdf_document, plumber_document.pages, strict=True), start=1
        ):
            fitz_text = _pymupdf_text(fitz_page)
            plumber_text = _normalize_text(
                plumber_page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            )
            page_text = max((fitz_text, plumber_text), key=_text_quality)
            recovered_tables = _recover_tables(plumber_page, page_text)
            if recovered_tables:
                page_text = page_text + "\n\n[Recovered table]\n" + "\n\n".join(recovered_tables)
            pages.append(ExtractedPage(page_number=index, text=page_text))
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(f"Failed while extracting {path}: {exc}") from exc
    finally:
        pymupdf_document.close()
        plumber_document.close()

    product_name = _product_name(path, [page.text for page in pages[:2]])
    return ExtractedDocument(path, product_name, tuple(pages))
