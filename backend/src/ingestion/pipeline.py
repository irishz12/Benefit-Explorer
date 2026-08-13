"""End-to-end orchestration for brochure ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .chunker import ClauseAwareChunker
from .models import Chunk
from .pdf_extractor import extract_pdf
from .section_detector import detect_sections
from .tokenizer import TokenCounter, build_token_counter

LOGGER = logging.getLogger(__name__)


class IngestionPipeline:
    """Extract, section, chunk, enrich, and serialize insurance PDFs."""

    def __init__(
        self,
        min_tokens: int = 400,
        target_tokens: int = 550,
        max_tokens: int = 700,
        overlap_ratio: float = 0.20,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.token_counter = token_counter or build_token_counter()
        self.chunker = ClauseAwareChunker(
            self.token_counter,
            min_tokens=min_tokens,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_ratio=overlap_ratio,
        )

    @staticmethod
    def _stable_id(
        source_file: str,
        page_numbers: tuple[int, ...],
        section_type: str,
        ordinal: int,
        text: str,
    ) -> str:
        normalized_text = " ".join(text.split())
        pages = ",".join(str(page_number) for page_number in page_numbers)
        identity = f"{source_file}|{pages}|{section_type}|{ordinal}|{normalized_text}"
        return "chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    def process_pdf(self, pdf_path: Path) -> list[Chunk]:
        """Process one PDF and return retrieval-ready chunks."""

        LOGGER.info("Extracting %s", pdf_path.name)
        document = extract_pdf(pdf_path)
        sections = detect_sections(document.pages)
        drafts = self.chunker.chunk_sections(sections)
        chunks: list[Chunk] = []
        for ordinal, draft in enumerate(drafts, start=1):
            source_file = pdf_path.name
            chunks.append(
                Chunk(
                    chunk_id=self._stable_id(
                        source_file,
                        draft.page_numbers,
                        draft.section_type,
                        ordinal,
                        draft.text,
                    ),
                    product_name=document.product_name,
                    page_number=draft.page_numbers[0],
                    page_numbers=draft.page_numbers,
                    section_type=draft.section_type,
                    section_types=draft.section_types,
                    source_file=source_file,
                    text=draft.text,
                )
            )
        LOGGER.info("Created %d chunks from %s", len(chunks), pdf_path.name)
        return chunks

    def run(self, input_dir: Path, output_path: Path, recursive: bool = True) -> dict[str, Any]:
        """Process every PDF in ``input_dir`` and write a JSON array."""

        input_dir = input_dir.resolve()
        output_path = output_path.resolve()
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_paths = sorted(
            (path for path in input_dir.glob(pattern) if path.is_file()),
            key=lambda path: str(path).casefold(),
        )
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {input_dir}")

        chunks: list[Chunk] = []
        for pdf_path in pdf_paths:
            chunks.extend(self.process_pdf(pdf_path))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump([chunk.to_dict() for chunk in chunks], output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")
        temporary_path.replace(output_path)

        section_counts = Counter(chunk.section_type for chunk in chunks)
        return {
            "pdf_count": len(pdf_paths),
            "chunk_count": len(chunks),
            "output_path": str(output_path),
            "sections": dict(sorted(section_counts.items())),
        }
