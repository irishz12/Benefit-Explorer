"""Command-line entry point for PDF ingestion.

Run from ``backend`` with::

    python -m src.ingestion.run_ingestion
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .pipeline import IngestionPipeline

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = BACKEND_DIR / "data"


def build_parser() -> argparse.ArgumentParser:
    """Build the ingestion command-line parser."""

    parser = argparse.ArgumentParser(description="Extract and chunk insurance brochure PDFs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing PDFs (default: backend/data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKEND_DIR / "data" / "processed_chunks.json",
        help="Destination JSON file (default: backend/data/processed_chunks.json).",
    )
    parser.add_argument("--min-tokens", type=int, default=400)
    parser.add_argument("--target-tokens", type=int, default=550)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--overlap", type=float, default=0.20, help="Overlap ratio (0.15-0.25).")
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """Run ingestion using command-line arguments."""

    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    pipeline = IngestionPipeline(
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap,
    )
    input_dir = args.input_dir or DEFAULT_INPUT_DIR
    summary = pipeline.run(input_dir, args.output, recursive=not args.no_recursive)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
