"""Command-line entry point for BenefitExplorer evaluation."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.src.runner import build_parser, run_evaluation


def main() -> None:
    asyncio.run(run_evaluation(build_parser().parse_args()))


if __name__ == "__main__":
    main()
