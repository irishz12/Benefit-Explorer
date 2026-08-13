"""RAGAS-only Faithfulness and Answer Correctness evaluation."""

from __future__ import annotations

import asyncio
import copy
import os
from dataclasses import dataclass
from typing import Any, Sequence

from openai import OpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import answer_correctness, faithfulness
from ragas.run_config import RunConfig


@dataclass(frozen=True, slots=True)
class RagasScores:
    faithfulness: float | None
    answer_correctness: float | None
    errors: tuple[str, ...] = ()


class BGERagasEmbeddings(BaseRagasEmbeddings):
    """Expose the existing BGE embedder through RAGAS's embedding interface."""

    def __init__(self, embedder: Any, run_config: RunConfig) -> None:
        super().__init__()
        self.embedder = embedder
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> list[float]:
        return list(self.embedder.embed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self.embedder.embed_documents(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


class RagasEvaluator:
    """Run the exact RAGAS metric implementations required by this project."""

    def __init__(
        self,
        embedder: Any,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 180.0,
    ) -> None:
        run_config = RunConfig(timeout=timeout, max_retries=2)
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        judge_llm = llm_factory(
            model,
            provider="openai",
            client=client,
            adapter="instructor",
            temperature=0.0,
            max_tokens=int(os.getenv("RAGAS_MAX_OUTPUT_TOKENS", "1800")),
        )
        ragas_embeddings = BGERagasEmbeddings(embedder, run_config)

        # Copy the official metric singletons so evaluation state is isolated.
        self.faithfulness_metric = copy.deepcopy(faithfulness)
        self.answer_correctness_metric = copy.deepcopy(answer_correctness)
        self.faithfulness_metric.llm = judge_llm
        self.answer_correctness_metric.llm = judge_llm
        self.answer_correctness_metric.embeddings = ragas_embeddings
        self.faithfulness_metric.init(run_config)
        self.answer_correctness_metric.init(run_config)
        self.timeout = timeout

    async def score(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        contexts: Sequence[str],
    ) -> RagasScores:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=reference_answer,
            retrieved_contexts=list(contexts),
        )
        errors: list[str] = []
        try:
            faithfulness_score = float(
                await self.faithfulness_metric.single_turn_ascore(
                    sample,
                    timeout=self.timeout,
                )
            )
        except Exception as error:
            faithfulness_score = None
            errors.append(f"faithfulness: {type(error).__name__}: {error}")
        try:
            correctness_score = float(
                await self.answer_correctness_metric.single_turn_ascore(
                    sample,
                    timeout=self.timeout,
                )
            )
        except Exception as error:
            correctness_score = None
            errors.append(f"answer_correctness: {type(error).__name__}: {error}")
        return RagasScores(faithfulness_score, correctness_score, tuple(errors))
