"""Rate-limit-aware RAGAS judging through OpenAI-compatible providers."""

from __future__ import annotations

import asyncio
import copy
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Sequence

from openai import OpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import answer_correctness, faithfulness
from ragas.run_config import RunConfig


@dataclass(frozen=True, slots=True)
class MetricOutcome:
    value: float | None
    status: str
    error: str | None = None
    attempts: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RagasScores:
    faithfulness: MetricOutcome
    answer_correctness: MetricOutcome

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            "faithfulness": self.faithfulness.to_dict(),
            "answer_correctness": self.answer_correctness.to_dict(),
        }


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


def _provider_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    message = str(error).casefold()
    direct = status_code == 429 or (isinstance(status_code, int) and status_code >= 500) or (
        error.__class__.__name__
        in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
            "ServiceUnavailableError",
        }
    )
    if direct or "rate limit" in message or "status code: 429" in message:
        return True
    nested = error.__cause__ or error.__context__
    return isinstance(nested, Exception) and nested is not error and _provider_error(nested)


class RagasEvaluator:
    """Run official RAGAS metrics with bounded exponential provider retries."""

    def __init__(
        self,
        embedder: Any,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 180.0,
        max_attempts: int = 4,
        backoff_seconds: float = 2.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        self.model_id = model
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        # RAGAS retries are disabled here so one explicit retry policy controls
        # provider pressure and makes the number of attempts auditable.
        run_config = RunConfig(timeout=timeout, max_retries=0)
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

        self.faithfulness_metric = copy.deepcopy(faithfulness)
        self.answer_correctness_metric = copy.deepcopy(answer_correctness)
        self.faithfulness_metric.llm = judge_llm
        self.answer_correctness_metric.llm = judge_llm
        self.answer_correctness_metric.embeddings = ragas_embeddings
        self.faithfulness_metric.init(run_config)
        self.answer_correctness_metric.init(run_config)
        self.timeout = timeout

    async def _score_metric(
        self,
        scorer: Callable[[], Awaitable[float]],
    ) -> MetricOutcome:
        for attempt in range(1, self.max_attempts + 1):
            try:
                value = float(await scorer())
                if not math.isfinite(value):
                    raise ValueError(f"RAGAS returned a non-finite score: {value}")
                return MetricOutcome(value, "ok", attempts=attempt)
            except Exception as error:
                provider_error = _provider_error(error)
                if provider_error and attempt < self.max_attempts:
                    await asyncio.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
                status = "provider_error" if provider_error else "metric_error"
                return MetricOutcome(
                    None,
                    status,
                    f"{type(error).__name__}: {error}",
                    attempt,
                )
        raise AssertionError("unreachable")

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
        faithfulness_outcome = await self._score_metric(
            lambda: self.faithfulness_metric.single_turn_ascore(
                sample,
                timeout=self.timeout,
            )
        )
        correctness_outcome = await self._score_metric(
            lambda: self.answer_correctness_metric.single_turn_ascore(
                sample,
                timeout=self.timeout,
            )
        )
        return RagasScores(faithfulness_outcome, correctness_outcome)
