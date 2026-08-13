from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.generation.citation_utils import VerifiedAnswer, VerifiedCitation
from src.generation.generator import QARun
from src.main import app


class _Pipeline:
    def answer(self, message: str) -> QARun:
        assert message == "What does the policy pay?"
        return QARun(
            response=VerifiedAnswer(
                answer="The policy pays the maturity benefit [1].",
                citations=(
                    VerifiedCitation(
                        index=1,
                        chunk_id="chunk_benefit",
                        product="Kotak EDGE",
                        page=4,
                        supporting_text="The maturity benefit shall be paid.",
                    ),
                ),
            ),
            final_contexts=(),
            detected_products=("Kotak EDGE",),
            product_retrieval_mode="hard_filter",
        )


def test_streaming_api_returns_answer_and_verified_citations(monkeypatch) -> None:
    monkeypatch.setattr("src.main.build_pipeline", lambda: _Pipeline())

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "What does the policy pay?"},
        )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[0] == {
        "event": "status",
        "message": "Searching product brochures…",
    }
    result = next(event for event in events if event["event"] == "result")
    assert result["answer"] == "The policy pays the maturity benefit [1]."
    assert result["detected_products"] == ["Kotak EDGE"]
    assert result["retrieval_mode"] == "hard_filter"
    assert result["citations"][0]["chunk_id"] == "chunk_benefit"


def test_streaming_api_rejects_blank_questions() -> None:
    with TestClient(app) as client:
        response = client.post("/api/chat/stream", json={"message": ""})

    assert response.status_code == 422
