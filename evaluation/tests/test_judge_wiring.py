import os

import instructor

from evaluation.src.ragas_metrics import RagasEvaluator


class _FakeEmbedder:
    def embed_query(self, text):  # noqa: ANN001, ANN201
        return [0.0]

    def embed_documents(self, texts):  # noqa: ANN001, ANN201
        return [[0.0] for _ in texts]


def _build_evaluator():
    # Constructor makes no network calls; fake credentials are sufficient.
    return RagasEvaluator(
        _FakeEmbedder(),
        "openai.gpt-oss-120b",
        "fake-key",
        "https://example.invalid/v1",
        180.0,
    )


def test_judge_uses_instructor_md_json_mode() -> None:
    evaluator = _build_evaluator()
    judge = evaluator.faithfulness_metric.llm
    assert judge.client.mode is instructor.Mode.MD_JSON


def test_judge_model_and_config_wired() -> None:
    evaluator = _build_evaluator()
    faithfulness_judge = evaluator.faithfulness_metric.llm
    correctness_judge = evaluator.answer_correctness_metric.llm

    assert faithfulness_judge.model == "openai.gpt-oss-120b"
    assert correctness_judge.model == "openai.gpt-oss-120b"

    args = dict(faithfulness_judge.model_args)
    assert args["temperature"] == 0.0
    assert args["max_tokens"] == int(os.getenv("RAGAS_MAX_OUTPUT_TOKENS", "1800"))
