# BenefitExplorer Evaluation

This project-level evaluation suite is intentionally separate from the backend
application. It generates fresh RAG answers and reports exactly three metrics:

1. Faithfulness — official `ragas.metrics.faithfulness`
2. Context Recall@4 — evidence groups covered by the final contexts divided by
   all material evidence groups. Overlapping chunks are alternatives inside a
   group, so duplicate ingestion windows do not make perfect recall impossible.
3. Answer Correctness — official `ragas.metrics.answer_correctness`

Every answer is scored twice. The self-judge is the generation model itself
(`qwen.qwen3-32b`) and the independent judge is `openai.gpt-oss-120b`. Both run
on the same Bedrock Mantle endpoint and credentials configured in
`backend/.env`, so no second provider key is needed. Override the independent
judge with `INDEPENDENT_JUDGE_MODEL` or `--independent-judge-model`; the runner
refuses any judge that is the same id as, or from the same model family as, the
generator.

RAGAS prompts return structured output, so judges are given a 16K output-token
ceiling (`--judge-max-output-tokens`, or `RAGAS_MAX_OUTPUT_TOKENS`).

Structured replies are defended in three layers. Mantle prefills the opening of
the JSON object under `response_format: json_object`, and a judge that then
emits a complete object of its own produces two openings; the evaluator
recovers the intact object before parsing. A reply that is malformed some other
way is re-asked in place by instructor with the validation error attached
(`--judge-repair-attempts`, default 3), then resampled by the outer retry loop.
A reply that still will not parse is recorded as `parse_error` rather than
being silently averaged away.

## Install

Run every command below from the repository root:

```bash
backend/.venv/bin/pip install -r evaluation/requirements.txt
```

`langchain-community` is pinned because RAGAS 0.4.3 imports a compatibility
module removed by the newer 0.4.x LangChain Community releases. Python 3.12 or
3.13 remains the safest choice if a scientific dependency lacks a wheel for a
newer Python release.

## Run

Generation and scoring are separate stages so that one answer set can be
judged repeatedly without being regenerated:

```bash
backend/.venv/bin/python evaluation/run_evaluation.py generate
backend/.venv/bin/python evaluation/run_evaluation.py score
backend/.venv/bin/python evaluation/run_evaluation.py report
```

`all` (the default stage) runs generate then score.

## Checkpoint and resume

The runner atomically checkpoints every question to
`evaluation/results/generation_artifacts.json` and
`evaluation/results/evaluation_results.json`. If a run is interrupted, resume
with:

```bash
backend/.venv/bin/python evaluation/run_evaluation.py score --resume
```

Use the same `--golden` and judge options as the original run; a resume whose
configuration or judge model IDs differ from the checkpoint fails loudly
rather than mixing runs. Do not start a resume process while the original
evaluation is still running. Add `--retry-provider-errors` to re-score
questions whose metrics ended in `provider_error` or `parse_error`.

Outputs are written only to `evaluation/results/` and `evaluation/reports/`.
