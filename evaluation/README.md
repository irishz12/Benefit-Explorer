# BenefitExplorer Evaluation

This project-level evaluation suite is intentionally separate from the backend
application. It generates fresh RAG answers and reports exactly three metrics:

1. Faithfulness — official `ragas.metrics.faithfulness`
2. Context Recall@4 — evidence groups covered by the final contexts divided by
   all material evidence groups. Overlapping chunks are alternatives inside a
   group, so duplicate ingestion windows do not make perfect recall impossible.
3. Answer Correctness — official `ragas.metrics.answer_correctness`

Faithfulness and Answer Correctness use RAGAS; Context Recall@4 is computed
against the manually labelled evidence groups. The RAGAS judge uses the Bedrock
Mantle endpoint and credentials configured in `backend/.env`. Set
`RAGAS_JUDGE_MODEL` there to use a judge model different from the
answer-generation model.

## Final results

The benchmark contains 30 insurance QA questions across six products. The final
run used the judge model `openai.gpt-oss-120b`.

| Metric | Score |
|---|---:|
| Faithfulness | 0.848 |
| Context Recall@4 | 0.878 |
| Answer Correctness | 0.748 |

Coverage: 26 of 30 questions evaluated, 4 pipeline failures, 5 questions with a
RAGAS judge error. Each metric is averaged only over the questions where it was
produced; failed and judge-error questions are not scored as zero.

The golden evidence labels were re-mapped to the current active chunk boundaries
before this run. The previously published scores used an older chunking/index
snapshot, so the two are not a strict apples-to-apples comparison.

## Install

```bash
cd /Users/irishe/Documents/ChatGPT/RAG/Insurance-Product-RAG
backend/.venv/bin/pip install -r evaluation/requirements.txt
```

`langchain-community` is pinned because RAGAS 0.4.3 imports a compatibility
module removed by the newer 0.4.x LangChain Community releases. Python 3.12 or
3.13 remains the safest choice if a scientific dependency lacks a wheel for a
newer Python release.

## Run

```bash
cd /Users/irishe/Documents/ChatGPT/RAG/Insurance-Product-RAG
backend/.venv/bin/python evaluation/run_evaluation.py
```

Quick test:

```bash
backend/.venv/bin/python evaluation/run_evaluation.py --limit 3
```

## Checkpoint and resume

The runner atomically checkpoints every successful question to
`evaluation/results/evaluation_results.json`. If a run is interrupted, resume
from the next unprocessed question with:

```bash
backend/.venv/bin/python evaluation/run_evaluation.py --resume
```

Use the same `--limit`, `--golden`, and judge options as the original run. Do
not start a resume process while the original evaluation is still running.
Failed rows and incomplete RAGAS scores are retried automatically. Condensed
historical summary files are ignored as checkpoints instead of causing resume
to fail.

Outputs are written only to `evaluation/results/` and `evaluation/reports/`.
