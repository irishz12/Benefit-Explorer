# BenefitExplorer Evaluation

This project-level evaluation suite is intentionally separate from the backend
application. It generates fresh RAG answers and reports exactly three metrics:

1. Faithfulness — official `ragas.metrics.faithfulness`
2. Context Recall@4 — evidence groups covered by the final contexts divided by
   all material evidence groups. Overlapping chunks are alternatives inside a
   group, so duplicate ingestion windows do not make perfect recall impossible.
3. Answer Correctness — official `ragas.metrics.answer_correctness`

The RAGAS judge uses the Bedrock Mantle endpoint and credentials configured in
`backend/.env`. Set `RAGAS_JUDGE_MODEL` there to use a judge model different
from the answer-generation model.

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
