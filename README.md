# BenefitExplorer

> Insurance answers grounded in product brochures.

BenefitExplorer is a full-stack retrieval-augmented generation (RAG) system
that answers factual and comparative questions about insurance products. It
extracts evidence from PDF brochures, combines dense and lexical retrieval,
reranks the strongest passages, generates a grounded response, and verifies
inline citations before displaying the answer and sources.

This repository is a portfolio and research project. It is not financial,
insurance, or legal advice. Product terms must always be verified against the
latest official policy documents.

## Problem statement

Insurance brochures are long, terminology-heavy, and difficult to compare.
Important details such as eligibility limits, waiting periods, exclusions,
surrender rules, and benefit calculations may be spread across multiple pages.
BenefitExplorer turns those documents into a searchable, citation-backed
knowledge assistant while preserving page-level provenance.

## Key features

- Page-aware PDF extraction with PyMuPDF and pdfplumber
- Clause- and section-aware chunking with stable chunk IDs
- BGE-M3 embeddings stored in persistent Chroma collections
- BM25 and dense retrieval fused with Reciprocal Rank Fusion (RRF)
- Product detection with hard filtering and score boosting
- BGE cross-encoder reranking with intent, section, exact-phrase, and focused-window signals
- Near-duplicate removal and dynamic final-context selection
- Product-balanced evidence selection for comparison questions
- Grounded Qwen3 32B generation through Amazon Bedrock Mantle
- Strict inline citation validation against retrieved brochure text
- Streaming FastAPI and Next.js chat experience with expandable source cards
- RAGAS evaluation with a manually labeled golden dataset

## Architecture overview

```mermaid
flowchart LR
    A["Insurance brochure PDFs"] --> B["Page-aware extraction"]
    B --> C["Clause-aware chunks"]
    C --> D["BGE-M3 embeddings"]
    C --> E["BM25 index"]
    D --> F["Persistent Chroma store"]

    Q["Customer question"] --> P["Product and intent detection"]
    P --> F
    P --> E
    F --> H["Hybrid retrieval and RRF"]
    E --> H
    H --> R["BGE cross-encoder reranking"]
    R --> G["Bedrock Mantle generation"]
    G --> V["Citation validation"]
    V --> U["Next.js answer and source panel"]
```

The production pipeline retrieves up to 40 candidates and dynamically selects
at most four final context chunks. Comparison queries reserve evidence for each
detected product. The generator receives full chunks so citations can be
validated and rendered with product and page metadata.

## Tech stack

| Layer | Technology |
|---|---|
| PDF ingestion | PyMuPDF, pdfplumber, tiktoken |
| Embeddings | `BAAI/bge-m3`, Sentence Transformers |
| Sparse retrieval | BM25 |
| Vector database | Chroma |
| Rank fusion | Reciprocal Rank Fusion |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| Generation | Qwen3 32B through Amazon Bedrock Mantle |
| Backend | Python, FastAPI, Pydantic, Uvicorn |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui primitives |
| Streaming | NDJSON through a server-side Next.js proxy |
| Evaluation | RAGAS plus deterministic Context Recall@4 |

## Evaluation results

The final evaluation set contains **30 questions across six insurance
products**. Answers were assessed using exactly three metrics.

| Metric | Score |
|---|---:|
| Faithfulness (RAGAS) | **0.936** |
| Context Recall@4 (legacy flat chunk labels) | **0.837** |
| Answer Correctness (RAGAS) | **0.725** |

Faithfulness and Answer Correctness were available for 29 questions because
one prior RAGAS judge request failed; Context Recall@4 was available for all 30.
The evaluation code now uses evidence-group recall so overlapping chunks count
as alternatives rather than separate required hits. The historical `0.837`
recall value predates that correction and must be recalculated on the next full
run; it is retained only for transparent comparison. These results should not
be treated as a universal benchmark.

## Supported brochure set

The golden dataset represents six products:

- Kotak EDGE
- Kotak TULIP
- Kotak Gen2Gen Protect
- Kotak Fortune Maximiser
- Kotak GAIN
- Kotak Assured Pension

Raw brochures and generated indexes are intentionally excluded from Git. Add
only documents that you have permission to use or redistribute.

## Data Source

The system was evaluated using publicly available product brochures from Kotak
Life Insurance, downloaded from the official website for research and
educational purposes.

The original PDF files are not included in this repository due to copyright.
They remain local, are ignored by Git, and must be obtained independently from
the official publisher. A synthetic processed-chunk fixture and an example
document manifest are provided under `backend/data/` so the repository schema
can be inspected without redistributing brochure content.

## Project structure

```text
Insurance-Product-RAG/
├── backend/
│   ├── data/                         # Ignored local PDFs plus sanitized fixtures
│   ├── src/
│   │   ├── ingestion/                # PDF extraction and chunking
│   │   ├── retrieval/                # Embeddings, BM25, Chroma, RRF
│   │   ├── generation/               # Reranking, generation, citations
│   │   └── main.py                   # FastAPI application
│   ├── .env.example
│   ├── configure_mantle.sh
│   └── requirements.txt
├── frontend/
│   ├── app/                          # App Router pages and API proxy
│   ├── components/                   # Chat, citations, and source cards
│   ├── lib/                          # Shared types and utilities
│   ├── .env.example
│   └── package.json
├── evaluation/
│   ├── golden/                       # 30-question golden dataset
│   ├── src/                          # Three-metric evaluation pipeline
│   ├── results/                      # Generated JSON and CSV results
│   ├── reports/                      # Generated Markdown summaries
│   └── run_evaluation.py
├── .gitignore
└── README.md
```

## How to run

### Prerequisites

- Python 3.11–3.13 recommended
- Node.js 20 or later
- npm
- Amazon Bedrock Mantle API access to the configured model
- Sufficient disk space for the embedding and reranking models

The initial model download requires internet access. Once cached, retrieval and
reranking can load locally with `RAG_OFFLINE=true`.

### 1. Install the backend

```bash
git clone <your-repository-url>
cd Insurance-Product-RAG/backend

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints.txt
cp .env.example .env
```

Add the Bedrock credential safely with the included hidden-input helper:

```bash
chmod +x configure_mantle.sh
./configure_mantle.sh
```

The key is stored only in `backend/.env`, which is ignored by Git. Never put
provider credentials in frontend environment variables.

### 2. Ingest and index brochures

Place authorized PDF brochures in `backend/data/`, then run from `backend/`:

```bash
python -m src.ingestion.run_ingestion
python -m src.retrieval.run_retrieval index
```

To rebuild the vector collection after changing source documents or the
embedding model:

```bash
python -m src.retrieval.run_retrieval index --force
```

### 3. Start the API

```bash
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 4. Start the frontend

From a second terminal:

```bash
cd Insurance-Product-RAG/frontend
npm install
cp .env.example .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The Next.js server proxies
requests to FastAPI, so the Bedrock credential never reaches the browser.

### 5. Run evaluation

Install the optional evaluation dependencies from the repository root:

```bash
backend/.venv/bin/pip install -r evaluation/requirements.txt
```

Run all golden questions:

```bash
backend/.venv/bin/python evaluation/run_evaluation.py
```

Run a smoke test or resume an interrupted run:

```bash
backend/.venv/bin/python evaluation/run_evaluation.py --limit 3
backend/.venv/bin/python evaluation/run_evaluation.py --resume
```

Progress is checkpointed atomically after every completed question. Do not run
an original and resumed evaluation concurrently.

## Golden dataset

The dataset at `evaluation/golden/golden_questions.json` contains 30 realistic
customer questions spanning factual details, eligibility, exclusions,
benefits, surrender conditions, and cross-product comparisons. Each record
contains:

- A stable question ID
- The customer-style question
- A concise brochure-grounded reference answer
- One or more evidence-bearing chunk IDs
- Product and question-type labels where applicable

`evaluation/golden/evidence_groups.json` groups overlapping chunks that support
the same material claim. Context Recall@4 counts covered evidence groups, not
duplicate overlap windows; comparison products and distinct claims remain
separate evidence groups.

Relevant chunks are labeled only when they independently support the complete
answer or a major claim. Overlapping chunks are included when each contains
usable evidence. See `evaluation/golden/LABELING.md` for the labeling policy.

## Configuration

### Backend

| Variable | Default | Purpose |
|---|---|---|
| `AWS_BEARER_TOKEN_BEDROCK` | Required | Server-side Mantle credential |
| `AWS_REGION` | `us-east-1` | Mantle region |
| `OPENAI_BASE_URL` | Regional Mantle URL | OpenAI-compatible endpoint |
| `MANTLE_MODEL` | `qwen.qwen3-32b` | Generation model |
| `MANTLE_MAX_OUTPUT_TOKENS` | `1100` | Generation output limit |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Dense embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | Persistent index location |
| `CHROMA_COLLECTION` | `insurance_products` | Chroma collection name |
| `RAG_OFFLINE` | `true` | Require cached local retrieval models |
| `FRONTEND_ORIGINS` | `http://localhost:3000` | Allowed browser origins |
| `RAGAS_JUDGE_MODEL` | Generation model | Optional evaluation judge override |

### Frontend

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_API_URL` | `http://127.0.0.1:8000` | Server-side FastAPI URL |

## Limitations

- Answers are limited to the indexed brochure versions and may not reflect later product changes.
- OCR and complex PDF tables can introduce extraction or chunk-boundary errors.
- Context Recall@4 depends on manually maintained evidence-group labels.
- RAGAS metrics use a model judge and may vary across judge models or provider versions.
- Answer Correctness remains the weakest measured metric, especially for multi-part questions.
- The first local request can be slow while embedding and reranking models load.
- This application does not replace policy contracts or professional advice.

## Future improvements

- Add explicit multi-part question decomposition before retrieval and generation
- Add numeric consistency checks for ages, percentages, and policy thresholds
- Improve comparison-specific clause selection and evidence balancing
- Expand the golden dataset with independently reviewed labels
- Add automated backend and frontend test suites to CI
- Add authentication, observability, and deployment infrastructure
- Support OCR fallback for scanned brochures and richer table extraction

## Security

- Secrets belong only in ignored `.env` files; safe placeholders live in `.env.example`.
- The frontend never accepts or stores provider credentials.
- Raw PDFs, derived chunks, Chroma indexes, model caches, and generated reports are ignored.
- Review brochure redistribution rights before publishing any source document.

## License

The source code is available under the [MIT License](LICENSE). The license does
not apply to third-party insurance brochures and does not grant permission to
redistribute them.
