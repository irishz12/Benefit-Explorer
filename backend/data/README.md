# Local brochure data

Place authorized insurance brochure PDFs in this directory before running the
ingestion pipeline. The following generated or potentially restricted files
are intentionally excluded from Git:

- Source brochure PDFs (`*.pdf`)
- `processed_chunks.json`
- The persistent `chroma_db/` index

Generate the local artifacts from `backend/`:

```bash
python -m src.ingestion.run_ingestion
python -m src.retrieval.run_retrieval index
```

Only use and redistribute documents for which you have the necessary rights.

For reproducibility without redistributing third-party text, this directory
also includes:

- `document_manifest.example.json` — schema for recording local source URLs,
  retrieval dates, and SHA-256 digests
- `fixtures/sample_processed_chunks.json` — entirely synthetic chunk examples

Copy the example manifest to `document_manifest.json` for local use. The local
manifest is ignored because it may describe brochure files that are not
redistributed with this repository.
