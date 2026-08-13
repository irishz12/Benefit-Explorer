# Sanitized test fixture

`sample_processed_chunks.json` is synthetic and contains no Kotak Life
Insurance brochure text. It demonstrates the processed-chunk schema and can be
used for lightweight development and tests without distributing source PDFs.

For a real local dataset:

1. Copy `document_manifest.example.json` to `document_manifest.json`.
2. Record the official source URL, retrieval date, and SHA-256 digest for every
   locally authorized PDF.
3. Place those PDFs directly in `backend/data/`.
4. Run ingestion and indexing as documented in the root README.

`document_manifest.json`, PDFs, processed chunks, and the Chroma index are
machine-local artifacts and are ignored by Git.
