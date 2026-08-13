# Golden dataset relevance standard

`golden_questions.json` contains the questions and flat, auditable chunk labels.
`evidence_groups.json` is the evidence source of truth for Context Recall@4.
Faithfulness is evaluated against the final contexts selected by the live RAG
pipeline. Faithfulness and Answer Correctness are computed by RAGAS.

## Relevance rule

A chunk is relevant when it independently contains enough brochure evidence to
support at least one material claim in the reference answer. A chunk does not
need to contain the complete answer when the answer legitimately requires
evidence from multiple products, pages, or clauses.

Label a chunk as relevant when:

- it directly states a benefit, condition, limitation, amount, age, period, or
  comparison point used in the reference answer; or
- it is a clear overlapping chunk that repeats the same answer-bearing clause.

Do not label a chunk as relevant when it:

- only mentions the product or option name;
- contains the same keywords but discusses a different clause;
- provides background or an illustration that does not establish a material
  claim in the reference answer; or
- belongs to the correct product but cannot be cited to answer the question.

For cross-product comparisons, `product` is `null`, and `relevant_chunk_ids`
must contain direct evidence for every product discussed in the answer.

Chunks that state the same claim because of ingestion overlap belong to one
evidence group. Distinct material claims and each product in a comparison use
separate groups. Retrieving any chunk inside a group covers that evidence item;
retrieving multiple members does not earn extra recall. Every grouped ID must
also exist in `relevant_chunk_ids` and in `backend/data/processed_chunks.json`.

## Review checklist

Before evaluating:

1. Confirm that every answer is fully supported by the labeled chunks taken
   together.
2. Search all chunks from each named product for overlapping answer-bearing
   clauses.
3. Add every overlap that independently supports a material claim.
4. Confirm that comparison questions have evidence for both products.
5. Reject labels based only on shared keywords.
