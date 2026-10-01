# DocuMind Evaluation

## Goal

Validate the design choice of hybrid retrieval (semantic + BM25, merged via Reciprocal Rank Fusion) by comparing it against each retriever used alone.

## Methodology

- **Benchmark:** 20 questions written against the source documents (`sample.pdf`, `path_params.pdf`, `python_classes.pdf`), with pre-annotated expected source and expected keywords for each question. See `questions.json`.
- **Hit criterion:** A retrieved chunk counts as correct if (a) it comes from the expected source document and (b) all expected keywords appear in the chunk text (case-insensitive).
- **Metrics:**
  - **Recall@5** — fraction of questions where the correct chunk appeared in the top-5 retrieved chunks.
  - **MRR (Mean Reciprocal Rank)** — mean of `1 / rank` across all questions, where `rank` is the position of the correct chunk (0 if not found in top-5).
- **Retrievers compared:**
  - `Semantic-only` — ChromaDB similarity search over BGE embeddings, top-5.
  - `BM25-only` — rank_bm25 keyword search over the same chunk corpus, top-5.
  - `Hybrid (RRF)` — top-8 from each retriever merged via Reciprocal Rank Fusion (k=60), final top-5.

## Results

| Retriever       | Recall@5     | MRR    |
|-----------------|--------------|--------|
| Semantic-only   | 11/20 (55%)  | 0.517  |
| BM25-only       | 11/20 (55%)  | 0.310  |
| Hybrid (RRF)    | 12/20 (60%)  | 0.483  |

Per-question results are in `results.json` (produced by `run_eval.py`).

## Interpretation

- **Hybrid retrieval marginally outperforms both baselines on Recall@5** (12/20 vs 11/20), validating the design direction. The gain is modest on this benchmark, likely because many questions are answerable from a single clearly-dominant chunk that either retriever alone can find.
- **Semantic-only has the highest MRR,** meaning when it finds the correct chunk, it ranks it higher than hybrid does. This suggests room to tune the RRF constant (currently `k=60`) or the per-retriever top-k values.
- **BM25-only has the lowest MRR,** confirming that keyword search alone finds relevant chunks but ranks them poorly — a classic BM25 limitation that semantic rescoring addresses.

## Known limitations

This is a **small, self-designed benchmark**, not a public one. Specific limitations:

- **Author bias.** The same person wrote the questions and built the system. Questions may unconsciously favor patterns the retriever already handles well.
- **Strict hit criterion.** Requiring *all* expected keywords in a single chunk is conservative. A chunk that genuinely answers the question but phrases it differently is marked as a miss.
- **Only 20 questions.** Too few to establish statistical significance of the Recall@5 difference.
- **No answer-quality evaluation.** This benchmark measures retrieval quality only, not the faithfulness or correctness of the final LLM answer.

## Future work

- Expand to 100+ questions written by annotators independent of the system author.
- Add faithfulness scoring using an LLM-as-judge framework (e.g. RAGAS) to evaluate generated answers, not just retrieval.
- Validate on a public benchmark subset (e.g. BEIR, HotpotQA) to establish absolute performance against published baselines.
- Tune the RRF constant and per-retriever top-k values on a validation split.

## Reproduce

```bash
export GROQ_API_KEY="your_key"
python eval/run_eval.py
```