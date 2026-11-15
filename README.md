# DocuMind

Ask questions over your documents and get short answers with cited sources. A Retrieval-Augmented Generation (RAG) system with **hybrid retrieval** (semantic + BM25, merged with Reciprocal Rank Fusion) and a grounded LLM answer step.

**Live demo:** [documind-manogna.streamlit.app](https://documind-manogna.streamlit.app/)  
**Stack:** Python · FastAPI · Streamlit · ChromaDB · `BAAI/bge-small-en-v1.5` · rank_bm25 · an open-weight LLM via the Groq API (default: gpt-oss-120b) · pdfplumber

<!-- add demo.gif here: ![demo](demo.gif) -->

## How it works

```
PDF/TXT -> pdfplumber -> text cleanup -> chunks (500 characters, 50 overlap)
        -> BGE embeddings -> ChromaDB
question -> semantic top-8 + BM25 top-8 -> Reciprocal Rank Fusion (k=60) -> top-5 chunks
         -> LLM via Groq (grounded prompt, stop sequences) -> post-processing -> answer + sources
```

- **Ingestion:** `pdfplumber` loading, cleanup (ligature fixing, hyphen-joining), recursive chunking.
- **Embedding:** `BAAI/bge-small-en-v1.5` through `sentence-transformers`, run locally (no embedding API cost).
- **Retrieval:** semantic search and BM25 run side by side and are merged with Reciprocal Rank Fusion, which needs no score normalisation.
- **Generation:** a strict grounded prompt (answer only from context, cite sources, one short answer), stop sequences, and regex post-processing.
- **API:** FastAPI + Pydantic with auto-generated OpenAPI docs (`/query`, `/ingest`, `/health`).

## Evaluation

A 20-question evaluation benchmark written against three source documents. A retrieved chunk counts as a hit only if it comes from the expected document **and** contains all the expected keywords.

| Retriever | Recall@5 | MRR |
|---|---|---|
| Semantic only | 11/20 (55%) | **0.517** |
| BM25 only | 11/20 (55%) | 0.310 |
| **Hybrid (RRF)** | **12/20 (60%)** | 0.483 |

**What the numbers say, honestly:**
- Hybrid retrieval has the best Recall@5, but only by one question (12 against 11), which is not statistically meaningful on 20 questions.
- Semantic-only has the best MRR: when both find the right chunk, semantic ranks it higher. This suggests tuning the RRF constant and the per-retriever top-k, which is on the to-do list.
- BM25 alone ranks relevant chunks poorly (MRR 0.310), as expected.

**Limitations:** the benchmark is small and written by the same person who built the system; the hit criterion is strict; and it measures retrieval only, not answer quality. Details and the per-question results are in [`eval/results.md`](eval/results.md) and `eval/results.json`.

## Two bugs I found and fixed

**1. Garbled text from PDF extraction.** Early answers started mid-word ("aling with" instead of "dealing with"). I traced it to `pypdf` dropping characters at line breaks, switched to `pdfplumber`, and added a cleanup step for ligatures (a single "fi" character), hyphenated line breaks and stray whitespace. Retrieval quality improved noticeably (observed in manual testing, not yet measured).

**2. Invented follow-up questions.** Even with a grounded prompt, the model sometimes continued a correct answer with invented follow-up questions and answers. I fixed it in three layers:

1. a stricter system prompt (answer once, then stop; never invent follow-up questions),
2. `stop` sequences that cut generation when a follow-up pattern starts,
3. regex post-processing that trims any follow-up that still gets through.

_Not yet quantified:_ I have not measured how often either problem happened before and after the fixes. A before/after check on the benchmark questions is planned.

## Design trade-offs

- **pdfplumber vs pypdf:** better text extraction but much slower, which is fine because ingestion is a one-time cost, not query-time.
- **Hybrid retrieval with RRF:** semantic search alone misses exact keywords (for example "uvicorn"); RRF combines the two rankings without fragile score normalisation.
- **Post-processing as a safety net:** stop tokens alone did not remove invented follow-ups.
- **A design decision I would change now:** the vector index is global to the process, so a multi-user deployment would mix different users' documents. I would create one collection per user or session.

## Run it locally

```bash
git clone https://github.com/manogna2507/documind.git
cd documind
python -m venv venv && source venv/bin/activate

# Streamlit demo
pip install -r requirements.txt
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your Groq key
streamlit run app_streamlit.py

# or the FastAPI backend
pip install -r requirements-api.txt
export GROQ_API_KEY="your_key_here"
uvicorn main:app --reload        # Swagger UI at http://127.0.0.1:8000/docs
```

To reproduce the evaluation: `export GROQ_API_KEY=... && python eval/run_eval.py`.

## Deployment notes

The public demo runs on Streamlit Community Cloud. It indexes the two original documents in `demo_docs/`. File upload is **disabled in the public demo** so that visitors' documents are not shared between sessions; the API supports `/ingest` when you run it yourself. Each demo session is limited to 10 questions to protect the free API quota.

## Known limitations and next steps

- Expand the benchmark to 100+ questions written by someone other than the author.
- Add answer-quality (faithfulness) evaluation, and measure the follow-up fix before and after.
- Tune the RRF constant and top-k values on a validation split.
- Per-user collections for a multi-user deployment.
- Re-check the follow-up fixes on the current model: the project was built with Llama 3.3 70B, which Groq retired on 2026-08-16. The model is now set by the `LLM_MODEL` environment variable (default `openai/gpt-oss-120b`).

## Author

Manogna Gampala · [LinkedIn](https://www.linkedin.com/in/manogna-gampala-36220b292) · [GitHub](https://github.com/manogna2507)
