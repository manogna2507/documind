# DocuMind

An end-to-end Retrieval-Augmented Generation (RAG) system that answers natural-language questions over technical documentation.

**Live API:** [Coming soon — deploying]  
**Interactive docs:** `/docs` on the deployed URL

## Architecture

- **Ingestion:** PDF/TXT loading via `pdfplumber` → text cleanup (ligature fixing, hyphen-join) → recursive chunking (500 tokens, 50 overlap)
- **Embedding:** `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local, no API cost)
- **Vector store:** ChromaDB (persistent)
- **Retrieval:** Hybrid — semantic (top-8) + BM25 (top-8), merged with Reciprocal Rank Fusion (k=60), final top-5
- **Generation:** Llama 3.3 70B via Groq API, with grounded prompting, output stops, and post-processing to prevent hallucinated follow-ups
- **API:** FastAPI + Pydantic + auto-generated OpenAPI 3.1 docs

## Endpoints

- `GET /health` — service status + total chunks in index
- `POST /query` — ask a question, returns `{answer, sources}`
- `POST /ingest` — upload a PDF/TXT to add to the index
- `GET /docs` — Swagger UI

## Tech Stack

Python 3.11 · FastAPI · LangChain · sentence-transformers · ChromaDB · rank_bm25 · Groq · Llama 3.3 · pdfplumber · Uvicorn

## Design Tradeoffs

- **pdfplumber vs pypdf:** ~3× better text extraction but ~10× slower — worth it for ingestion (one-time cost, not query-time)
- **Hybrid retrieval with RRF:** Semantic search alone misses exact keyword matches (e.g. "uvicorn"). RRF combines rankings without requiring score normalization, which is fragile across different retrievers.
- **Grounded prompting + post-processing:** LLMs still hallucinate follow-up questions even with `stop` tokens. Regex-based post-processing acts as a safety net.

## Author

Manogna Gampala · [LinkedIn](https://linkedin.com/in/manogna-gampala-36220b292) · [GitHub](https://github.com/manogna2507)