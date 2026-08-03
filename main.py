"""
DocuMind API - FastAPI backend for the RAG service.
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag

app = FastAPI(
    title="DocuMind API",
    description="Ask questions over your technical documentation using RAG.",
    version="0.1.0",
)

# CORS: allow the frontend (Day 4) to call this API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---
class QueryRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    index: int
    source: str
    chunk_index: int
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks_in_index: int


# --- Lifecycle ---
@app.on_event("startup")
def startup_event():
    print("Starting DocuMind API...")
    rag.initialize()
    print("DocuMind API ready.")


# --- Endpoints ---
@app.get("/health")
def health():
    """Liveness check + basic stats."""
    return {
        "status": "ok",
        "total_chunks": rag.collection.count(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ask a question, get a grounded answer with source citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = rag.answer(req.question)
    return result


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    """Upload a PDF or TXT file and add it to the index."""
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf and .txt files are supported.",
        )

    # Save uploaded file to a temp path
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks_added = rag.ingest_single_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # clean up temp file

    return {
        "filename": file.filename,
        "chunks_added": chunks_added,
        "total_chunks_in_index": rag.collection.count(),
    }