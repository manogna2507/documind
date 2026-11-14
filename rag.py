"""
DocuMind - Day 3: RAG engine (used by main.py FastAPI backend).
"""
import os
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import re
import glob
from pathlib import Path

from langchain_community.document_loaders import TextLoader
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from groq import Groq

# --- Config ---
DOCS_DIR = "docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")  # llama-3.3-70b-versatile was retired by Groq on 2026-08-16
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEMANTIC_TOP_K = 8
BM25_TOP_K = 8
FINAL_TOP_K = 5

# --- Init ---
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Initializing ChromaDB vector store...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documind")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

bm25_index = None
bm25_corpus_texts = []
bm25_corpus_metas = []


# --- Text cleanup ---
LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "ft", "ﬆ": "st",
    "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u00A0": " ",
}

def clean_text(text):
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text):
    return re.findall(r"\w+", text.lower())


# --- Ingestion ---
def load_document(path):
    if path.endswith(".pdf"):
        docs = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"page": page_num, "source": path},
                    ))
        return docs
    else:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()


def _chunk_and_store(docs, source_name):
    """Shared logic: clean, chunk, embed, add to Chroma. Returns chunk count."""
    for d in docs:
        d.page_content = clean_text(d.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source_name, "chunk_index": i}
        for i in range(len(chunks))
    ]
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    return len(chunks)


def ingest_all_docs():
    files = glob.glob(f"{DOCS_DIR}/*.pdf") + glob.glob(f"{DOCS_DIR}/*.txt")
    if not files:
        print(f"No documents found in ./{DOCS_DIR}/.")
        return
    for path in files:
        source_name = Path(path).stem
        print(f"Ingesting {source_name}...")
        docs = load_document(path)
        n = _chunk_and_store(docs, source_name)
        print(f"  -> {n} chunks")


def ingest_single_file(path: str) -> int:
    """Ingest one file (used by the /ingest endpoint). Returns chunk count."""
    source_name = Path(path).stem
    docs = load_document(path)
    n = _chunk_and_store(docs, source_name)
    build_bm25_index()  # rebuild BM25 after adding new docs
    return n


def build_bm25_index():
    global bm25_index, bm25_corpus_texts, bm25_corpus_metas
    all_data = collection.get()
    bm25_corpus_texts = all_data["documents"]
    bm25_corpus_metas = all_data["metadatas"]
    tokenized_corpus = [tokenize(t) for t in bm25_corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built over {len(bm25_corpus_texts)} chunks.")


# --- Hybrid Retrieval ---
def semantic_search(query, top_k):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "meta": meta, "semantic_score": 1.0 - dist})
    return hits


def bm25_search(query, top_k):
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        hits.append({
            "text": bm25_corpus_texts[idx],
            "meta": bm25_corpus_metas[idx],
            "bm25_score": float(scores[idx]),
        })
    return hits


def hybrid_retrieve(query, final_k=FINAL_TOP_K):
    sem_hits = semantic_search(query, SEMANTIC_TOP_K)
    bm25_hits = bm25_search(query, BM25_TOP_K)
    K = 60  # RRF constant
    fused = {}
    for rank, hit in enumerate(sem_hits):
        key = (hit["meta"]["source"], hit["meta"]["chunk_index"])
        fused.setdefault(key, {"text": hit["text"], "meta": hit["meta"], "score": 0.0})
        fused[key]["score"] += 1.0 / (K + rank)
    for rank, hit in enumerate(bm25_hits):
        key = (hit["meta"]["source"], hit["meta"]["chunk_index"])
        fused.setdefault(key, {"text": hit["text"], "meta": hit["meta"], "score": 0.0})
        fused[key]["score"] += 1.0 / (K + rank)
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:final_k]


# --- Generation ---
def answer(query):
    """RAG pipeline: hybrid retrieve, then ask the LLM. Returns dict."""
    hits = hybrid_retrieve(query)

    if not hits:
        return {"answer": "I don't have enough information in the provided context.", "sources": []}

    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        m = hit["meta"]
        context_blocks.append(
            f"[Source {i} | {m['source']} chunk {m['chunk_index']}]\n{hit['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""Context:
{context}

Question: {query}"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content":
                "You are a documentation Q&A assistant. Rules you MUST follow:\n"
                "1. Answer ONLY from the provided context. Do not use outside knowledge.\n"
                "2. Give exactly ONE answer, 2-4 sentences maximum. Then STOP.\n"
                "3. Cite sources inline like [Source 1].\n"
                "4. If the context does not contain the answer, respond with exactly: "
                "'I don't have enough information in the provided context.' and nothing else.\n"
                "5. Never invent follow-up questions.\n"
                "6. Never say both an answer AND 'I don't have enough information'. Pick one."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        **({"reasoning_effort": "low"} if "gpt-oss" in LLM_MODEL else {}),
        max_tokens=1200,
        stop=["\n\nQ:", "\n\nQuestion:", "Please verify", "Here are the answers"],
    )

    answer_text = response.choices[0].message.content.strip()

    # Trim hallucinated follow-ups
    cutoff_patterns = [
        r"\n\s*Q:",
        r"\n\s*Question:",
        r"\n\s*Please\s+(verify|answer)",
        r"\n\s*Here\s+are\s+the\s+answers",
        r"\.\s+What\s+(is|does|are|do|can)\s",
        r"\.\s+Why\s+",
        r"\.\s+How\s+(do|does|can|is)\s",
        r"\.\s+(Can|Is|Does|Are)\s+you\s+",
        r"I don'?t have any information about the (tool|user|context)",
    ]
    for pat in cutoff_patterns:
        m = re.search(pat, answer_text, flags=re.IGNORECASE)
        if m:
            answer_text = answer_text[:m.start()].strip()
            if answer_text and answer_text[-1] not in ".!?":
                answer_text += "."
            break

    sources = [
        {
            "index": i,
            "source": hit["meta"]["source"],
            "chunk_index": hit["meta"]["chunk_index"],
            "text_preview": hit["text"][:200],
        }
        for i, hit in enumerate(hits, start=1)
    ]
    return {"answer": answer_text, "sources": sources}


# --- Startup helper for the API ---
def initialize():
    """Called once at API startup: ingest docs if needed, build BM25."""
    if collection.count() == 0:
        ingest_all_docs()
    else:
        print(f"Collection already has {collection.count()} chunks. Skipping ingestion.")
    build_bm25_index()