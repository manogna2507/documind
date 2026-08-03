"""
DocuMind - Day 2: hybrid retrieval (semantic + BM25), multi-doc, cleaner output.
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
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEMANTIC_TOP_K = 8   # candidates from vector search
BM25_TOP_K = 8       # candidates from BM25
FINAL_TOP_K = 5      # after merging & re-ranking

# --- Initialize ---
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Initializing ChromaDB vector store...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documind")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory BM25 index (rebuilt from Chroma on startup)
bm25_index = None
bm25_corpus_texts = []
bm25_corpus_metas = []


# --- Text cleanup ---
# Unicode ligatures commonly emitted by PDF text extractors
LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "ft", "ﬆ": "st",
    "\u2013": "-", "\u2014": "-",     # en/em dashes -> hyphen
    "\u2018": "'", "\u2019": "'",     # curly single quotes
    "\u201C": '"', "\u201D": '"',     # curly double quotes
    "\u00A0": " ",                    # non-breaking space
}

def clean_text(text):
    """Fix common PDF extraction issues."""
    # 1. Replace ligatures and typographic characters
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    # 2. Join words split across lines by hyphen: "Fast-\nAPI" -> "FastAPI"
    text = re.sub(r"-\s*\n\s*", "", text)
    # 3. Turn single newlines inside paragraphs into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 4. Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # 5. Collapse 3+ newlines to just 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text):
    """Simple tokenizer for BM25: lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


# --- Ingestion ---
def load_document(path):
    """Load a PDF or plain text file."""
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


def ingest_all_docs():
    """Ingest every PDF/TXT in DOCS_DIR."""
    files = glob.glob(f"{DOCS_DIR}/*.pdf") + glob.glob(f"{DOCS_DIR}/*.txt")
    if not files:
        print(f"No documents found in ./{DOCS_DIR}/. Add some PDFs or TXT files.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for path in files:
        source_name = Path(path).stem  # filename without extension
        print(f"Ingesting {source_name}...")

        docs = load_document(path)
        for d in docs:
            d.page_content = clean_text(d.page_content)

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
        print(f"  -> {len(chunks)} chunks")


def build_bm25_index():
    """Build BM25 index from all chunks in Chroma."""
    global bm25_index, bm25_corpus_texts, bm25_corpus_metas
    all_data = collection.get()
    bm25_corpus_texts = all_data["documents"]
    bm25_corpus_metas = all_data["metadatas"]
    tokenized_corpus = [tokenize(t) for t in bm25_corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built over {len(bm25_corpus_texts)} chunks.")


# --- Hybrid Retrieval ---
def semantic_search(query, top_k):
    """Vector similarity search via Chroma."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    # Chroma returns cosine distance (lower = better). Convert to similarity.
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "meta": meta,
            "semantic_score": 1.0 - dist,  # higher = better
        })
    return hits


def bm25_search(query, top_k):
    """Keyword search via BM25."""
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    # Get top_k indices
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
    """Combine semantic + BM25 using Reciprocal Rank Fusion (RRF)."""
    sem_hits = semantic_search(query, SEMANTIC_TOP_K)
    bm25_hits = bm25_search(query, BM25_TOP_K)

    # RRF: score each doc by sum of 1/(k + rank) across both lists
    K = 60  # RRF constant (standard)
    fused = {}  # key: (source, chunk_index) -> {"text", "meta", "score"}

    for rank, hit in enumerate(sem_hits):
        key = (hit["meta"]["source"], hit["meta"]["chunk_index"])
        fused.setdefault(key, {"text": hit["text"], "meta": hit["meta"], "score": 0.0})
        fused[key]["score"] += 1.0 / (K + rank)

    for rank, hit in enumerate(bm25_hits):
        key = (hit["meta"]["source"], hit["meta"]["chunk_index"])
        fused.setdefault(key, {"text": hit["text"], "meta": hit["meta"], "score": 0.0})
        fused[key]["score"] += 1.0 / (K + rank)

    # Sort by fused score, take top-k
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:final_k]


# --- Generation ---
def answer(query):
    """RAG pipeline: hybrid retrieve, then ask the LLM."""
    hits = hybrid_retrieve(query)

    if not hits:
        return "I don't have enough information in the provided context."

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
                "5. Never invent follow-up questions. Never write 'The question is', "
                "'Please verify', or 'Here are the answers'.\n"
                "6. Never say both an answer AND 'I don't have enough information'. Pick one."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=400,
        stop=["\n\nQ:", "\n\nQuestion:", "Please verify", "Here are the answers"],
    )

    answer_text = response.choices[0].message.content.strip()
    # Trim anything after the first hallucinated follow-up question
    # Trim anything after the first hallucinated follow-up.
    # Splits at obvious "new question" markers whether on a new line or inline.
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
            # Add back the sentence-ending period if we cut mid-sentence
            if answer_text and answer_text[-1] not in ".!?":
                answer_text += "."
            break
    # Append source list for transparency
    source_lines = "\n".join(
        f"  [Source {i}] {hit['meta']['source']}, chunk {hit['meta']['chunk_index']}"
        for i, hit in enumerate(hits, start=1)
    )
    return f"{answer_text}\n\nSources used:\n{source_lines}"


def initialize():
    """Called once at API startup: ingest docs if needed, build BM25."""
    if collection.count() == 0:
        ingest_all_docs()
    else:
        print(f"Collection already has {collection.count()} chunks. Skipping ingestion.")
    build_bm25_index()


def ingest_single_file(path: str) -> int:
    """Ingest one file (used by the /ingest endpoint). Returns chunk count."""
    from pathlib import Path
    source_name = Path(path).stem
    docs = load_document(path)
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
    build_bm25_index()  # rebuild BM25 after adding new docs
    return len(chunks)