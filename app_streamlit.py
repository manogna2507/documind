"""
DocuMind - Streamlit demo.

Calls the real functions in rag.py (hybrid_retrieve, semantic_search, bm25_search, answer).
Run locally:   streamlit run app_streamlit.py
Secrets:       put GROQ_API_KEY in .streamlit/secrets.toml (local) or in the host's Secrets box (cloud).
"""
import os
import re

import streamlit as st

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

# --- Secrets -> environment (rag.py builds the Groq client at import time, so do this first) ---
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # no secrets file: fall back to an environment variable

MAX_QUESTIONS_PER_SESSION = 10
MAX_QUESTION_CHARS = 300

EXAMPLES = [
    "What is the difference between PUT and PATCH?",
    "What does status code 409 mean?",
    "What is Reciprocal Rank Fusion?",
    "How does DocuMind avoid invented follow-up questions?",
]


@st.cache_resource(show_spinner="Loading the embedding model and indexing the demo documents (the first load takes a minute)...")
def load_engine():
    import rag  # heavy import: loads the embedding model, ChromaDB and the Groq client

    rag.DOCS_DIR = "demo_docs"  # the public demo only uses the original documents in demo_docs/
    rag.initialize()
    return rag


def tidy_citations(text):
    """Show citations as [Source N] whatever brackets the model used."""
    return re.sub(r"[\u3010\[]\s*(Source\s*\d+)\s*[\u3011\]]", r"[\1]", text)


def show_chunks(title, hits, key_text="text"):
    st.markdown(f"**{title}**")
    if not hits:
        st.caption("No results.")
        return
    for i, h in enumerate(hits, start=1):
        meta = h.get("meta", {})
        st.markdown(f"{i}. `{meta.get('source', '?')}` chunk {meta.get('chunk_index', '?')}")
        st.caption(h.get(key_text, "")[:160] + "...")


# ------------------------------------------------------------------ header
st.title("📄 DocuMind")
st.caption(
    "Ask questions over documents. Hybrid retrieval (semantic + BM25, merged with Reciprocal Rank Fusion) "
    "and a grounded LLM answer with cited sources."
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it in the app's Secrets (see README).")
    st.stop()

if "asked" not in st.session_state:
    st.session_state.asked = 0
if "question" not in st.session_state:
    st.session_state.question = ""

tab_demo, tab_eval, tab_about = st.tabs(["Try it", "Evaluation", "How it works"])

# ------------------------------------------------------------------ demo tab
with tab_demo:
    st.write("The demo index contains two short documents: **REST API basics** and **About DocuMind**.")

    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state.question = ex

    question = st.text_input("Your question", key="question", max_chars=MAX_QUESTION_CHARS)
    ask = st.button("Ask", type="primary")

    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.asked
    st.caption(f"{remaining} questions left in this session (a limit that protects the free API quota).")

    if ask and question.strip():
        if remaining <= 0:
            st.warning("Session limit reached. Refresh the page to start a new session.")
        else:
            rag = load_engine()
            st.session_state.asked += 1
            with st.spinner("Retrieving and generating..."):
                try:
                    result = rag.answer(question.strip())
                    sem_hits = rag.semantic_search(question.strip(), 5)
                    bm25_hits = rag.bm25_search(question.strip(), 5)
                    hybrid_hits = rag.hybrid_retrieve(question.strip())
                except Exception as exc:  # e.g. rate limit from the free API tier
                    st.error(f"Something went wrong: {exc}")
                    st.stop()

            st.subheader("Answer")
            st.markdown(tidy_citations(result["answer"]))

            with st.expander("Sources used for the answer", expanded=True):
                for s in result["sources"]:
                    st.markdown(f"**[Source {s['index']}]** `{s['source']}` chunk {s['chunk_index']}")
                    st.caption(s["text_preview"] + "...")

            with st.expander("Compare the three retrievers (top 5 each)"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    show_chunks("Semantic only", sem_hits)
                with c2:
                    show_chunks("BM25 only", bm25_hits)
                with c3:
                    show_chunks("Hybrid (RRF)", hybrid_hits)
    elif ask:
        st.info("Type a question first.")

    st.caption(
        "File upload is disabled in this public demo so that visitors' documents are not shared. "
        "The API (`main.py`, `POST /ingest`) supports uploads when you run it yourself."
    )

# ------------------------------------------------------------------ evaluation tab
with tab_eval:
    st.subheader("Retrieval evaluation (20-question benchmark)")
    st.table(
        {
            "Retriever": ["Semantic only", "BM25 only", "Hybrid (RRF)"],
            "Recall@5": ["11/20 (55%)", "11/20 (55%)", "12/20 (60%)"],
            "MRR": ["0.517", "0.310", "0.483"],
        }
    )
    st.markdown(
        "- Hybrid retrieval has the **best Recall@5** (12/20 against 11/20), a small gain.\n"
        "- **Semantic-only has the best MRR**, so hybrid ranks the correct chunk lower when both find it. "
        "That suggests tuning the RRF constant or the per-retriever top-k.\n"
        "- **Limitations:** a small, self-written benchmark; a strict hit criterion; retrieval only "
        "(no answer-quality scoring yet). Treat the numbers as indicative, not as a proof."
    )

# ------------------------------------------------------------------ about tab
with tab_about:
    st.markdown(
        """
**Pipeline:** PDF/TXT loading (pdfplumber) → text cleanup → chunking (500 characters, 50 overlap) →
BGE embeddings (`BAAI/bge-small-en-v1.5`, local) → ChromaDB → **semantic (top-8) + BM25 (top-8)** merged with
**Reciprocal Rank Fusion (k=60)** → top-5 chunks → an open-weight LLM via Groq (gpt-oss-120b) with a grounded prompt, stop sequences and
post-processing that trims invented follow-up questions.

**Why hybrid:** semantic search misses exact keywords, BM25 ignores meaning, and RRF combines the two rankings
without needing to normalise scores.

Source code and the evaluation: [github.com/manogna2507/documind](https://github.com/manogna2507/documind)
"""
    )
