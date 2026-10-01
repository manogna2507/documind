"""
DocuMind Evaluation Script
Runs 3 retrieval variants over questions.json and reports Recall@5 and MRR.
"""
import json
import sys
from pathlib import Path

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import rag


def check_hit(chunk_text, chunk_source, expected_source, expected_keywords):
    """A hit = correct source AND all expected keywords present (case-insensitive)."""
    if chunk_source != expected_source:
        return False
    text_lower = chunk_text.lower()
    return all(kw.lower() in text_lower for kw in expected_keywords)


def evaluate_retriever(retriever_name, retriever_fn, questions, top_k=5):
    """Run a retriever over all questions, return metrics."""
    hits = 0
    reciprocal_ranks = []
    per_question = []

    for q in questions:
        results = retriever_fn(q["question"], top_k)

        found_rank = None
        for rank, hit in enumerate(results, start=1):
            source = hit["meta"]["source"]
            text = hit["text"]
            if check_hit(text, source, q["expected_source"], q["expected_keywords"]):
                found_rank = rank
                break

        if found_rank:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
            per_question.append({"id": q["id"], "hit": True, "rank": found_rank})
        else:
            reciprocal_ranks.append(0.0)
            per_question.append({"id": q["id"], "hit": False, "rank": None})

    total = len(questions)
    recall_at_k = hits / total
    mrr = sum(reciprocal_ranks) / total

    return {
        "name": retriever_name,
        "recall_at_5": recall_at_k,
        "hits": hits,
        "total": total,
        "mrr": mrr,
        "per_question": per_question,
    }


def semantic_only(query, top_k):
    return rag.semantic_search(query, top_k)


def bm25_only(query, top_k):
    return rag.bm25_search(query, top_k)


def hybrid(query, top_k):
    return rag.hybrid_retrieve(query, final_k=top_k)


def main():
    # Load questions
    questions_path = Path(__file__).parent / "questions.json"
    with open(questions_path) as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions.\n")

    # Ensure indexes are built
    if rag.collection.count() == 0:
        print("Vector store empty. Ingesting documents...")
        rag.ingest_all_docs()
    rag.build_bm25_index()

    # Run each retriever
    print("Running Semantic-only...")
    sem = evaluate_retriever("Semantic-only", semantic_only, questions)

    print("Running BM25-only...")
    bm25 = evaluate_retriever("BM25-only", bm25_only, questions)

    print("Running Hybrid (RRF)...")
    hyb = evaluate_retriever("Hybrid (RRF)", hybrid, questions)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Retriever':<20} {'Recall@5':<15} {'MRR':<10}")
    print("-" * 60)
    for r in [sem, bm25, hyb]:
        recall_str = f"{r['hits']}/{r['total']} ({r['recall_at_5']:.0%})"
        print(f"{r['name']:<20} {recall_str:<15} {r['mrr']:.3f}")

    # Save full results as JSON
    output = {
        "summary": {
            "semantic_only": {"recall_at_5": sem["recall_at_5"], "hits": sem["hits"], "total": sem["total"], "mrr": sem["mrr"]},
            "bm25_only": {"recall_at_5": bm25["recall_at_5"], "hits": bm25["hits"], "total": bm25["total"], "mrr": bm25["mrr"]},
            "hybrid_rrf": {"recall_at_5": hyb["recall_at_5"], "hits": hyb["hits"], "total": hyb["total"], "mrr": hyb["mrr"]},
        },
        "per_question": {
            "semantic_only": sem["per_question"],
            "bm25_only": bm25["per_question"],
            "hybrid_rrf": hyb["per_question"],
        }
    }
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()