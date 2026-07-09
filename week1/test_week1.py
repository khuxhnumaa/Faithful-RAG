import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import load_retriever, retrieve, build_context, run_rag


# ── Test Queries ──────────────────────────────────────────────────────────────

TEST_QUERIES = [
    # Should answer correctly from docs
    ("How tall is the Eiffel Tower?",                    "easy"),
    ("When was the Eiffel Tower built?",                 "easy"),
    ("What is supervised learning?",                     "easy"),
    ("What is reinforcement learning used for?",         "medium"),

    # Should say "I don't have enough information"
    ("Who is the current President of France?",          "out-of-scope"),
    ("What is the speed of light?",                      "out-of-scope"),
]


def test_retrieval(vectorstore):
    """Check that retrieval returns relevant chunks."""
    print("\n" + "="*60)
    print("TEST 1: Retrieval quality")
    print("="*60)

    query = "How tall is the Eiffel Tower?"
    docs, scores = retrieve(vectorstore, query, top_k=3)

    print(f"\nTop chunk preview:\n{docs[0].page_content[:200]}")
    print(f"\nScores: {[round(s, 3) for s in scores]}")

    assert len(docs) > 0, "ERROR: No documents retrieved!"
    assert scores[0] > 0.3,  f"WARNING: Top score {scores[0]:.3f} seems low. Check embeddings."
    print("\n[PASS] Retrieval working correctly.")


def test_context_builder(vectorstore):
    """Check that context is properly formatted."""
    print("\n" + "="*60)
    print("TEST 2: Context builder")
    print("="*60)

    query = "What is deep learning?"
    docs, _ = retrieve(vectorstore, query, top_k=2)
    context = build_context(docs)

    assert "[Source 1:" in context, "ERROR: Sources not labeled in context!"
    assert "---" in context,        "ERROR: Chunk separator missing!"
    print(f"\nContext preview (first 300 chars):\n{context[:300]}...")
    print("\n[PASS] Context builder working correctly.")


def test_end_to_end(backend: str = "ollama"):
    """Run a full RAG query and print results."""
    print("\n" + "="*60)
    print(f"TEST 3: End-to-end RAG (backend={backend})")
    print("="*60)

    query = "When was the Eiffel Tower built and how tall is it?"
    result = run_rag(query, backend="ollama")

    print(f"\nRetrieval scores: {[round(s, 3) for s in result.retrieval_scores]}")
    print(f"Context length:   {len(result.context)} chars")
    print(f"Answer length:    {len(result.answer)} chars")

    assert result.answer.strip() != "", "ERROR: Empty answer!"
    print("\n[PASS] End-to-end pipeline working.")
    return result


def run_all_queries(backend: str = "ollama"):
    """Run all test queries and display results in a table."""
    print("\n" + "="*60)
    print("TEST 4: Full query suite")
    print("="*60)

    results = []
    for query, difficulty in TEST_QUERIES:
        print(f"\n[{difficulty.upper()}] {query}")
        result = run_rag(query, backend=backend)
        results.append({
            "query":      query,
            "difficulty": difficulty,
            "answer":     result.answer.strip()[:100],
            "top_score":  round(result.retrieval_scores[0], 3),
        })

    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Query':<45} {'Difficulty':<12} {'Top Score':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['query'][:44]:<45} {r['difficulty']:<12} {r['top_score']:<10}")

    print(f"\n[INFO] {len(results)} queries tested.")
    print("[INFO] Check that out-of-scope queries say 'I don't have enough information'.")
    print("[INFO] If they don't — your LLM is hallucinating! That's what Week 4 fixes.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama", help="ollama | huggingface")
    parser.add_argument("--db_path", default="./vectordb")
    parser.add_argument("--quick",   action="store_true", help="Run retrieval tests only (no LLM)")
    args = parser.parse_args()

    print("\n RAG Pipeline - Week 1 Tests")
    print(" Make sure you ran: python ingest.py first!\n")

    try:
        vectorstore = load_retriever(args.db_path)
    except Exception as e:
        print(f"\n[ERROR] Could not load vector DB: {e}")
        print("Make sure you ran:  python ingest.py --docs_dir ./docs --db_path ./vectordb")
        sys.exit(1)

    test_retrieval(vectorstore)
    test_context_builder(vectorstore)

    if not args.quick:
        test_end_to_end(backend=args.backend)
        run_all_queries(backend=args.backend)
    else:
        print("\n[--quick mode] Skipping LLM tests. Run without --quick to test full pipeline.")

    print("\n Week 1 complete! Move on to week2/ for the context quality filter.")