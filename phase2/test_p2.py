

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))
from rag_p2 import run_rag_w2


# ── Test queries ──────────────────────────────────────────────────────────────
# Mix of: in-scope (answerable from doc), out-of-scope (should trigger fallback)

TEST_QUERIES = [
    # ── In-scope: your DCMH paper should answer these ─────────────────────────
    {
        "query":       "What is deep cross-modal hashing?",
        "type":        "in-scope",
        "expect":      "answer from paper",
    },
    {
        "query":       "What loss function does DCMH use?",
        "type":        "in-scope",
        "expect":      "answer from paper",
    },
    {
        "query":       "What datasets were used to evaluate DCMH?",
        "type":        "in-scope",
        "expect":      "answer from paper",
    },

    # ── Out-of-scope: NOT in your paper — LLM should say "I don't know" ───────
    {
        "query":       "Who won the Nobel Prize in 2024?",
        "type":        "out-of-scope",
        "expect":      "fallback / I don't have info",
    },
    {
        "query":       "What is the boiling point of water?",
        "type":        "out-of-scope",
        "expect":      "fallback / I don't have info",
    },
    {
        "query":       "Who is the CEO of Google?",
        "type":        "out-of-scope",
        "expect":      "fallback / I don't have info",
    },

    # ── Tricky: sounds related but isn't in your specific paper ───────────────
    {
        "query":       "What accuracy did DCMH achieve on ImageNet?",
        "type":        "tricky",
        "expect":      "fallback — ImageNet likely not in your paper",
    },
    {
        "query":       "Did the paper compare against GPT-4?",
        "type":        "tricky",
        "expect":      "fallback — GPT-4 not in 2016 paper",
    },
]


# ── Run tests ─────────────────────────────────────────────────────────────────

def run_tests(skip_nli: bool = False, db_path: str = "../week1/vectordb"):
    results = []

    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n\n{'#'*60}")
        print(f"TEST {i}/{len(TEST_QUERIES)} [{test['type'].upper()}]")
        print(f"Expected: {test['expect']}")
        print(f"{'#'*60}")

        result = run_rag_w2(
            query=test["query"],
            db_path=db_path,
            skip_nli=skip_nli,
        )

        results.append({
            "query":        test["query"],
            "type":         test["type"],
            "expect":       test["expect"],
            "raw_chunks":   len(result.raw_docs),
            "after_sim":    len(result.after_sim_filter.kept_docs),
            "after_nli":    len(result.after_nli_filter.kept_docs),
            "fallback":     result.was_fallback,
            "answer":       result.answer.strip()[:120],
        })

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("WEEK 2 FILTER COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Query':<40} {'Type':<12} {'Raw':<5} {'SimF':<5} {'NLIF':<5} {'Fallback'}")
    print("-" * 80)

    for r in results:
        fallback_flag = "YES ✓" if r["fallback"] else "no"
        print(
            f"{r['query'][:39]:<40} "
            f"{r['type']:<12} "
            f"{r['raw_chunks']:<5} "
            f"{r['after_sim']:<5} "
            f"{r['after_nli']:<5} "
            f"{fallback_flag}"
        )

    # ── Stats ─────────────────────────────────────────────────────────────────
    out_of_scope = [r for r in results if r["type"] in ("out-of-scope", "tricky")]
    caught       = [r for r in out_of_scope if r["fallback"]]

    print(f"\n{'='*80}")
    print(f"Out-of-scope / tricky queries:  {len(out_of_scope)}")
    print(f"Correctly returned fallback:    {len(caught)}")
    print(f"Hallucination prevention rate:  {len(caught)/len(out_of_scope)*100:.0f}%")
    print(f"\nNOTE: Queries where fallback=NO but type=out-of-scope are hallucinations")
    print(f"      the filters MISSED. These become your Week 3 detection targets.")
    print(f"{'='*80}\n")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_nli", action="store_true", help="Run similarity filter only")
    parser.add_argument("--db_path",  default="../week1/vectordb")
    args = parser.parse_args()

    print("\n Week 2 Test Suite")
    print(" Make sure ollama serve is running in another terminal!\n")

    run_tests(skip_nli=args.skip_nli, db_path=args.db_path)