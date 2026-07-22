import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))

import argparse


# ── Test 1: Sentence splitter ─────────────────────────────────────────────────

def test_sentence_splitter():
    from sentence_splitter import split_into_sentences

    print("\n" + "="*60)
    print("TEST 1: Sentence splitter")
    print("="*60)

    test_cases = [
        {
            "text": (
                "DCMH is a deep cross-modal hashing method. "
                "It was proposed in 2016 by Jiang et al. "
                "The method achieves state-of-the-art results on NUS-WIDE dataset. "
                "It uses a joint learning framework to preserve semantic similarity."
            ),
            "expected_count": 4,
        },
        {
            "text": (
                "The model achieved 87.3% accuracy on the test set. "
                "This is compared to 72.1% for the baseline. "
                "The improvement is statistically significant with p < 0.05."
            ),
            "expected_count": 3,
        },
        {
            "text": "Short answer.",   # too short, should be filtered
            "expected_count": 0,
        },
        # {
        #     "text":("Hello my Name is Dr. kim"
        #             "Today, prof. henry's class scheduled at 3pm "
        #             "So delete this sentences"),
        #     "expected_count": 0,
        # }
    ]

    all_passed = True
    for i, tc in enumerate(test_cases, 1):
        sentences = split_into_sentences(tc["text"])
        passed = len(sentences) == tc["expected_count"]
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"\n  Test {i}: [{status}] got {len(sentences)} sentences "
              f"(expected {tc['expected_count']})")
        for j, s in enumerate(sentences, 1):
            print(f"    [{j}] {s[:80]}...")

    print(f"\n[splitter] {'All tests passed!' if all_passed else 'Some tests FAILED'}")
    return all_passed


# ── Test 2: NLI detector on constructed examples ──────────────────────────────

def test_nli_detector_constructed():
    from nli_detector import NLIDetector

    print("\n" + "="*60)
    print("TEST 2: NLI detector — constructed hallucination example")
    print("="*60)

    # This is a controlled test — we know exactly what the answer says
    # and we know which sentence is hallucinated
    context = """
    DCMH (Deep Cross-Modal Hashing) is a method for cross-modal retrieval.
    It was proposed by Jiang et al. in 2017.
    The method uses deep neural networks to learn hash codes.
    Experiments were conducted on the NUS-WIDE and MIRFlickr datasets.
    The model achieved 78.5% mAP on the NUS-WIDE dataset.
    """

    # Answer with one hallucinated sentence (wrong accuracy number)
    answer_with_hallucination = (
        "DCMH is a deep cross-modal hashing method proposed by Jiang et al. "
        "The method uses deep neural networks to learn compact hash codes. "
        "It achieved 97% accuracy on the ImageNet benchmark dataset. "   # ← hallucinated
        "Experiments were performed on the NUS-WIDE dataset."
    )

    print(f"\nContext (ground truth):\n{context.strip()}")
    print(f"\nAnswer being checked:\n{answer_with_hallucination}")
    print(f"\nExpected: sentence 3 should be flagged as hallucinated or uncertain")

    detector = NLIDetector()
    result = detector.detect(answer_with_hallucination, context)
    result.print_full()

    print(f"Faithfulness score: {result.faithfulness_score}")
    print(f"(Perfect score would be 1.0 — lower = more hallucination detected)")
    return result


# ── Test 3: Full pipeline queries ─────────────────────────────────────────────

TEST_QUERIES = [
    # In-scope: answerable from your DCMH paper
    {
        "query":  "What is deep cross-modal hashing?",
        "type":   "in-scope",
        "note":   "Should get high faithfulness — answer comes from paper",
    },
    {
        "query":  "What datasets were used to evaluate the method?",
        "type":   "in-scope",
        "note":   "Specific fact from paper — should be grounded",
    },
    # Out-of-scope: not in your paper
    {
        "query":  "Who won the Nobel Prize in Physics in 2024?",
        "type":   "out-of-scope",
        "note":   "Should trigger fallback — never reaches detector",
    },
    {
        "query":  "What is the capital of France?",
        "type":   "out-of-scope",
        "note":   "Should trigger fallback — clearly not in paper",
    },
    # Tricky: sounds relevant but details may not match
    {
        "query":  "Did the paper achieve 99% accuracy?",
        "type":   "tricky",
        "note":   "LLM may say yes even if wrong — detector should catch it",
    },
]


def run_pipeline_tests(
    db_path: str = "../week1/vectordb",
    skip_nli_filter: bool = False,
):
    from rag_pipeline_p3 import run_rag_w3

    print("\n" + "="*60)
    print("TEST 3: Full Week 3 pipeline")
    print("="*60)
    print("Make sure 'ollama serve' is running in another terminal!\n")

    results = []
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n\n{'#'*60}")
        print(f"QUERY {i}/{len(TEST_QUERIES)} [{test['type'].upper()}]")
        print(f"Note: {test['note']}")
        print(f"{'#'*60}")

        result = run_rag_w3(
            query=test["query"],
            db_path=db_path,
            skip_nli_filter=skip_nli_filter,
            verbose=True,
        )

        results.append({
            "query":          test["query"],
            "type":           test["type"],
            "fallback":       result.was_fallback,
            "faithfulness":   result.faithfulness,
            "total_sent":     result.detection_result.total,
            "grounded":       result.detection_result.grounded_count,
            "uncertain":      result.detection_result.uncertain_count,
            "hallucinated":   result.detection_result.hallucinated_count,
        })

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("WEEK 3 DETECTION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Query':<38} {'Type':<12} {'Faith':<7} {'G':<4} {'U':<4} {'H':<4} {'Fallback'}")
    print("-"*80)

    for r in results:
        print(
            f"{r['query'][:37]:<38} "
            f"{r['type']:<12} "
            f"{r['faithfulness']:<7} "
            f"{r['grounded']:<4} "
            f"{r['uncertain']:<4} "
            f"{r['hallucinated']:<4} "
            f"{'YES' if r['fallback'] else 'no'}"
        )

    print(f"\nG=grounded  U=uncertain  H=hallucinated  Faith=faithfulness score")
    print(f"\nWhat to look for:")
    print(f"  In-scope queries:    faithfulness close to 1.0, low H count")
    print(f"  Out-of-scope:        fallback=YES (never reaches detector)")
    print(f"  Tricky queries:      faithfulness low, H count > 0 = hallucination CAUGHT")
    print(f"{'='*80}\n")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",           action="store_true",
                        help="Run splitter test only — no LLM needed")
    parser.add_argument("--skip_nli_filter", action="store_true",
                        help="Skip Week 2 NLI filter for speed")
    parser.add_argument("--db_path",         default="../week1/vectordb")
    args = parser.parse_args()

    print("\n Week 3 Test Suite — Prevention + Detection")
    print(" ─────────────────────────────────────────────")

    # Always run sentence splitter test (no model needed)
    test_sentence_splitter()

    if args.quick:
        print("\n[--quick mode] Skipping LLM and NLI tests.")
        print("Run without --quick to test the full pipeline.")
    else:
        # Run NLI detector on constructed example first
        test_nli_detector_constructed()
        # Run full pipeline tests
        run_pipeline_tests(
            db_path=args.db_path,
            skip_nli_filter=args.skip_nli_filter,
        )