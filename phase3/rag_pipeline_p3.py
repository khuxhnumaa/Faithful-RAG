import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))

import argparse
from dataclasses import dataclass

from rag_p2 import run_rag_w2, RAGResultW2
# from phase2.rag_p2 import run_rag_w2, RAGResultW2
from nli_detector import NLIDetector, DetectionResult


# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH        = "../week1/vectordb"
# DB_PATH = "./vectordb"
SIM_THRESHOLD  = 0.60   # using the relaxed threshold after Week 2 fix
OLLAMA_MODEL   = "llama3.2:3b"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RAGResultW3:
    w2_result:        RAGResultW2
    detection_result: DetectionResult

    @property
    def query(self):
        return self.w2_result.query

    @property
    def answer(self):
        return self.w2_result.answer

    @property
    def faithfulness(self):
        return self.detection_result.faithfulness_score

    @property
    def was_fallback(self):
        return self.w2_result.was_fallback


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_rag_w3(
    query:         str,
    db_path:       str   = DB_PATH,
    sim_threshold: float = SIM_THRESHOLD,
    skip_nli_filter: bool = False,   # skip Week 2 NLI filter (not detector)
    verbose:       bool  = True,
) -> RAGResultW3:
    print(f"\n{'#'*60}")
    print(f"PHASE 1: RAG pipeline with prevention filters (Week 2)")
    print(f"{'#'*60}")

    w2_result = run_rag_w2(
        query=query,
        db_path=db_path,
        sim_threshold=sim_threshold,
        skip_nli=skip_nli_filter,
        top_k=6,
    )

    # ── Phase 2: Detection (Week 3) ───────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"PHASE 2: Sentence-level NLI detection (Week 3)")
    print(f"{'#'*60}")

    # If fallback was used (all chunks dropped), skip detection
    # — there's nothing to detect, answer is already the safe fallback
    if w2_result.was_fallback:
        print("[w3] Fallback was used — skipping detection (no LLM answer to check).")
        from nli_detector import DetectionResult
        detection_result = DetectionResult(
            answer=w2_result.answer,
            context="",
        )
        return RAGResultW3(
            w2_result=w2_result,
            detection_result=detection_result,
        )

    # Run NLI sentence-level detection on the generated answer
    detector = NLIDetector()
    detection_result = detector.detect(
        answer=w2_result.answer,
        context=w2_result.context,
        verbose=verbose,
    )

    # Print full per-sentence breakdown
    if verbose:
        detection_result.print_full()

    result = RAGResultW3(
        w2_result=w2_result,
        detection_result=detection_result,
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"WEEK 3 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Query:             {query}")
    print(f"Answer:            {w2_result.answer.strip()[:100]}...")
    print(f"Faithfulness:      {result.faithfulness}")
    print(f"Sentences:         {detection_result.total}")
    print(f"  Grounded:        {detection_result.grounded_count}")
    print(f"  Uncertain:       {detection_result.uncertain_count}")
    print(f"  Hallucinated:    {detection_result.hallucinated_count}")
    if detection_result.hallucinated_sentences:
        print(f"\nFlagged sentences:")
        for s in detection_result.hallucinated_sentences:
            print(f"  ✗ {s}")
    print(f"{'='*60}\n")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Week 3 RAG pipeline with detection")
    parser.add_argument("--query",            required=True)
    parser.add_argument("--db_path",          default=DB_PATH)
    parser.add_argument("--threshold",        default=SIM_THRESHOLD, type=float)
    parser.add_argument("--skip_nli_filter",  action="store_true",
                        help="Skip Week 2 NLI filter (not the detector)")
    args = parser.parse_args()

    run_rag_w3(
        query=args.query,
        db_path=args.db_path,
        sim_threshold=args.threshold,
        skip_nli_filter=args.skip_nli_filter,
    )


if __name__ == "__main__":
    main()