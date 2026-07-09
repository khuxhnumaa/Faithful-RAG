import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))

from dataclasses import dataclass
# from rag_pipeline_p3 import run_rag_w3, RAGResultW3

# To include the correct path:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase3'))
from rag_pipeline_p3 import run_rag_w3, RAGResultW3
from ragas_scorer    import RAGASScorer, RAGASScores
from verifier        import Verifier, VerdictResult


# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH       = "../week1/vectordb"
SIM_THRESHOLD = 0.60


# ── Master result dataclass ───────────────────────────────────────────────────

@dataclass
class RAGResultW4:
    w3_result: RAGResultW3
    scores:    RAGASScores
    result:    VerdictResult

    # ── Convenience properties for the UI ─────────────────────────────────────
    @property
    def query(self):           return self.w3_result.query
    @property
    def answer(self):          return self.result.final_answer
    @property
    def verdict(self):         return self.result.verdict
    @property
    def faithfulness(self):    return self.scores.faithfulness
    @property
    def combined_score(self):  return self.scores.combined
    @property
    def was_fallback(self):    return self.w3_result.was_fallback
    @property
    def sentences(self):       return self.w3_result.detection_result.sentences
    @property
    def hallucinated_sentences(self): return self.result.hallucinated_sentences
    @property
    def uncertain_sentences(self):    return self.result.uncertain_sentences
    @property
    def disclaimer(self):      return self.result.disclaimer
    @property
    def raw_chunks(self):      return len(self.w3_result.w2_result.raw_docs)
    @property
    def kept_after_sim(self):
        return len(self.w3_result.w2_result.after_sim_filter.kept_docs)
    @property
    def kept_after_nli_filter(self):
        return len(self.w3_result.w2_result.after_nli_filter.kept_docs)


# ── Full pipeline function ────────────────────────────────────────────────────

def run_full_pipeline(
    query:            str,
    db_path:          str   = DB_PATH,
    sim_threshold:    float = SIM_THRESHOLD,
    skip_nli_filter:  bool  = False,
    verbose:          bool  = True,
) -> RAGResultW4:
    # ── Weeks 1–3: retrieve, filter, generate, detect ─────────────────────────
    w3_result = run_rag_w3(
        query=query,
        db_path=db_path,
        sim_threshold=sim_threshold,
        skip_nli_filter=skip_nli_filter,
        verbose=verbose,
    )

    # ── Handle fallback (all chunks dropped in Week 2) ────────────────────────
    # If fallback was triggered, there's no real answer to score.
    # Return a zero-score result with hallucinated verdict immediately.
    if w3_result.was_fallback:
        from ragas_scorer import RAGASScores
        from verifier import VerdictResult
        zero_scores = RAGASScores(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_recall=0.0,
            combined=0.0,
        )
        fallback_verdict = VerdictResult(
            query=query,
            verdict="hallucinated",
            scores=zero_scores,
            final_answer=w3_result.answer,
            disclaimer=(
                "No relevant chunks were found in the uploaded document "
                "for this query. The answer could not be grounded."
            ),
        )
        return RAGResultW4(
            w3_result=w3_result,
            scores=zero_scores,
            result=fallback_verdict,
        )

    # ── Week 4a: compute RAGAS scores ─────────────────────────────────────────
    scorer = RAGASScorer()
    scores = scorer.compute(
        query=query,
        answer=w3_result.answer,
        context=w3_result.w2_result.context,
        detection=w3_result.detection_result,
    )

    # ── Week 4b: apply verdict threshold ──────────────────────────────────────
    verifier = Verifier()

    # Define re-query function — called by verifier if verdict is Hallucinated
    def requery_fn(rewritten_query: str) -> RAGResultW4:
        return run_full_pipeline(
            query=rewritten_query,
            db_path=db_path,
            sim_threshold=sim_threshold,
            skip_nli_filter=skip_nli_filter,
            verbose=False,     # quieter on retry
        )

    verdict_result = verifier.verify(
        query=query,
        answer=w3_result.answer,
        scores=scores,
        detection=w3_result.detection_result,
        pipeline_fn=None,    # ← disables re-query completely
    )

    final = RAGResultW4(
        w3_result=w3_result,
        scores=scores,
        result=verdict_result,
    )

    # ── Print final summary ────────────────────────────────────────────────────
    if verbose:
        verdict_result.display()

    return final


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Complete RAG hallucination detection system (Weeks 1-4)"
    )
    parser.add_argument("--query",           required=True,
                        help="Question to ask")
    parser.add_argument("--db_path",         default=DB_PATH,
                        help="Path to FAISS vector DB")
    parser.add_argument("--threshold",       default=SIM_THRESHOLD, type=float,
                        help="Similarity filter threshold")
    parser.add_argument("--skip_nli_filter", action="store_true",
                        help="Skip Week 2 NLI context filter (faster)")
    args = parser.parse_args()

    run_full_pipeline(
        query=args.query,
        db_path=args.db_path,
        sim_threshold=args.threshold,
        skip_nli_filter=args.skip_nli_filter,
    )


if __name__ == "__main__":
    main()