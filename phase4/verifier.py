from dataclasses import dataclass, field
from typing import Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))

from ragas_scorer import RAGASScores
# ── Verdict thresholds ────────────────────────────────────────────────────────

THRESHOLDS = {
    "grounded":     0.60,
    "borderline":   0.50,   # anything below this is "hallucinated"
}

VERDICT_LABELS = {
    "grounded":     "✓ GROUNDED",
    "borderline":   "⚠ BORDERLINE",
    "hallucinated": "✗ HALLUCINATED",
}

VERDICT_COLORS = {
    "grounded":     "green",
    "borderline":   "orange",
    "hallucinated": "red",
}

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VerdictResult:
    query:                str
    verdict:              str
    scores:               RAGASScores
    final_answer:         str
    hallucinated_sentences: list = field(default_factory=list)
    uncertain_sentences:    list = field(default_factory=list)
    requery_attempted:    bool = False
    requery_succeeded:    bool = False
    disclaimer:           str  = ""

    def display(self):
        label = VERDICT_LABELS.get(self.verdict, self.verdict)
        print(f"\n{'='*65}")
        print(f"FINAL VERDICT: {label}")
        print(f"{'='*65}")
        print(f"Query:            {self.query}")
        print(f"Combined score:   {self.scores.combined:.4f}")
        print(f"Faithfulness:     {self.scores.faithfulness:.4f}")
        print(f"Answer relevance: {self.scores.answer_relevance:.4f}")
        print(f"Context recall:   {self.scores.context_recall:.4f}")
        if self.requery_attempted:
            status = "succeeded" if self.requery_succeeded else "also failed"
            print(f"Re-query:         attempted → {status}")
        if self.hallucinated_sentences:
            print(f"\nHallucinated sentences ({len(self.hallucinated_sentences)}):")
            for s in self.hallucinated_sentences:
                print(f"  ✗ {s}")
        if self.uncertain_sentences:
            print(f"\nUncertain sentences ({len(self.uncertain_sentences)}):")
            for s in self.uncertain_sentences:
                print(f"  ⚠ {s}")
        if self.disclaimer:
            print(f"\nDisclaimer: {self.disclaimer}")
        print(f"\nAnswer shown to user:")
        print(f"  {self.final_answer[:200]}{'...' if len(self.final_answer)>200 else ''}")
        print(f"{'='*65}\n")



# ── Verdict logic ─────────────────────────────────────────────────────────────

def apply_threshold(combined_score: float, thresholds: dict) -> str:
    if combined_score >= thresholds["grounded"]:
        return "grounded"
    elif combined_score >= thresholds["borderline"]:
        return "borderline"
    else:
        return "hallucinated"


def build_disclaimer(verdict: str, scores: RAGASScores) -> str:
    if verdict == "borderline":
        return (
            f"This answer may be partially incomplete or uncertain. "
            f"Faithfulness score: {scores.faithfulness:.2f}. "
            f"Please verify against the source document."
        )
    return ""


# ── Verifier ──────────────────────────────────────────────────────────────────

# class Verifier:
#     def __init__(self, thresholds: dict = None):
#         self.thresholds = thresholds or THRESHOLDS

#     def verify(self, query, answer, scores, detection,
#                pipeline_fn=None, is_retry=False):

#         verdict = apply_threshold(scores.combined, self.thresholds)
CONTEXT_RELEVANCE_GATE = 0.35  # tune this using your 70 examples first

class Verifier:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or THRESHOLDS

    def verify(self, query, answer, scores, detection,
               pipeline_fn=None, is_retry=False):

        # ── Hard gate: was the retrieved context even about this topic? ──────
        if scores.context_relevance < CONTEXT_RELEVANCE_GATE:
            return VerdictResult(
                query=query,
                verdict="hallucinated",
                scores=scores,
                final_answer=(
                    "I could not find relevant information in the provided "
                    "document for this query."
                ),
                hallucinated_sentences=detection.hallucinated_sentences,
                uncertain_sentences=detection.uncertain_sentences,
                disclaimer="Retrieved context was not sufficiently relevant to the query.",
            )

        verdict = apply_threshold(scores.combined, self.thresholds)

        # ── Grounded: return immediately, no retry logic involved ────────────
        if verdict == "grounded":
            return VerdictResult(
                query=query,
                verdict="grounded",
                scores=scores,
                final_answer=answer,
                hallucinated_sentences=detection.hallucinated_sentences,
                uncertain_sentences=detection.uncertain_sentences,
            )

        # ── Borderline: return immediately, no retry logic involved ──────────
        if verdict == "borderline":
            return VerdictResult(
                query=query,
                verdict="borderline",
                scores=scores,
                final_answer=answer,
                hallucinated_sentences=detection.hallucinated_sentences,
                uncertain_sentences=detection.uncertain_sentences,
                disclaimer=build_disclaimer(verdict, scores),
            )

        # ── Hallucinated: try a re-query once, unless this is already a retry ─
        print(f"\n[verifier] Verdict is HALLUCINATED.")

        if is_retry or pipeline_fn is None:
            return VerdictResult(
                query=query,
                verdict="hallucinated",
                scores=scores,
                final_answer=(
                    "I could not generate a reliable answer grounded in the "
                    "provided document for this query. Please rephrase your "
                    "question or check that the relevant information is in "
                    "the uploaded document."
                ),
                hallucinated_sentences=detection.hallucinated_sentences,
                uncertain_sentences=detection.uncertain_sentences,
                requery_attempted=is_retry,
                requery_succeeded=False,
            )

        rewritten_query = query
        print(f"[verifier] Retrying once with original query ...")

        try:
            retry_result = pipeline_fn(rewritten_query)

            if retry_result.scores.combined >= self.thresholds["borderline"]:
                print(f"[verifier] Re-query succeeded! Score: {retry_result.scores.combined:.4f}")
                retry_result.result.requery_attempted = True
                retry_result.result.requery_succeeded = True
                return retry_result.result
            else:
                print(f"[verifier] Re-query also failed. Score: {retry_result.scores.combined:.4f}")
                return VerdictResult(
                    query=query,
                    verdict="hallucinated",
                    scores=scores,
                    final_answer=(
                        "I could not generate a reliable answer grounded in the "
                        "provided document for this query. Please rephrase your "
                        "question or verify that the relevant document is uploaded."
                    ),
                    hallucinated_sentences=detection.hallucinated_sentences,
                    uncertain_sentences=detection.uncertain_sentences,
                    requery_attempted=True,
                    requery_succeeded=False,
                )

        except Exception as e:
            print(f"[verifier] Re-query error: {e}")
            return VerdictResult(
                query=query,
                verdict="hallucinated",
                scores=scores,
                final_answer=(
                    "I could not generate a reliable answer for this query "
                    "from the provided document."
                ),
                hallucinated_sentences=detection.hallucinated_sentences,
                uncertain_sentences=detection.uncertain_sentences,
                requery_attempted=True,
                requery_succeeded=False,
            )