"""
Week 4 - File 2: verifier.py
-------------------------------
WHAT THIS FILE DOES:
    This is the final decision-making layer of the entire system.
    It takes the RAGAS combined score and applies threshold logic
    to produce a human-readable verdict: Grounded, Borderline,
    or Hallucinated.

    It also handles the AUTO RE-QUERY logic. When a verdict is
    Hallucinated, the system doesn't just give up. It automatically
    rewrites the query to be more specific, runs the entire pipeline
    again, and checks the new answer. If the second attempt passes,
    it returns that answer instead. If it still fails, it returns
    a safe "cannot answer" response.

    VERDICT THRESHOLDS:
      combined score ≥ 0.75  →  GROUNDED    (answer is trustworthy)
      combined score 0.50–0.75 → BORDERLINE  (answer shown with warning)
      combined score < 0.50   →  HALLUCINATED (answer rejected, re-query)

    WHY THESE NUMBERS:
      0.75 is used (not 0.85 as discussed earlier) because the combined
      score already blends faithfulness, relevance, and recall.
      Faithfulness alone uses 0.85 internally. The blended combined
      score naturally sits lower since relevance and recall add noise.
      You can tune these in the THRESHOLDS dict below.

    RE-QUERY STRATEGY:
      When rejected, the query is rewritten by appending:
      "Answer strictly based on the provided document context only."
      This forces the LLM to stay grounded rather than using its own
      training knowledge. One retry is allowed. No infinite loops.
"""

from dataclasses import dataclass, field
from typing import Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))

from ragas_scorer import RAGASScores


# ── Verdict thresholds ────────────────────────────────────────────────────────

THRESHOLDS = {
    "grounded":     0.75,   # combined ≥ 0.75 → Grounded
    "borderline":   0.50,   # combined 0.50–0.75 → Borderline
    # below 0.50 → Hallucinated
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
    """
    Final output of the entire Week 4 system for one query.

    Contains:
      - the verdict (grounded / borderline / hallucinated)
      - RAGAS scores
      - the answer shown to the user
      - which sentences were flagged
      - whether a re-query was attempted and succeeded
    """
    query:                str
    verdict:              str             # grounded / borderline / hallucinated
    scores:               RAGASScores
    final_answer:         str             # answer shown to user
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

def apply_threshold(combined_score: float) -> str:
    """
    Convert a numeric combined score into a verdict string.

    Args:
        combined_score: float in [0, 1] from RAGASScorer

    Returns:
        "grounded" | "borderline" | "hallucinated"
    """
    if combined_score >= THRESHOLDS["grounded"]:
        return "grounded"
    elif combined_score >= THRESHOLDS["borderline"]:
        return "borderline"
    else:
        return "hallucinated"


def build_disclaimer(verdict: str, scores: RAGASScores) -> str:
    """
    Build a user-facing disclaimer for borderline answers.
    Not shown for grounded answers. Not needed for hallucinated
    (those get re-queried or rejected entirely).
    """
    if verdict == "borderline":
        return (
            f"This answer may be partially incomplete or uncertain. "
            f"Faithfulness score: {scores.faithfulness:.2f}. "
            f"Please verify against the source document."
        )
    return ""


# ── Verifier ──────────────────────────────────────────────────────────────────

class Verifier:
    """
    Final layer of the hallucination detection system.

    Takes RAGAS scores + detection results and:
      1. Applies threshold to get verdict
      2. On Hallucinated: rewrites query and retries once
      3. Returns VerdictResult for the UI (Week 5)
    """

    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or THRESHOLDS

    def verify(
        self,
        query:      str,
        answer:     str,
        scores:     RAGASScores,
        detection,              # DetectionResult from Week 3
        pipeline_fn = None,     # callable: run_full_pipeline(query) → RAGResultW4
    ) -> VerdictResult:
        """
        Apply threshold, handle re-query if hallucinated.

        Args:
            query:       original user query
            answer:      LLM generated answer
            scores:      RAGASScores from ragas_scorer.py
            detection:   DetectionResult from nli_detector.py
            pipeline_fn: function to call for re-query (injected from pipeline)

        Returns:
            VerdictResult — the final output sent to the UI
        """
        verdict = apply_threshold(scores.combined)
        print(f"\n[verifier] Score={scores.combined:.4f} → verdict={verdict.upper()}")

        # ── Grounded: return answer as-is ─────────────────────────────────────
        if verdict == "grounded":
            return VerdictResult(
                query=query,
                verdict="grounded",
                scores=scores,
                final_answer=answer,
                hallucinated_sentences=detection.hallucinated_sentences,
                uncertain_sentences=detection.uncertain_sentences,
            )

        # ── Borderline: return answer with disclaimer ──────────────────────────
        if verdict == "borderline":
            disclaimer = build_disclaimer(verdict, scores)
            return VerdictResult(
                query=query,
                verdict="borderline",
                scores=scores,
                final_answer=answer,
                hallucinated_sentences=detection.hallucinated_sentences,
                uncertain_sentences=detection.uncertain_sentences,
                disclaimer=disclaimer,
            )

        # ── Hallucinated: attempt re-query ────────────────────────────────────
        print(f"\n[verifier] Verdict is HALLUCINATED. Attempting re-query ...")

        if pipeline_fn is None:
            # No pipeline injected — return safe fallback
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
                requery_attempted=False,
            )

        # ── Rewrite query and retry ────────────────────────────────────────────
        rewritten_query = (
            f"{query} "
            f"Answer strictly based on the provided document context only. "
            f"Do not use any external knowledge."
        )
        print(f"[verifier] Rewritten query: {rewritten_query[:100]}...")

        try:
            retry_result = pipeline_fn(rewritten_query)

            if retry_result.scores.combined >= self.thresholds["borderline"]:
                # Retry succeeded
                print(f"[verifier] Re-query succeeded! "
                      f"New score: {retry_result.scores.combined:.4f}")
                retry_result.result.requery_attempted = True
                retry_result.result.requery_succeeded = True
                return retry_result.result
            else:
                # Retry also failed
                print(f"[verifier] Re-query also failed. "
                      f"Score: {retry_result.scores.combined:.4f}")
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