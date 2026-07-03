"""
Week 4 - File 1: ragas_scorer.py
----------------------------------
WHAT THIS FILE DOES:
    Takes the sentence-level NLI results from Week 3 and computes
    three RAGAS metrics that together measure how good and how honest
    the answer is.

    Three metrics computed here:

    1. FAITHFULNESS SCORE
       Formula: grounded_sentences / total_sentences
       Meaning: what fraction of the answer is actually supported by
                the retrieved context?
       Range:   0.0 (everything hallucinated) to 1.0 (everything grounded)
       Source:  comes directly from Week 3's NLI detector output

    2. ANSWER RELEVANCE SCORE
       Meaning: does the answer actually address what the user asked?
                A faithful answer can still be off-topic.
       How:     embeds both the query and the answer, computes cosine
                similarity between them.
       Range:   0.0 (completely off-topic) to 1.0 (directly answers query)

    3. CONTEXT RECALL SCORE
       Meaning: did the retrieved chunks contain enough information
                to answer the question fully?
       How:     checks how many sentences in the answer can be traced
                back to at least one retrieved chunk.
       Range:   0.0 (retrieval missed everything) to 1.0 (retrieval complete)

    COMBINED SCORE:
       Weighted average of all three.
       Default weights: faithfulness=0.5, relevance=0.3, recall=0.2
       Faithfulness gets highest weight because it's the hallucination signal.

WHY NO PAID API:
    Original RAGAS uses GPT-4 internally to judge faithfulness.
    This implementation replaces that with the NLI scores from Week 3
    and sentence-transformers for embeddings — both free and local.

Paper reference:
    RAGAS: Automated Evaluation of Retrieval Augmented Generation
    Es et al., 2023. arxiv.org/abs/2309.15217
"""

from dataclasses import dataclass
from typing import List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week3'))
from nli_detector import DetectionResult


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Weights for the combined score
WEIGHT_FAITHFULNESS = 0.5
WEIGHT_RELEVANCE    = 0.3
WEIGHT_RECALL       = 0.2


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class RAGASScores:
    """
    Stores all three RAGAS metrics plus the combined score.
    This is what gets passed to the verifier in verifier.py.
    """
    faithfulness:    float   # from NLI detection (Week 3)
    answer_relevance: float  # answer vs query similarity
    context_recall:  float   # how much of answer is traceable to context
    combined:        float   # weighted average of all three

    def display(self):
        print(f"\n[RAGAS SCORES]")
        print(f"  Faithfulness:     {self.faithfulness:.4f}  "
              f"(weight={WEIGHT_FAITHFULNESS}) — is answer grounded in context?")
        print(f"  Answer relevance: {self.answer_relevance:.4f}  "
              f"(weight={WEIGHT_RELEVANCE}) — does answer address the query?")
        print(f"  Context recall:   {self.context_recall:.4f}  "
              f"(weight={WEIGHT_RECALL}) — did retrieval find enough info?")
        print(f"  Combined score:   {self.combined:.4f}")


# ── Scorer ────────────────────────────────────────────────────────────────────

class RAGASScorer:
    """
    Computes RAGAS-inspired metrics using local free tools.
    No OpenAI key. No paid API.

    Uses:
      - Week 3 NLI detection output for faithfulness
      - sentence-transformers for answer relevance (cosine similarity)
      - sentence-level overlap for context recall
    """

    def __init__(self):
        self._embedder = None   # lazy load

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[ragas] Loading embedding model: {EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    # ── Metric 1: Faithfulness ────────────────────────────────────────────────

    def compute_faithfulness(self, detection: DetectionResult) -> float:
        """
        Directly uses the NLI-based faithfulness score from Week 3.
        Formula: grounded_sentences / total_sentences
        """
        score = detection.faithfulness_score
        print(f"[ragas] Faithfulness: {score:.4f} "
              f"({detection.grounded_count}/{detection.total} sentences grounded)")
        return score

    # ── Metric 2: Answer relevance ────────────────────────────────────────────

    def compute_answer_relevance(self, query: str, answer: str) -> float:
        """
        Measures how well the answer addresses the query.
        Uses cosine similarity between query and answer embeddings.

        Even if an answer is fully faithful to context, it might not
        actually answer what was asked. This metric catches that.
        """
        if not answer or not answer.strip():
            return 0.0

        embedder = self._load_embedder()
        import numpy as np

        q_vec = embedder.encode(query,  normalize_embeddings=True)
        a_vec = embedder.encode(answer, normalize_embeddings=True)

        # Cosine similarity — vectors are already normalized so just dot product
        score = float(np.dot(q_vec, a_vec))
        score = max(0.0, min(1.0, score))   # clamp to [0, 1]

        print(f"[ragas] Answer relevance: {score:.4f} "
              f"(query-answer cosine similarity)")
        return round(score, 4)

    # ── Metric 3: Context recall ──────────────────────────────────────────────

    def compute_context_recall(
        self,
        detection: DetectionResult,
        context: str
    ) -> float:
        """
        Measures how much of the answer is traceable to the context.

        Logic: for each sentence in the answer, check if any part of
        the context contains semantically similar content.
        Uses embedding similarity per sentence vs context chunks.

        A low score means retrieval didn't find enough relevant info
        — the LLM had to fill gaps from its own knowledge.
        """
        from sentence_splitter import split_into_sentences
        import numpy as np

        sentences = [s.sentence for s in detection.sentences]
        if not sentences or not context.strip():
            return 0.0

        embedder = self._load_embedder()

        # Split context into chunks for comparison
        context_chunks = [c.strip() for c in context.split("---") if c.strip()]
        if not context_chunks:
            context_chunks = [context]

        # Embed all context chunks
        ctx_vecs = embedder.encode(context_chunks, normalize_embeddings=True)

        recalled = 0
        for sentence in sentences:
            s_vec = embedder.encode(sentence, normalize_embeddings=True)
            # Similarity to each context chunk
            sims = [float(np.dot(s_vec, c)) for c in ctx_vecs]
            max_sim = max(sims) if sims else 0.0
            # If any context chunk is > 0.5 similar → sentence is recalled
            if max_sim > 0.50:
                recalled += 1

        score = round(recalled / len(sentences), 4) if sentences else 0.0
        print(f"[ragas] Context recall: {score:.4f} "
              f"({recalled}/{len(sentences)} sentences traceable to context)")
        return score

    # ── Combined score ────────────────────────────────────────────────────────

    def compute(
        self,
        query:     str,
        answer:    str,
        context:   str,
        detection: DetectionResult,
    ) -> RAGASScores:
        """
        Compute all three RAGAS metrics and return combined score.

        Args:
            query:     original user question
            answer:    LLM generated answer
            context:   retrieved context chunks (combined string)
            detection: Week 3 NLI DetectionResult

        Returns:
            RAGASScores with all metrics
        """
        print(f"\n[ragas] Computing RAGAS metrics ...")

        f = self.compute_faithfulness(detection)
        r = self.compute_answer_relevance(query, answer)
        c = self.compute_context_recall(detection, context)

        combined = round(
            WEIGHT_FAITHFULNESS * f +
            WEIGHT_RELEVANCE    * r +
            WEIGHT_RECALL       * c,
            4
        )
        print(f"[ragas] Combined score: {combined:.4f} "
              f"(w={WEIGHT_FAITHFULNESS}/{WEIGHT_RELEVANCE}/{WEIGHT_RECALL})")

        scores = RAGASScores(
            faithfulness=f,
            answer_relevance=r,
            context_recall=c,
            combined=combined,
        )
        scores.display()
        return scores