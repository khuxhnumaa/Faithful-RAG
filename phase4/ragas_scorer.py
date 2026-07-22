from dataclasses import dataclass
from typing import List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week3'))
from nli_detector import DetectionResult


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Weights for the combined score
WEIGHT_FAITHFULNESS = 0.25
WEIGHT_RELEVANCE    = 0.4
WEIGHT_RECALL       = 0.35


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class RAGASScores:
    faithfulness:      float
    answer_relevance:  float
    context_recall:    float
    context_relevance: float   # ← new
    combined:          float

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
        score = detection.faithfulness_score
        print(f"[ragas] Faithfulness: {score:.4f} "
              f"({detection.grounded_count}/{detection.total} sentences grounded)")
        return score

    # ── Metric 2: Answer relevance ────────────────────────────────────────────

    def compute_answer_relevance(self, query: str, answer: str) -> float:
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
    # ── Metric 4: Context relevance (query vs retrieved context) ─────────────

    def compute_context_relevance(self, query: str, context: str) -> float:
        """Is the retrieved context actually about the query topic?"""
        if not context.strip():
            return 0.0
        embedder = self._load_embedder()
        import numpy as np
        q_vec = embedder.encode(query, normalize_embeddings=True)
        chunks = [c.strip() for c in context.split("---") if c.strip()] or [context]
        ctx_vecs = embedder.encode(chunks, normalize_embeddings=True)
        sims = [float(np.dot(q_vec, c)) for c in ctx_vecs]
        score = round(max(sims), 4) if sims else 0.0
        print(f"[ragas] Context relevance: {score:.4f} (query-context cosine similarity)")
        return score

    # ── Combined score ────────────────────────────────────────────────────────

    # def compute(
    #     self,
    #     query:     str,
    #     answer:    str,
    #     context:   str,
    #     detection: DetectionResult,
    # ) -> RAGASScores:
    #     print(f"\n[ragas] Computing RAGAS metrics ...")

    #     f = self.compute_faithfulness(detection)
    #     r = self.compute_answer_relevance(query, answer)
    #     c = self.compute_context_recall(detection, context)

    #     combined = round(
    #         WEIGHT_FAITHFULNESS * f +
    #         WEIGHT_RELEVANCE    * r +
    #         WEIGHT_RECALL       * c,
    #         4
    #     )
    #     print(f"[ragas] Combined score: {combined:.4f} "
    #           f"(w={WEIGHT_FAITHFULNESS}/{WEIGHT_RELEVANCE}/{WEIGHT_RECALL})")

    #     scores = RAGASScores(
    #         faithfulness=f,
    #         answer_relevance=r,
    #         context_recall=c,
    #         combined=combined,
    #     )
    #     scores.display()
    #     return scores
    def compute(self, query, answer, context, detection) -> RAGASScores:
        print(f"\n[ragas] Computing RAGAS metrics ...")

        f  = self.compute_faithfulness(detection)
        r  = self.compute_answer_relevance(query, answer)
        c  = self.compute_context_recall(detection, context)
        cr = self.compute_context_relevance(query, context)   # ← new

        combined = round(
            WEIGHT_FAITHFULNESS * f +
            WEIGHT_RELEVANCE    * r +
            WEIGHT_RECALL       * c,
            4
        )

        scores = RAGASScores(
            faithfulness=f,
            answer_relevance=r,
            context_recall=c,
            context_relevance=cr,   # ← new field
            combined=combined,
        )
        scores.display()
        return scores