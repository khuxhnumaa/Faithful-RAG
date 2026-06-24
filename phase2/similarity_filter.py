"""
Filters retrieved chunks based on cosine similarity score and Chunks below the threshold are dropped before reaching the LLM.
This is the PREVENTION layer — stops bad context entering the LLM.
"""

from dataclasses import dataclass
from typing import List, Tuple
from langchain_core.documents import Document


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.60  

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    kept_docs:    List[Document]   # chunks that passed the filter
    kept_scores:  List[float]
    dropped_docs: List[Document]   # chunks that were removed
    dropped_scores: List[float]
    threshold:    float

    def summary(self) -> str:
        total = len(self.kept_docs) + len(self.dropped_docs)
        return (
            f"[similarity_filter] threshold={self.threshold} | "
            f"kept={len(self.kept_docs)}/{total} | "
            f"dropped={len(self.dropped_docs)}/{total}"
        )


# ── Filter ────────────────────────────────────────────────────────────────────

class SimilarityFilter:
    """ Drops retrieved chunks whose cosine similarity score
    falls below the threshold. """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def filter(
        self,
        docs: List[Document],
        scores: List[float]
    ) -> FilterResult:
        """
        Filter docs by similarity score.

        Args:
            docs:   List of retrieved LangChain Documents
            scores: Corresponding cosine similarity scores [0, 1]

        Returns:
            FilterResult with kept and dropped chunks
        """
        kept_docs, kept_scores     = [], []
        dropped_docs, dropped_scores = [], []

        for doc, score in zip(docs, scores):
            src     = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:60].replace("\n", " ")

            if score >= self.threshold:
                kept_docs.append(doc)
                kept_scores.append(score)
                print(f"  [KEEP  ] score={score:.3f} ≥ {self.threshold} | {src} | {preview}...")
            else:
                dropped_docs.append(doc)
                dropped_scores.append(score)
                print(f"  [DROP  ] score={score:.3f} < {self.threshold} | {src} | {preview}...")

        result = FilterResult(
            kept_docs=kept_docs,
            kept_scores=kept_scores,
            dropped_docs=dropped_docs,
            dropped_scores=dropped_scores,
            threshold=self.threshold,
        )
        print(result.summary())
        return result

    def tune_threshold(
        self,
        docs: List[Document],
        scores: List[float],
        thresholds: List[float] = [0.50, 0.60, 0.65, 0.72, 0.80, 0.85]
    ):
        """
        Helper: show how many chunks survive at each threshold value.
        Use this to pick the right threshold for your dataset.
        """
        print("\n[similarity_filter] Threshold tuning:")
        print(f"  {'Threshold':<12} {'Kept':<8} {'Dropped':<10} Kept chunks")
        print("  " + "-" * 50)
        for t in thresholds:
            kept = [(d, s) for d, s in zip(docs, scores) if s >= t]
            dropped = len(scores) - len(kept)
            srcs = [d.metadata.get("source", "?")[-20:] for d, _ in kept[:3]]
            print(f"  {t:<12} {len(kept):<8} {dropped:<10} {srcs}")
        print()