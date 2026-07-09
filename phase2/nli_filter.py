from dataclasses import dataclass, field
from typing import List, Dict
from langchain_core.documents import Document


# ── Config ────────────────────────────────────────────────────────────────────

NLI_MODEL      = "cross-encoder/nli-deberta-v3-base"
KEEP_LABELS    = {"entailment","neutral"}           # only keep entailment
DROP_LABELS    = {"contradiction"}
MIN_NLI_SCORE  = 0.30                     # minimum confidence for entailment label
# If entailment score < MIN_NLI_SCORE even though it's the top label,
# treat it as uncertain and drop.


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class NLIChunkResult:
    doc:        Document
    label:      str    # entailment / neutral / contradiction
    score:      float  # confidence of the predicted label
    kept:       bool

@dataclass
class NLIFilterResult:
    query:        str
    kept:         List[NLIChunkResult]    = field(default_factory=list)
    dropped:      List[NLIChunkResult]    = field(default_factory=list)
    kept_docs:    List[Document]          = field(default_factory=list)

    def summary(self) -> str:
        total = len(self.kept) + len(self.dropped)
        labels = [r.label for r in self.dropped]
        return (
            f"[nli_filter] kept={len(self.kept)}/{total} | "
            f"dropped={len(self.dropped)}/{total} "
            f"(labels: {labels})"
        )


# ── NLI Filter ────────────────────────────────────────────────────────────────

class NLIFilter:

    def __init__(self, model_name: str = NLI_MODEL):
        self.model_name = model_name
        self._pipeline  = None   # lazy load — only downloads on first use

    def _load(self):

        if self._pipeline is None:
            from transformers import pipeline
            print(f"[nli_filter] Loading NLI model: {self.model_name}")
            print("[nli_filter] First run downloads ~500MB — cached after that.")
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,   # CPU; change to 0 for GPU
            )
            print("[nli_filter] Model loaded.")

    def _classify(self, premise: str, hypothesis: str) -> Dict:
        
        self._load()
        candidate_labels = ["entailment", "neutral", "contradiction"]
        result = self._pipeline(
            sequences=premise,
            candidate_labels=candidate_labels,
            hypothesis_template="This passage is relevant to the topic: {}",
        )
        # pipeline returns labels sorted by score descending
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        return {"label": top_label, "score": top_score}

    def filter(
        self,
        docs: List[Document],
        query: str,
        verbose: bool = True
    ) -> NLIFilterResult:
        
        result = NLIFilterResult(query=query)

        print(f"\n[nli_filter] Checking {len(docs)} chunks against query: '{query[:60]}...'")

        for doc in docs:
            src     = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:60].replace("\n", " ")

            nli_out = self._classify(
                premise=doc.page_content,
                hypothesis=query,
            )
            label = nli_out["label"]
            score = nli_out["score"]

            # Keep only entailment with sufficient confidence
            keep = (label in KEEP_LABELS) and (score >= MIN_NLI_SCORE)

            chunk_result = NLIChunkResult(
                doc=doc, label=label, score=score, kept=keep
            )

            if keep:
                result.kept.append(chunk_result)
                result.kept_docs.append(doc)
                if verbose:
                    print(f"  [KEEP  ] label={label:<15} conf={score:.3f} | {src} | {preview}...")
            else:
                result.dropped.append(chunk_result)
                if verbose:
                    print(f"  [DROP  ] label={label:<15} conf={score:.3f} | {src} | {preview}...")

        print(result.summary())
        return result


# ── Fallback handler ──────────────────────────────────────────────────────────

def handle_empty_context(query: str) -> str:
   
    return (
        f"I could not find any relevant information in the provided documents "
        f"to answer: '{query}'. "
        f"Please rephrase your question or check that the relevant document has been ingested."
    )