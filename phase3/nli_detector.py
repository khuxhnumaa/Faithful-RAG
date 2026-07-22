from dataclasses import dataclass, field
from typing import List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase3'))
from sentence_splitter import split_into_sentences


# ── Config ────────────────────────────────────────────────────────────────────

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Label → verdict mapping
LABEL_VERDICT = {
    "entailment":    "grounded",      # ✓ safe — context supports this
    "neutral":       "uncertain",     # ⚠ weak — context doesn't confirm
    "contradiction": "hallucinated",  # ✗ bad  — context contradicts this
}

# Verdict → emoji for display
VERDICT_EMOJI = {
    "grounded":     "✓",
    "uncertain":    "⚠",
    "hallucinated": "✗",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SentenceResult:
    
    sentence: str
    label:    str
    score:    float
    verdict:  str

    def is_safe(self) -> bool:
        return self.verdict == "grounded"

    def is_hallucinated(self) -> bool:
        return self.verdict == "hallucinated"

    def display(self) -> str:
        emoji = VERDICT_EMOJI.get(self.verdict, "?")
        preview = self.sentence[:70] + ('...' if len(self.sentence) > 70 else '')
        return f"  {emoji} [{self.verdict:<12}] conf={self.score:.3f} | {preview}"


@dataclass
class DetectionResult:
    
    answer:           str
    context:          str
    sentences:        List[SentenceResult] = field(default_factory=list)

    # ── Computed properties ────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        return len(self.sentences)

    @property
    def grounded_count(self) -> int:
        return sum(1 for s in self.sentences if s.verdict == "grounded")

    @property
    def uncertain_count(self) -> int:
        return sum(1 for s in self.sentences if s.verdict == "uncertain")

    @property
    def hallucinated_count(self) -> int:
        return sum(1 for s in self.sentences if s.verdict == "hallucinated")

    @property
    def faithfulness_score(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.grounded_count / self.total, 4)

    @property
    def hallucinated_sentences(self) -> List[str]:
        return [s.sentence for s in self.sentences if s.verdict == "hallucinated"]

    @property
    def uncertain_sentences(self) -> List[str]:
        return [s.sentence for s in self.sentences if s.verdict == "uncertain"]

    def summary(self) -> str:
        return (
            f"[nli_detector] sentences={self.total} | "
            f"grounded={self.grounded_count} | "
            f"uncertain={self.uncertain_count} | "
            f"hallucinated={self.hallucinated_count} | "
            f"faithfulness={self.faithfulness_score}"
        )

    def print_full(self) -> None:
        print(f"\n{'='*60}")
        print(f"NLI DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Answer: {self.answer[:100]}...")
        print(f"\nPer-sentence breakdown ({self.total} sentences):")
        for r in self.sentences:
            print(r.display())
        print(f"\nSummary:")
        print(f"  Grounded:     {self.grounded_count}/{self.total}")
        print(f"  Uncertain:    {self.uncertain_count}/{self.total}")
        print(f"  Hallucinated: {self.hallucinated_count}/{self.total}")
        print(f"  Faithfulness: {self.faithfulness_score}")
        if self.hallucinated_sentences:
            print(f"\nHallucinated sentences:")
            for s in self.hallucinated_sentences:
                print(f"  ✗ {s}")
        print(f"{'='*60}\n")


# ── Detector ──────────────────────────────────────────────────────────────────

class NLIDetector:
    def __init__(self, model_name: str = NLI_MODEL):
        self.model_name = model_name
        self._pipeline  = None   # lazy load

    def _load(self):
        """Load NLI model on first use. Cached after first download."""
        if self._pipeline is None:
            from transformers import pipeline
            print(f"[nli_detector] Loading model: {self.model_name}")
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU; set to 0 for GPU
            )
            print("[nli_detector] Model loaded.")

    # def _check_sentence(self, sentence: str, context: str) -> SentenceResult:
        
    #     self._load()

    #     result = self._pipeline(
    #         sequences=context,
    #         candidate_labels=["entailment", "neutral", "contradiction"],
    #         hypothesis_template="Based on this text: {}",
    #     )

    #     # Override hypothesis_template to make it sentence-specific
    #     # Use direct NLI scoring: context as premise, sentence as hypothesis
    #     result2 = self._pipeline(
    #         sequences=sentence,
    #         candidate_labels=["supported", "not supported", "contradicted"],
    #         hypothesis_template="According to the provided context, this claim is {}",
    #     )

    #     # Map back to standard NLI labels
    #     label_map = {
    #         "supported":     "entailment",
    #         "not supported": "neutral",
    #         "contradicted":  "contradiction",
    #     }
    #     top_label_raw = result2["labels"][0]
    #     top_score     = result2["scores"][0]
    #     top_label     = label_map.get(top_label_raw, "neutral")
    #     verdict       = LABEL_VERDICT[top_label]

    #     return SentenceResult(
    #         sentence=sentence,
    #         label=top_label,
    #         score=round(top_score, 4),
    #         verdict=verdict,
    #     )
    # REPLACE the _check_sentence return logic with this simpler version:

    def _check_sentence(self, sentence: str, context: str) -> SentenceResult:
        self._load()

        # Truncate both to avoid token overflow
        ctx  = context[:600]
        sent = sentence[:200]

        result = self._pipeline(
            sequences=ctx,
            candidate_labels=["supported", "not supported"],
            hypothesis_template="This claim is {}: " + sent,
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        # Map to NLI labels
        if top_label == "supported" and top_score > 0.40:
            label   = "entailment"
            verdict = "grounded"
        elif top_label == "not supported" and top_score > 0.70:
            label   = "contradiction"
            verdict = "hallucinated"
        else:
            label   = "neutral"
            verdict = "uncertain"

        return SentenceResult(
            sentence=sentence,
            label=label,
            score=round(top_score, 4),
            verdict=verdict,
        )

    def detect(
        self,
        answer:  str,
        context: str,
        verbose: bool = True,
    ) -> DetectionResult:
        
        result = DetectionResult(answer=answer, context=context)

        # Step 1: Split into sentences
        sentences = split_into_sentences(answer)
        if not sentences:
            print("[nli_detector] No sentences found in answer.")
            return result

        print(f"\n[nli_detector] Checking {len(sentences)} sentences...")

        # Step 2: Check each sentence
        for i, sentence in enumerate(sentences, 1):
            sent_result = self._check_sentence(sentence, context)
            result.sentences.append(sent_result)

            if verbose:
                emoji = VERDICT_EMOJI[sent_result.verdict]
                print(
                    f"  [{i}/{len(sentences)}] {emoji} {sent_result.verdict:<12} "
                    f"conf={sent_result.score:.3f} | "
                    f"{sentence[:65]}{'...' if len(sentence)>65 else ''}"
                )

        # Step 3: Print summary
        print(result.summary())
        return result