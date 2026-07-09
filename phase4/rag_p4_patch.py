# ── PASTE THIS into rag_p4.py, replacing the existing requery_fn and verifier.verify() call ──

# Step 1: define requery_fn with is_retry=True so it NEVER loops again
from gc import set_threshold
from os import defpath
from unittest import result
from rag_pipeline_p3 import skip_nli_filter
from fastapi import Query
from phase4 import verifier
from phase4.ragas_scorer import RAGASScorer
from rag_pipeline_p3 import run_rag_w3


def requery_fn(rewritten_query: str):
    w3_retry = run_rag_w3(
        query=rewritten_query,
        db_path=defpath,
        sim_threshold=set_threshold,
        skip_nli_filter=skip_nli_filter,
        verbose=False,
    )
    scorer_retry  = RAGASScorer()
    scores_retry  = scorer_retry.compute(
        query=rewritten_query,
        answer=w3_retry.answer,
        context=w3_retry.w2_result.context,
        detection=w3_retry.detection_result,
    )
    verifier_retry = verifier.Verifier()
    verdict_retry  = verifier_retry.verify(
        query=rewritten_query,
        answer=w3_retry.answer,
        scores=scores_retry,
        detection=w3_retry.detection_result,
        pipeline_fn=None,     # ← no further re-query
        is_retry=True,        # ← stops the loop
    )
    from rag_p4 import RAGResultW4
    return RAGResultW4(
        w3_result=w3_retry,
        scores=scores_retry,
        result=verdict_retry,
    )

# Step 2: call verifier with the requery_fn
verdict_result = verifier.verify(
    query=Query,
    answer=result.answer,
    scores=codes,
    detection=result.detection_result,
    pipeline_fn=requery_fn,
    is_retry=False,   # ← first call is never a retry
)