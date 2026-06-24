
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))

import argparse
from dataclasses import dataclass, field
from typing import List

# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from similarity_filter import SimilarityFilter, FilterResult
from nli_filter import NLIFilter, NLIFilterResult, handle_empty_context


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH            = "../week1/vectordb"
TOP_K              = 6       # retrieve more initially — filters will trim down
SIM_THRESHOLD      = 0.5    # Week 2 filter 1
OLLAMA_MODEL       = "llama3.2:3b"

RAG_PROMPT = """You are a helpful assistant
If the answer is not in the context, say exactly: "I don't have enough information to answer this from the provided documents."


Context:
{context}

Question: {query}

Answer:"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RAGResultW2:
    query:              str
    raw_docs:           List[Document]    # before any filtering
    raw_scores:         List[float]
    after_sim_filter:   FilterResult      # after similarity filter
    after_nli_filter:   NLIFilterResult   # after NLI filter
    context:            str               # final context sent to LLM
    answer:             str
    was_fallback:       bool = False      # True if all chunks were dropped


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {src}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ── LLM ───────────────────────────────────────────────────────────────────────

def generate_answer(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model=model, temperature=0)
        return llm.invoke(prompt)
    except Exception as e:
        return f"[LLM error] {e}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_rag_w2(
    query:         str,
    db_path:       str  = DB_PATH,
    sim_threshold: float = SIM_THRESHOLD,
    top_k:         int   = TOP_K,
    skip_nli:      bool  = False,
) -> RAGResultW2:

    # ── 1. Retrieve ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    raw_results = vectorstore.similarity_search_with_score(query, k=top_k)
    raw_docs    = [r[0] for r in raw_results]
    raw_scores  = [float(1 / (1 + r[1])) for r in raw_results]

    print(f"\n[retriever] Retrieved {len(raw_docs)} chunks (top_k={top_k})")

    # ── 2. Similarity filter ──────────────────────────────────────────────────
    print(f"\n--- Filter 1: Similarity threshold (≥ {sim_threshold}) ---")
    sim_filter  = SimilarityFilter(threshold=sim_threshold)
    sim_result  = sim_filter.filter(raw_docs, raw_scores)

    # ── 3. NLI filter ─────────────────────────────────────────────────────────
    if not skip_nli and len(sim_result.kept_docs) > 0:
        print(f"\n--- Filter 2: NLI entailment filter ---")
        nli_filter  = NLIFilter()
        nli_result  = nli_filter.filter(sim_result.kept_docs, query)
        final_docs  = nli_result.kept_docs
    else:
        # Build a dummy NLIFilterResult if skipping
        from nli_filter import NLIFilterResult
        nli_result = NLIFilterResult(query=query)
        nli_result.kept_docs = sim_result.kept_docs
        final_docs = sim_result.kept_docs
        if skip_nli:
            print("[nli_filter] Skipped (--skip_nli flag)")

    # ── 4. Fallback if all chunks dropped ────────────────────────────────────
    if len(final_docs) == 0:
        print("\n[pipeline] ALL chunks dropped by filters.")
        fallback = handle_empty_context(query)
        return RAGResultW2(
            query=query,
            raw_docs=raw_docs,
            raw_scores=raw_scores,
            after_sim_filter=sim_result,
            after_nli_filter=nli_result,
            context="",
            answer=fallback,
            was_fallback=True,
        )

    # ── 5. Build context ──────────────────────────────────────────────────────
    context = build_context(final_docs)
    print(f"\n[pipeline] Final context: {len(final_docs)} chunks, {len(context)} chars")

    # ── 6. Generate ───────────────────────────────────────────────────────────
    prompt = RAG_PROMPT.format(context=context, query=query)
    print(f"\n[llm] Generating answer ...")
    answer = generate_answer(prompt)

    result = RAGResultW2(
        query=query,
        raw_docs=raw_docs,
        raw_scores=raw_scores,
        after_sim_filter=sim_result,
        after_nli_filter=nli_result,
        context=context,
        answer=answer,
        was_fallback=False,
    )

    # ── 7. Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ANSWER: {answer.strip()}")
    print(f"{'='*60}")
    print(f"\nFILTER SUMMARY:")
    print(f"  Raw chunks retrieved:      {len(raw_docs)}")
    print(f"  After similarity filter:   {len(sim_result.kept_docs)}")
    print(f"  After NLI filter:          {len(final_docs)}")
    print(f"  Fallback used:             {result.was_fallback}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Week 2 RAG pipeline with prevention filters")
    parser.add_argument("--query",      required=True)
    parser.add_argument("--db_path",    default=DB_PATH)
    parser.add_argument("--threshold",  default=SIM_THRESHOLD, type=float)
    parser.add_argument("--top_k",      default=TOP_K, type=int)
    parser.add_argument("--skip_nli",   action="store_true", help="Skip NLI filter (faster)")
    args = parser.parse_args()

    run_rag_w2(
        query=args.query,
        db_path=args.db_path,
        sim_threshold=args.threshold,
        top_k=args.top_k,
        skip_nli=args.skip_nli,
    )


if __name__ == "__main__":
    main()