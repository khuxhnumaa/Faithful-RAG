import argparse
from dataclasses import dataclass, field
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH         = "./vectordb"
TOP_K           = 4        # number of chunks to retrieve
LLM_BACKEND     = "ollama" # "ollama" | "huggingface"
OLLAMA_MODEL = "llama3.2:3b" # any model you pulled with: ollama pull <model>
HF_MODEL        = "google/flan-t5-base"  # fallback, smaller/slower


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    query:          str
    retrieved_docs: List[Document]
    retrieval_scores: List[float]   # cosine similarity scores
    context:        str             # combined chunk text sent to LLM
    answer:         str
    llm_backend:    str


# ── Retriever ─────────────────────────────────────────────────────────────────

def load_retriever(db_path: str = DB_PATH):
    """Load FAISS index from disk and return a retriever."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[retriever] Loaded FAISS index from '{db_path}'")
    return vectorstore


def retrieve(vectorstore, query: str, top_k: int = TOP_K):
    """
    Retrieve top-k chunks with similarity scores.
    Returns (docs, scores) — scores are cosine similarity [0, 1].
    """
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    docs   = [r[0] for r in results]
    # FAISS returns L2 distance; convert to similarity for readability
    scores = [float(1 / (1 + r[1])) for r in results]

    print(f"\n[retriever] Query: '{query}'")
    print(f"[retriever] Retrieved {len(docs)} chunks:")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        src = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i+1}] score={score:.3f} | {src} | {preview}...")

    return docs, scores


# ── Context Builder ───────────────────────────────────────────────────────────

def build_context(docs: List[Document]) -> str:
   
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {src}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ── Prompt ────────────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using the information in the context below and your global information.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""


# ── LLM Backends ─────────────────────────────────────────────────────────────

def answer_with_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model=model, temperature=0)
        return llm.invoke(prompt)
    except Exception as e:
        return f"[Ollama error] {e}\nTip: run 'ollama pull mistral' first."


def answer_with_huggingface(prompt: str, model: str = HF_MODEL) -> str:
    
    from transformers import pipeline
    print(f"[llm] Loading HuggingFace model: {model} (first run downloads it) ...")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        max_new_tokens=256,
    )
    result = pipe(prompt)[0]["generated_text"]
    return result


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_rag(query: str, db_path: str = DB_PATH, backend: str = LLM_BACKEND) -> RAGResult:
    
    # 1. Retrieve
    vectorstore = load_retriever(db_path)
    docs, scores = retrieve(vectorstore, query)

    # 2. Build context
    context = build_context(docs)

    # 3. Build prompt
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)

    # 4. Generate
    print(f"\n[llm] Generating answer with backend='{backend}' ...")
    if backend == "ollama":
        answer = answer_with_ollama(prompt)
    elif backend == "huggingface":
        answer = answer_with_huggingface(prompt)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'huggingface'.")

    result = RAGResult(
        query=query,
        retrieved_docs=docs,
        retrieval_scores=scores,
        context=context,
        answer=answer,
        llm_backend=backend,
    )

    print(f"\n{'='*60}")
    print(f"Query:  {query}")
    print(f"Answer: {answer.strip()}")
    print(f"{'='*60}\n")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run a RAG query")
    parser.add_argument("--query",   required=True,          help="Question to ask")
    parser.add_argument("--db_path", default=DB_PATH,        help="FAISS DB path")
    parser.add_argument("--backend", default=LLM_BACKEND,   help="ollama | huggingface")
    parser.add_argument("--top_k",   default=TOP_K, type=int, help="Chunks to retrieve")
    args = parser.parse_args()

    run_rag(query=args.query, db_path=args.db_path, backend=args.backend)


if __name__ == "__main__":
    main()