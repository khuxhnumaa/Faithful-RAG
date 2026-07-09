import os
import argparse
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
CHUNK_SIZE      = 500    # characters per chunk
CHUNK_OVERLAP   = 100   # overlap so context isn't cut mid-sentence


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_documents(docs_dir: str):
    """
    Load .txt and .pdf files from a directory.
    Returns a list of LangChain Document objects.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents = []

    # Load .txt files
    txt_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    documents.extend(txt_loader.load())

    # Load .pdf files
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
    )
    documents.extend(pdf_loader.load())

    print(f"[ingest] Loaded {len(documents)} document pages from '{docs_dir}'")
    return documents


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(documents):
    """
    Split documents into overlapping chunks.
    Smaller chunks = more precise retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── Embedding & Vector Store ──────────────────────────────────────────────────

def build_vectorstore(chunks, db_path: str):
    """
    Embed chunks using a local HuggingFace model (no API key needed),
    then store in FAISS and save to disk.
    """
    print(f"[ingest] Loading embedding model: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[ingest] Building FAISS index ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    Path(db_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(db_path)
    print(f"[ingest] Vector DB saved to '{db_path}' ({len(chunks)} vectors)")

    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS")
    parser.add_argument("--docs_dir", default="./docs",     help="Folder with .txt/.pdf files")
    parser.add_argument("--db_path",  default="./vectordb", help="Where to save the FAISS index")
    args = parser.parse_args()

    docs   = load_documents(args.docs_dir)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks, args.db_path)
    print("[ingest] Done! Run rag_pipeline.py to ask questions.")


if __name__ == "__main__":
    main()