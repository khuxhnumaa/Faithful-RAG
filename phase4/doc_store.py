"""
doc_store.py — Persistent Document Storage
=============================================
WHAT THIS FILE DOES:
    Manages a local SQLite database that stores metadata about
    every document the user uploads. This means documents persist
    across page refreshes and app restarts.

    Also manages the FAISS index folder for each document — each
    document gets its own subfolder so switching between documents
    just swaps the active FAISS path.

    DATABASE: stored at data/doc_store.db (created automatically)
    FAISS INDEXES: stored at data/indexes/<doc_id>/ (one per document)
    UPLOADED FILES: stored at data/uploads/<filename>

    TABLE: documents
      id          TEXT  — unique ID (uuid4)
      name        TEXT  — original filename
      uploaded_at TEXT  — timestamp
      chunk_count INT   — number of chunks indexed
      file_path   TEXT  — path to saved original file
      index_path  TEXT  — path to FAISS index folder

Usage:
    from doc_store import DocStore
    db = DocStore()
    db.save_document(doc_id, name, chunk_count, file_path, index_path)
    docs = db.list_documents()
    db.delete_document(doc_id)
"""

import sqlite3
import os
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent / "data"
DB_PATH     = BASE_DIR / "doc_store.db"
UPLOADS_DIR = BASE_DIR / "uploads"
INDEXES_DIR = BASE_DIR / "indexes"

# Create dirs on import
BASE_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
INDEXES_DIR.mkdir(exist_ok=True)


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class StoredDocument:
    id:           str
    name:         str
    uploaded_at:  str
    chunk_count:  int
    file_path:    str
    index_path:   str

    @property
    def display_name(self):
        return self.name

    @property
    def short_date(self):
        try:
            dt = datetime.fromisoformat(self.uploaded_at)
            return dt.strftime("%d %b %Y, %H:%M")
        except:
            return self.uploaded_at


# ── DocStore ──────────────────────────────────────────────────────────────────

class DocStore:
    """
    SQLite-backed document store.
    All methods are safe to call multiple times — the DB
    and table are created automatically if they don't exist.
    """

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id           TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    uploaded_at  TEXT NOT NULL,
                    chunk_count  INTEGER DEFAULT 0,
                    file_path    TEXT,
                    index_path   TEXT
                )
            """)
            conn.commit()

    def save_document(
        self,
        doc_id:      str,
        name:        str,
        chunk_count: int,
        file_path:   str,
        index_path:  str,
    ) -> StoredDocument:
        """
        Save a new document record to the database.
        If doc_id already exists, updates chunk_count and paths.
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO documents (id, name, uploaded_at, chunk_count, file_path, index_path)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    chunk_count = excluded.chunk_count,
                    file_path   = excluded.file_path,
                    index_path  = excluded.index_path
            """, (doc_id, name, now, chunk_count, file_path, index_path))
            conn.commit()

        return StoredDocument(
            id=doc_id, name=name, uploaded_at=now,
            chunk_count=chunk_count, file_path=file_path, index_path=index_path
        )

    def list_documents(self) -> List[StoredDocument]:
        """Return all documents, newest first."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, name, uploaded_at, chunk_count, file_path, index_path
                FROM documents
                ORDER BY uploaded_at DESC
            """).fetchall()
        return [StoredDocument(*r) for r in rows]

    def get_document(self, doc_id: str) -> Optional[StoredDocument]:
        """Get a single document by ID."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT id, name, uploaded_at, chunk_count, file_path, index_path
                FROM documents WHERE id = ?
            """, (doc_id,)).fetchone()
        return StoredDocument(*row) if row else None

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document record AND its associated files:
          - uploaded file
          - FAISS index folder
        Returns True if deleted, False if not found.
        """
        doc = self.get_document(doc_id)
        if not doc:
            return False

        # Delete FAISS index folder
        if doc.index_path and os.path.exists(doc.index_path):
            shutil.rmtree(doc.index_path, ignore_errors=True)

        # Delete uploaded file
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except:
                pass

        # Delete DB record
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

        return True

    def document_exists(self, name: str) -> Optional[StoredDocument]:
        """Check if a document with this filename already exists."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT id, name, uploaded_at, chunk_count, file_path, index_path
                FROM documents WHERE name = ?
            """, (name,)).fetchone()
        return StoredDocument(*row) if row else None

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


# ── Helper: get index path for a doc_id ──────────────────────────────────────

def get_index_path(doc_id: str) -> str:
    """Returns the FAISS index path for a given document ID."""
    return str(INDEXES_DIR / doc_id)


def get_upload_path(filename: str) -> str:
    """Returns the save path for an uploaded file."""
    return str(UPLOADS_DIR / filename)