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
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, name, uploaded_at, chunk_count, file_path, index_path
                FROM documents
                ORDER BY uploaded_at DESC
            """).fetchall()
        return [StoredDocument(*r) for r in rows]

    def get_document(self, doc_id: str) -> Optional[StoredDocument]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT id, name, uploaded_at, chunk_count, file_path, index_path
                FROM documents WHERE id = ?
            """, (doc_id,)).fetchone()
        return StoredDocument(*row) if row else None

    def delete_document(self, doc_id: str) -> bool:
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
        # Check if a document with this filename already exists
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
    return str(INDEXES_DIR / doc_id)


def get_upload_path(filename: str) -> str:
    return str(UPLOADS_DIR / filename)