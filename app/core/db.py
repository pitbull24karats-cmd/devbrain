import sqlite3
from contextlib import contextmanager
from app.core.config import settings

DDL = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    project_id  TEXT,
    source_file TEXT NOT NULL,
    chunk_type  TEXT NOT NULL,  -- summary | reusable | insight | raw
    level       INTEGER NOT NULL DEFAULT 0,
    content     TEXT NOT NULL,
    token_count INTEGER,
    chroma_id   TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id          TEXT PRIMARY KEY,
    project_id  TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    file_count  INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    error       TEXT,
    started_at  TEXT,
    finished_at TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_project   ON chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type      ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_level     ON chunks(level);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    id UNINDEXED, content, chunk_type, project_id
);
"""


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(DDL)


@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
