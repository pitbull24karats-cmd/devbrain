from __future__ import annotations
import uuid
import json
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

import aiofiles
import tiktoken

from app.core.config import settings
from app.core.db import get_conn
from app.services.embed import embed_texts
from app.services.summarize import summarize_content, abstract_summary
from app.services import chroma as chroma_svc

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks, current, count = [], [], 0
    for word in words:
        wt = len(_enc.encode(word))
        if count + wt > settings.chunk_max_tokens and count >= settings.chunk_min_tokens:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(word)
        count += wt
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def _ensure_project_dirs(project_id: str) -> dict[str, Path]:
    base = settings.projects_dir / project_id
    dirs = {
        "raw": base / "raw",
        "processed": base / "processed",
        "knowledge": base / "knowledge",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _upsert_project(project_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO projects(id, name) VALUES(?, ?)
            ON CONFLICT(id) DO UPDATE SET updated_at=datetime('now')
            """,
            (project_id, project_id),
        )


async def run_ingest(
    job_id: str,
    file_contents: List[tuple[str, bytes]],
    content_type: str,
    project_id: Optional[str],
) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE ingest_jobs SET status='running', started_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), job_id),
        )

    if project_id:
        _upsert_project(project_id)
        dirs = _ensure_project_dirs(project_id)
    else:
        dirs = None

    chunk_count = 0
    try:
        for filename, raw_bytes in file_contents:
            text = raw_bytes.decode("utf-8", errors="replace")

            if dirs:
                raw_path = dirs["raw"] / filename
                async with aiofiles.open(raw_path, "w", encoding="utf-8") as f:
                    await f.write(text)

            raw_chunks = _chunk_text(text)

            summaries = await summarize_content(text[:8000], content_type)

            processed_chunks = []

            raw_chunk_ids = []
            for i, chunk in enumerate(raw_chunks):
                cid = str(uuid.uuid4())
                raw_chunk_ids.append(cid)
                _save_chunk(
                    cid, project_id, filename, "raw", 0,
                    chunk, _count_tokens(chunk), None
                )
                processed_chunks.append((cid, chunk))

            summary_text = summaries.get("summary", "")
            insight_text = summaries.get("insights", "")
            reusable_text = summaries.get("reusable", "")

            if summary_text:
                sid = str(uuid.uuid4())
                _save_chunk(sid, project_id, filename, "summary", 1,
                            summary_text, _count_tokens(summary_text), None)
                processed_chunks.append((sid, summary_text))
                chunk_count += 1

                if dirs:
                    async with aiofiles.open(
                        dirs["processed"] / f"{Path(filename).stem}_summary.txt", "w"
                    ) as f:
                        await f.write(summary_text)

                abstract = await abstract_summary(summary_text)
                abs_text = abstract.get("abstract", "")
                abs_reusable = abstract.get("reusable", "")

                if abs_text:
                    aid = str(uuid.uuid4())
                    _save_chunk(aid, project_id, filename, "summary", 2,
                                abs_text, _count_tokens(abs_text), None)
                    processed_chunks.append((aid, abs_text))
                    chunk_count += 1

                if abs_reusable:
                    rid = str(uuid.uuid4())
                    _save_chunk(rid, project_id, filename, "reusable", 3,
                                abs_reusable, _count_tokens(abs_reusable), None)
                    processed_chunks.append((rid, abs_reusable))
                    chunk_count += 1

            if insight_text:
                iid = str(uuid.uuid4())
                _save_chunk(iid, project_id, filename, "insight", 1,
                            insight_text, _count_tokens(insight_text), None)
                processed_chunks.append((iid, insight_text))
                chunk_count += 1

            if reusable_text:
                reid = str(uuid.uuid4())
                _save_chunk(reid, project_id, filename, "reusable", 2,
                            reusable_text, _count_tokens(reusable_text), None)
                processed_chunks.append((reid, reusable_text))
                chunk_count += 1

            chunk_count += len(raw_chunks)

            if processed_chunks:
                ids = [c[0] for c in processed_chunks]
                texts = [c[1] for c in processed_chunks]
                vectors = await embed_texts(texts)
                metadatas = []
                with get_conn() as conn:
                    for cid in ids:
                        row = conn.execute(
                            "SELECT chunk_type, level, project_id, source_file FROM chunks WHERE id=?",
                            (cid,)
                        ).fetchone()
                        if row:
                            metadatas.append({
                                "chunk_type": row["chunk_type"],
                                "level": row["level"],
                                "project_id": row["project_id"] or "",
                                "source_file": row["source_file"],
                            })
                        else:
                            metadatas.append({})
                chroma_svc.upsert(ids, vectors, texts, metadatas)
                with get_conn() as conn:
                    for cid in ids:
                        conn.execute(
                            "UPDATE chunks SET chroma_id=? WHERE id=?", (cid, cid)
                        )

            if dirs:
                meta_path = settings.projects_dir / project_id / "meta.json"
                meta = {
                    "project_id": project_id,
                    "last_ingest": datetime.now(timezone.utc).isoformat(),
                    "chunk_count": chunk_count,
                }
                async with aiofiles.open(meta_path, "w") as f:
                    await f.write(json.dumps(meta, ensure_ascii=False, indent=2))

        with get_conn() as conn:
            conn.execute(
                """UPDATE ingest_jobs
                   SET status='done', chunk_count=?, finished_at=?
                   WHERE id=?""",
                (chunk_count, datetime.now(timezone.utc).isoformat(), job_id),
            )
    except Exception as exc:
        with get_conn() as conn:
            conn.execute(
                "UPDATE ingest_jobs SET status='error', error=?, finished_at=? WHERE id=?",
                (str(exc), datetime.now(timezone.utc).isoformat(), job_id),
            )
        raise


def _save_chunk(
    cid: str, project_id, source_file, chunk_type, level, content, token_count, chroma_id
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (id, project_id, source_file, chunk_type, level, content, token_count, chroma_id)
               VALUES(?,?,?,?,?,?,?,?)""",
            (cid, project_id, source_file, chunk_type, level, content, token_count, chroma_id),
        )
        conn.execute(
            "INSERT OR REPLACE INTO chunks_fts(id, content, chunk_type, project_id) VALUES(?,?,?,?)",
            (cid, content, chunk_type, project_id or ""),
        )
