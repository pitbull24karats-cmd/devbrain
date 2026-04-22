from __future__ import annotations
from typing import List, Optional
import httpx

from app.core.config import settings
from app.core.db import get_conn
from app.services.embed import embed_text
from app.services import chroma as chroma_svc
from app.models.schemas import SearchResult

TYPE_PRIORITY = {"reusable": 4, "insight": 3, "summary": 2, "raw": 1}

_TRANSLATE_PROMPT = (
    "Translate the following query to English for semantic search. "
    "Return only the translated text:\n{query}"
)


async def _translate_to_english(query: str) -> str:
    try:
        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=30.0
        ) as client:
            resp = await client.post(
                "/api/generate",
                json={
                    "model": settings.llm_model,
                    "prompt": _TRANSLATE_PROMPT.format(query=query),
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()
    except Exception:
        return query


async def hybrid_search(
    query: str,
    limit: int = 10,
    project_id: Optional[str] = None,
) -> List[SearchResult]:
    embed_query = await _translate_to_english(query)
    vector = await embed_text(embed_query)
    where = {"project_id": project_id} if project_id else None
    vector_hits = chroma_svc.query(vector, n_results=limit * 2, where=where)

    keyword_scores: dict[str, float] = {}
    with get_conn() as conn:
        fts_q = f'"{query}"' if " " in query else query
        if project_id:
            rows = conn.execute(
                """SELECT c.id, c.chunk_type, c.level, c.content, c.project_id, c.source_file
                   FROM chunks_fts f
                   JOIN chunks c ON c.id = f.id
                   WHERE chunks_fts MATCH ? AND c.project_id = ?
                   LIMIT ?""",
                (fts_q, project_id, limit * 2),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT c.id, c.chunk_type, c.level, c.content, c.project_id, c.source_file
                   FROM chunks_fts f
                   JOIN chunks c ON c.id = f.id
                   WHERE chunks_fts MATCH ?
                   LIMIT ?""",
                (fts_q, limit * 2),
            ).fetchall()
    for rank, row in enumerate(rows):
        keyword_scores[row["id"]] = 1.0 - rank / max(len(rows), 1)

    scores: dict[str, dict] = {}
    for hit in vector_hits:
        scores[hit["id"]] = {
            "vector_score": hit["vector_score"],
            "keyword_score": keyword_scores.get(hit["id"], 0.0),
            "content": hit["content"],
            "meta": hit["meta"],
        }
    for chunk_id, ks in keyword_scores.items():
        if chunk_id not in scores:
            scores[chunk_id] = {"vector_score": 0.0, "keyword_score": ks, "meta": {}}

    for chunk_id in list(scores.keys()):
        if not scores[chunk_id].get("content"):
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT content, chunk_type, level, project_id, source_file FROM chunks WHERE id=?",
                    (chunk_id,)
                ).fetchone()
                if row:
                    scores[chunk_id]["content"] = row["content"]
                    scores[chunk_id]["meta"] = {
                        "chunk_type": row["chunk_type"],
                        "level": row["level"],
                        "project_id": row["project_id"] or "",
                        "source_file": row["source_file"],
                    }

    results = []
    for chunk_id, data in scores.items():
        vs = data["vector_score"]
        ks = data["keyword_score"]
        hybrid = settings.hybrid_vector_weight * vs + settings.hybrid_keyword_weight * ks
        meta = data.get("meta", {})
        chunk_type = meta.get("chunk_type", "raw")
        priority_bonus = TYPE_PRIORITY.get(chunk_type, 1) * 0.01
        final_score = hybrid + priority_bonus
        results.append(
            SearchResult(
                id=chunk_id,
                chunk_type=chunk_type,
                level=meta.get("level", 0),
                content=data.get("content", ""),
                score=round(final_score, 4),
                project_id=meta.get("project_id") or None,
                source_file=meta.get("source_file", ""),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
