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


def _build_chroma_where(
    project_id: Optional[str], exclude_types: Optional[List[str]]
) -> Optional[dict]:
    conditions = []
    if project_id:
        conditions.append({"project_id": project_id})
    if exclude_types:
        if len(exclude_types) == 1:
            conditions.append({"chunk_type": {"$ne": exclude_types[0]}})
        else:
            conditions.append({"chunk_type": {"$nin": exclude_types}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


async def hybrid_search(
    query: str,
    limit: int = 10,
    project_id: Optional[str] = None,
    exclude_types: Optional[List[str]] = None,
) -> List[SearchResult]:
    embed_query = await _translate_to_english(query)
    vector = await embed_text(embed_query)
    where = _build_chroma_where(project_id, exclude_types)
    vector_hits = chroma_svc.query(vector, n_results=limit * 2, where=where)

    keyword_scores: dict[str, float] = {}
    with get_conn() as conn:
        fts_q = f'"{query}"' if " " in query else query
        type_filter = ""
        type_params: list = []
        if exclude_types:
            placeholders = ",".join("?" * len(exclude_types))
            type_filter = f"AND c.chunk_type NOT IN ({placeholders})"
            type_params = list(exclude_types)
        if project_id:
            rows = conn.execute(
                f"""SELECT c.id, c.chunk_type, c.level, c.content, c.project_id, c.source_file
                   FROM chunks_fts f
                   JOIN chunks c ON c.id = f.id
                   WHERE chunks_fts MATCH ? AND c.project_id = ? {type_filter}
                   LIMIT ?""",
                (fts_q, project_id, *type_params, limit * 2),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""SELECT c.id, c.chunk_type, c.level, c.content, c.project_id, c.source_file
                   FROM chunks_fts f
                   JOIN chunks c ON c.id = f.id
                   WHERE chunks_fts MATCH ? {type_filter}
                   LIMIT ?""",
                (fts_q, *type_params, limit * 2),
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

    if exclude_types:
        results = [r for r in results if r.chunk_type not in exclude_types]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
