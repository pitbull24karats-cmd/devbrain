from __future__ import annotations
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from app.models.schemas import SummarizeRequest, SummarizeResponse
from app.core.db import get_conn
from app.services.summarize import summarize_content, abstract_summary
from app.services.embed import embed_text
from app.services import chroma as chroma_svc
from app.services.ingest import _save_chunk, _count_tokens

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    with get_conn() as conn:
        if req.chunk_ids:
            placeholders = ",".join("?" * len(req.chunk_ids))
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND level=0",
                req.chunk_ids,
            ).fetchall()
        elif req.project_id:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE project_id=? AND level=0",
                (req.project_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chunks WHERE level=0").fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No raw chunks found to summarize")

    count = 0
    for row in rows:
        if not req.force:
            with get_conn() as conn:
                existing = conn.execute(
                    """SELECT id FROM chunks
                       WHERE source_file=? AND project_id IS ?  AND level=1 AND chunk_type='summary'""",
                    (row["source_file"], row["project_id"]),
                ).fetchone()
            if existing:
                continue

        result = await summarize_content(row["content"], "file")
        for key, ctype, level in [
            ("summary", "summary", 1),
            ("insights", "insight", 1),
            ("reusable", "reusable", 2),
        ]:
            text = result.get(key, "")
            if not text:
                continue
            cid = str(uuid.uuid4())
            _save_chunk(
                cid, row["project_id"], row["source_file"],
                ctype, level, text, _count_tokens(text), None
            )
            vec = await embed_text(text)
            cid_list = [cid]
            chroma_svc.upsert(
                cid_list, [vec], [text],
                [{"chunk_type": ctype, "level": level,
                  "project_id": row["project_id"] or "", "source_file": row["source_file"]}]
            )
            with get_conn() as conn:
                conn.execute("UPDATE chunks SET chroma_id=? WHERE id=?", (cid, cid))
        count += 1

    return SummarizeResponse(summarized=count, message=f"Re-summarized {count} chunk(s)")
