from __future__ import annotations
import uuid
import asyncio
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from datetime import datetime, timezone

from app.models.schemas import IngestRequest, IngestResponse
from app.core.db import get_conn
from app.services.ingest import run_ingest

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_id: Optional[str] = Form(None),
    type: str = Form("file"),
):
    if type not in ("log", "code", "file"):
        raise HTTPException(status_code=422, detail="type must be log|code|file")

    job_id = str(uuid.uuid4())
    file_contents = [(f.filename or "unknown", await f.read()) for f in files]

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO ingest_jobs(id, project_id, status, file_count)
               VALUES(?, ?, 'pending', ?)""",
            (job_id, project_id, len(file_contents)),
        )

    background_tasks.add_task(
        run_ingest, job_id, file_contents, type, project_id
    )

    return IngestResponse(
        job_id=job_id,
        project_id=project_id,
        status="pending",
        message=f"{len(file_contents)} file(s) queued for ingestion",
    )


@router.get("/ingest/{job_id}")
async def get_job(job_id: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ingest_jobs WHERE id=?", (job_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return dict(row)
