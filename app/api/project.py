from fastapi import APIRouter, HTTPException
from app.models.schemas import ProjectInfo, ChunkInfo
from app.core.db import get_conn

router = APIRouter()


@router.get("/project/{project_id}", response_model=ProjectInfo)
async def get_project(project_id: str):
    with get_conn() as conn:
        proj = conn.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        if not proj:
            raise HTTPException(status_code=404, detail="Project not found")

        chunks = conn.execute(
            """SELECT id, chunk_type, level, content, token_count, source_file, created_at
               FROM chunks WHERE project_id=? ORDER BY level DESC, chunk_type""",
            (project_id,),
        ).fetchall()

        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE project_id=?", (project_id,)
        ).fetchone()["cnt"]

    return ProjectInfo(
        id=proj["id"],
        name=proj["name"],
        description=proj["description"],
        created_at=proj["created_at"],
        updated_at=proj["updated_at"],
        chunk_count=count,
        chunks=[
            ChunkInfo(
                id=c["id"],
                chunk_type=c["chunk_type"],
                level=c["level"],
                content=c["content"][:500],
                token_count=c["token_count"],
                source_file=c["source_file"],
                created_at=c["created_at"],
            )
            for c in chunks
        ],
    )


@router.get("/projects")
async def list_projects():
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT p.*, COUNT(c.id) as chunk_count
               FROM projects p
               LEFT JOIN chunks c ON c.project_id = p.id
               GROUP BY p.id
               ORDER BY p.updated_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]
