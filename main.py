import httpx
import chromadb
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.db import init_db
from app.api import ingest, search, project, summarize
from app.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="DevBrain", version="0.1.0", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(project.router)
app.include_router(summarize.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    ollama_status = "ok"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            r.raise_for_status()
    except Exception as e:
        ollama_status = f"error: {e}"

    chroma_status = "ok"
    try:
        from app.services.chroma import _collection
        _collection.count()
    except Exception as e:
        chroma_status = f"error: {e}"

    db_status = "ok"
    try:
        from app.core.db import get_conn
        with get_conn() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        db_status = f"error: {e}"

    overall = "ok" if all(
        s == "ok" for s in [ollama_status, chroma_status, db_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        chroma=chroma_status,
        db=db_status,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=True)
