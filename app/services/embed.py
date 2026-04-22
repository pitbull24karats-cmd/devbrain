import asyncio
from typing import List
import httpx
from app.core.config import settings

# Limit concurrent Ollama calls to avoid timeouts under load
_ollama_sem = asyncio.Semaphore(1)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    for text in texts:
        async with _ollama_sem:
            async with httpx.AsyncClient(
                base_url=settings.ollama_base_url, timeout=300.0
            ) as client:
                resp = await client.post(
                    "/api/embeddings",
                    json={"model": settings.embed_model, "prompt": text},
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
    return vectors


async def embed_text(text: str) -> List[float]:
    result = await embed_texts([text])
    return result[0]
