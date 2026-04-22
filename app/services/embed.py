from typing import List
import httpx
from app.core.config import settings

_client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=120.0)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    for text in texts:
        resp = await _client.post(
            "/api/embeddings",
            json={"model": settings.embed_model, "prompt": text},
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embedding"])
    return vectors


async def embed_text(text: str) -> List[float]:
    result = await embed_texts([text])
    return result[0]
