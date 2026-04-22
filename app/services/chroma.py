from __future__ import annotations
from typing import List, Optional
import chromadb
from app.core.config import settings

_client = chromadb.PersistentClient(path=str(settings.embeddings_dir))
_collection = _client.get_or_create_collection(
    name="devbrain",
    metadata={"hnsw:space": "cosine"},
)

TYPE_PRIORITY = {"reusable": 4, "insight": 3, "summary": 2, "raw": 1}


def upsert(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[dict],
) -> None:
    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(
    vector: List[float],
    n_results: int = 10,
    where: Optional[dict] = None,
) -> list[dict]:
    kwargs: dict = {"query_embeddings": [vector], "n_results": n_results}
    if where:
        kwargs["where"] = where
    res = _collection.query(**kwargs, include=["documents", "metadatas", "distances"])
    out = []
    for i, doc_id in enumerate(res["ids"][0]):
        meta = res["metadatas"][0][i]
        dist = res["distances"][0][i]
        vector_score = 1.0 - dist
        out.append(
            {
                "id": doc_id,
                "content": res["documents"][0][i],
                "meta": meta,
                "vector_score": vector_score,
            }
        )
    return out
