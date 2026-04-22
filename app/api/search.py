from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchRequest, SearchResponse
from app.services.search import hybrid_search

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    results = await hybrid_search(req.query, req.limit, req.project_id, req.exclude_types)
    return SearchResponse(results=results, total=len(results), query=req.query)
