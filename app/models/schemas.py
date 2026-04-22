from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    project_id: Optional[str] = None
    type: Literal["log", "code", "file"] = "file"


class IngestResponse(BaseModel):
    job_id: str
    project_id: Optional[str]
    status: str
    chunk_count: int = 0
    message: str = ""


class SearchRequest(BaseModel):
    query: str
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    project_id: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    chunk_type: str
    level: int
    content: str
    score: float
    project_id: Optional[str]
    source_file: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str


class ChunkInfo(BaseModel):
    id: str
    chunk_type: str
    level: int
    content: str
    token_count: Optional[int]
    source_file: str
    created_at: str


class ProjectInfo(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    chunk_count: int
    chunks: List[ChunkInfo] = []


class SummarizeRequest(BaseModel):
    project_id: Optional[str] = None
    chunk_ids: Optional[List[str]] = None
    force: bool = False


class SummarizeResponse(BaseModel):
    summarized: int
    message: str


class HealthResponse(BaseModel):
    status: str
    ollama: str
    chroma: str
    db: str
