from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .search import JobSearch

state = {}

@asynccontextmanager
async def lifespan(app):
    state["engine"] = JobSearch(use_reranker=False)  # set True for better quality (slower)
    yield

app = FastAPI(title="Job Search (hybrid RAG retrieval)", lifespan=lifespan)

class Profile(BaseModel):
    skills: list[str] = []
    location: Optional[str] = None
    years: float = 0

class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    profile: Optional[Profile] = None
    top_k: int = Field(5, ge=1, le=20)

@app.get("/health")
def health():
    return {"status": "ok", "jobs": len(state["engine"].jobs)}

@app.post("/search")
def search(req: SearchRequest):
    prof = req.profile.model_dump() if req.profile else None
    return {"query": req.query, "results": state["engine"].search(req.query, prof, req.top_k)}
