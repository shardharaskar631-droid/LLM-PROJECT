# 1 · Job Search — Hybrid Semantic Search + Personalisation

Covers: **semantic search, embeddings, RAG retrieval layer, ranking, personalisation, FastAPI, Docker**.

Pipeline: BM25 (keyword) + MiniLM embeddings (semantic) → Reciprocal Rank Fusion → optional cross-encoder rerank → blend with a profile score (skill overlap, location, experience).

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
curl -X POST localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "build LLM agents with RAG",
  "profile": {"skills": ["python","rag","fastapi"], "location": "Pune", "years": 0},
  "top_k": 5}'
```
Docker: `docker build -t job-search . && docker run -p 8000:8000 job-search`

## Upgrade ideas (to talk about in interviews)
- Replace the numpy matrix with **Qdrant** (you already know it) + payload filters (location, min_years).
- Swap `jobs.json` for Postgres + `pgvector`; add a nightly re-embedding job.
- Add query rewriting with an LLM (expand "ML jobs" → skills list).
- Log clicks and learn the `alpha` weight / train a LightGBM ranker (see project 3).
- Measure everything with project 4 (recall@k, MRR, nDCG).
