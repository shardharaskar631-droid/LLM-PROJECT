"""Hybrid semantic job search: BM25 + dense embeddings -> RRF fusion -> optional cross-encoder rerank -> personalisation."""
import json, re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

DATA = Path(__file__).resolve().parent.parent / "data" / "jobs.json"
tokenize = lambda s: re.findall(r"[a-z0-9+#.]+", s.lower())


class JobSearch:
    def __init__(self, use_reranker: bool = False):
        self.jobs = json.loads(DATA.read_text())
        self.texts = [
            f"{j['title']} at {j['company']}. {j['description']} Skills: {', '.join(j['skills'])}"
            for j in self.jobs
        ]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.emb = self.model.encode(self.texts, normalize_embeddings=True)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None

    # --- retrievers -------------------------------------------------
    def sparse(self, query, n=30):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:n])

    def dense(self, query, n=30):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.emb @ q))[:n])

    @staticmethod
    def rrf(rankings, k=60):
        """Reciprocal Rank Fusion: robust way to merge rankings with incomparable scores."""
        fused = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return fused

    # --- personalisation -------------------------------------------
    @staticmethod
    def personal_score(job, profile):
        if not profile:
            return 0.0, []
        have = {s.lower() for s in profile.get("skills", [])}
        need = {s.lower() for s in job["skills"]}
        overlap = sorted(have & need)
        skill = len(overlap) / max(len(need), 1)
        loc = 1.0 if profile.get("location") and profile["location"].lower() in (job["location"].lower(), "remote") else 0.0
        exp = 1.0 if profile.get("years", 0) >= job["min_years"] else 0.3
        return 0.6 * skill + 0.2 * loc + 0.2 * exp, overlap

    # --- main entry -------------------------------------------------
    def search(self, query, profile=None, top_k=5, alpha=0.7):
        fused = self.rrf([self.sparse(query), self.dense(query)])
        cand = sorted(fused, key=fused.get, reverse=True)[:15]
        base = {i: fused[i] for i in cand}
        if self.reranker:
            ce = self.reranker.predict([(query, self.texts[i]) for i in cand])
            base = {i: float(s) for i, s in zip(cand, ce)}
        lo, hi = min(base.values()), max(base.values())
        norm = {i: (v - lo) / (hi - lo + 1e-9) for i, v in base.items()}
        out = []
        for i in cand:
            p, overlap = self.personal_score(self.jobs[i], profile)
            final = alpha * norm[i] + (1 - alpha) * p if profile else norm[i]
            out.append({**self.jobs[i], "score": round(final, 4), "relevance": round(norm[i], 4),
                        "personalisation": round(p, 4), "matched_skills": overlap})
        return sorted(out, key=lambda r: r["score"], reverse=True)[:top_k]
