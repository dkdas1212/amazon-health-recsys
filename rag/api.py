"""Add to your existing FastAPI app:
    from rag.api import router as rag_router
    app.include_router(rag_router)
(If your app imports modules without the package prefix, use `from api import router`.)
"""
import os
from typing import Optional
from collections import defaultdict
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from common import connect, embed

router = APIRouter(prefix="/rag", tags=["rag"])
llm = OpenAI(api_key=os.getenv("LLM_API_KEY", "missing"),
             base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MIN_SIM = float(os.getenv("MIN_SIMILARITY", 0.30))
TOP_K = int(os.getenv("TOP_K", 6))

SYSTEM_PROMPT = """You answer shopper questions about food products using ONLY the numbered
customer reviews provided.
Rules:
1. Base every claim on the reviews and cite them like [1], [2].
2. If reviews disagree, say so and give both sides.
3. If the reviews do not answer the question, say there is not enough review evidence. Do not guess.
4. Answer the exact question asked. If the reviews only speak to something related
   (for example taste when asked about freshness), say that instead of answering yes or no.
5. Cite with plain square brackets like [1], never other bracket styles. Quote reviewers
   only with exact words that appear in the reviews.
6. Keep the answer under 120 words, in plain language."""

FALLBACK = "There isn't enough review evidence to answer that reliably."

def _search(question, k, product_id=None):
    q = embed([question])[0]
    where, params = "", [q]
    if product_id:
        where, params = "WHERE product_id = %s", [q, product_id]
    sql = (f"SELECT product_id, rating, content, 1 - (embedding <=> %s) AS sim "
           f"FROM review_chunks {where} ORDER BY embedding <=> %s LIMIT %s")
    params = [params[0]] + params[1:] + [q, k]
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()

class Ask(BaseModel):
    question: str
    product_id: Optional[str] = None

@router.post("/ask")
def ask(body: Ask):
    hits = _search(body.question, TOP_K, body.product_id)
    best = float(hits[0][3]) if hits else 0.0
    if best < MIN_SIM:
        return {"answer": FALLBACK, "fallback": True, "confidence": round(best, 3), "sources": []}
    context = "\n".join(f"[{i+1}] ({h[1]}/5 stars) {h[2][:500]}" for i, h in enumerate(hits))
    r = llm.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Reviews:\n{context}\n\nQuestion: {body.question}"}])
    return {"answer": r.choices[0].message.content, "fallback": False, "confidence": round(best, 3),
            "sources": [{"n": i+1, "product_id": h[0], "rating": h[1],
                         "snippet": h[2][:200], "similarity": round(float(h[3]), 3)}
                        for i, h in enumerate(hits)]}

@router.get("/search")
def search(q: str, k: int = 5):
    """Semantic product search, e.g. q='low sugar snack for kids'."""
    hits = _search(q, 60)
    best = defaultdict(float)
    for pid, _, _, sim in hits:
        best[pid] = max(best[pid], float(sim))
    top = sorted(best.items(), key=lambda x: -x[1])[:k]
    return [{"product_id": p, "similarity": round(s, 3)} for p, s in top]