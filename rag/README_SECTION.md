## Semantic Search + Review Q&A (rag/)
- Embeds a sample of reviews with MiniLM into pgvector.
- `GET /rag/search?q=...` semantic product search; `POST /rag/ask` answers questions from reviews with citations and a low-evidence fallback.
- `python rag/eval_content.py` compares popularity, TF-IDF and embeddings (hit@10), logged to MLflow.
