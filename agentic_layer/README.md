# Agentic Layer for amazon-health-recsys

This scaffold adds a LangGraph agent on top of your **existing** FastAPI
recsys service. No new ML models, no new data pipeline — it only calls
what you already built.

## Files
- `tools.py` — wraps your FastAPI endpoints (`/recommend/svd`,
  `/recommend/content`, `/recommend/hybrid`) plus a read-only SQL tool
  over your dbt marts in Postgres.
- `agent.py` — LangGraph graph: `parse_intent -> select_tool ->
  apply_constraints -> format_response`. Uses Groq as the LLM.
- `streamlit_chat.py` — chat UI that replaces/augments your current
  Streamlit dashboard.

## Integration steps
1. **Match your real endpoints.** Open `tools.py` and update
   `BASE_URL` and the route paths/params (`/recommend/svd`, etc.) to
   whatever your FastAPI app actually exposes. If your responses use
   different field names than `name`/`price`, update `apply_constraints`
   in `agent.py` to match.
2. **Set env vars:**
   ```bash
   export GROQ_API_KEY=your_key_here
   export RECSYS_API_URL=http://localhost:8000
   export PGHOST=localhost PGPORT=5433 PGDATABASE=recsys PGUSER=postgres PGPASSWORD=postgres
   ```
3. **Install deps:**
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```
4. **Run your existing services first** (FastAPI + Postgres), then:
   ```bash
   streamlit run streamlit_chat.py
   ```
5. **(Optional) Log traces.** The `trace` list returned per query is
   ready to insert into a Postgres `agent_logs` table if you want a
   persistent explainability record — useful to show in interviews.

## Why this design
- LLM only does intent parsing, tool choice, constraint checks, and
  the final explanation — the actual recommendation math stays in your
  already-validated SVD/TF-IDF/hybrid models. This keeps the agent
  fast, cheap, and easy to reason about (no hallucinated numbers).
- The `apply_constraints` node is a simple guardrail: budget, excluded
  terms (e.g. allergens), category — rejects/filters before the answer
  ever reaches the user.
- Everything else in your stack (Kafka, Airflow, PySpark, dbt,
  MLflow, Docker Compose) is untouched.
