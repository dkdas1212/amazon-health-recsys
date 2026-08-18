"""
db_logger.py
Logs each agent interaction (query, response, reasoning trace) to Postgres
for explainability / auditing. Failures here never break the chat —
logging is best-effort only.
"""

import os
import json
import psycopg2

PG_CONFIG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=os.getenv("PGPORT", "5433"),
    dbname=os.getenv("PGDATABASE", "recsys"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "postgres"),
)


def log_interaction(user_id: str, query: str, response: str, trace: list) -> None:
    """Best-effort insert into agent_logs. Never raises — a logging failure
    should never take down the chat experience."""
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_logs (user_id, query, response, trace) VALUES (%s, %s, %s, %s)",
                (user_id, query, response, json.dumps(trace)),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[db_logger] Failed to log interaction: {e}")