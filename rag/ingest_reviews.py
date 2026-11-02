"""Sample reviews from clean_reviews, embed them, store in pgvector.

Samples N_USERS random users who have >= MIN_REVIEWS reviews (so the
evaluation in eval_content.py has enough history per user).
Run once:  python rag/ingest_reviews.py
"""
import os
import pandas as pd
from psycopg2.extras import execute_values
from common import connect, embed

# --- adjust these to match `\d clean_reviews` in psql ---
TABLE, USER, PRODUCT, RATING, SUMMARY, TEXT = (
    "clean_reviews", "user_id", "product_id", "rating", "summary", "text")
# --------------------------------------------------------
N_USERS = int(os.getenv("N_USERS", 8000))
MIN_REVIEWS = int(os.getenv("MIN_REVIEWS", 5))
MAX_CHARS = 1000

def main():
    with connect(vector=False) as conn:
        df = pd.read_sql(f"""
            SELECT {USER} AS user_id, {PRODUCT} AS product_id, {RATING} AS rating,
                   {SUMMARY} AS summary, {TEXT} AS text
            FROM {TABLE}
            WHERE {USER} IN (
                SELECT {USER} FROM {TABLE} GROUP BY {USER}
                HAVING COUNT(*) >= {MIN_REVIEWS} ORDER BY random() LIMIT {N_USERS})
        """, conn)
    df["content"] = (df["summary"].fillna("") + ". " + df["text"].fillna("")).str.slice(0, MAX_CHARS)
    # Fine Food Reviews has copy-pasted reviews shared across product variants.
    # Drop them, otherwise sibling variants look "similar" and inflate any eval.
    multi = df.groupby("content")["product_id"].transform("nunique") > 1
    print(f"Dropping {int(multi.sum()):,} reviews whose text appears on multiple products")
    df = df[~multi]
    df = df[df.groupby("user_id")["user_id"].transform("size") >= MIN_REVIEWS]
    print(f"Embedding {len(df):,} reviews from {df.user_id.nunique():,} users "
          f"and {df.product_id.nunique():,} products ...")
    vecs = embed(df["content"].tolist(), progress=True)
    rows = list(zip(df.user_id, df.product_id, df.rating.astype(int), df.content, list(vecs)))
    schema = open(os.path.join(os.path.dirname(__file__), "schema.sql")).read()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(schema)
        execute_values(cur, "INSERT INTO review_chunks (user_id, product_id, rating, content, embedding) VALUES %s",
                       rows, page_size=1000)
    print("Done.")

if __name__ == "__main__":
    main()