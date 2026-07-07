import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://recsys_user:recsys123@localhost:5432/recsys"
engine = create_engine(DB_URL)

df = pd.read_csv("data/processed/clean_reviews.csv")
print(f"Loading {len(df):,} clean reviews into PostgreSQL...")

df.to_sql("clean_reviews", engine, if_exists="replace", index=False)
print("Done! Table 'clean_reviews' created.")

with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM clean_reviews")).fetchone()[0]
    print(f"Rows in database: {count:,}")

    print("\nTop 5 most reviewed products:")
    result = pd.read_sql("""
        SELECT product_id,
               COUNT(*) as review_count,
               ROUND(AVG(rating)::numeric, 2) as avg_rating
        FROM clean_reviews
        GROUP BY product_id
        ORDER BY review_count DESC
        LIMIT 5
    """, engine)
    print(result)

    print("\nSparsity check:")
    result = pd.read_sql("""
        SELECT 
            COUNT(DISTINCT user_id) as users,
            COUNT(DISTINCT product_id) as products,
            COUNT(*) as reviews
        FROM clean_reviews
    """, engine)
    print(result)