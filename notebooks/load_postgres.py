import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://recsys_user:recsys123@localhost:5432/recsys"
engine = create_engine(DB_URL)

# Test connection
with engine.connect() as conn:
    result = conn.execute(text("SELECT version();"))
    print(f"Connected! {result.fetchone()[0]}")

# Load cleaned data
df = pd.read_csv("data/raw/health_reviews.csv")
print(f"\nLoading {len(df):,} reviews into PostgreSQL...")

# Save to PostgreSQL
df.to_sql("raw_reviews", engine, if_exists="replace", index=False)
print("Done! Table 'raw_reviews' created.")

# Verify
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM raw_reviews")).fetchone()[0]
    print(f"Rows in database: {count:,}")
    
    print("\nSample from database:")
    sample = pd.read_sql("SELECT * FROM raw_reviews LIMIT 5", engine)
    print(sample[["user_id", "product_id", "rating"]])