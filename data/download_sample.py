import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("Loading Amazon reviews dataset...")
df = pd.read_csv("data/raw/Reviews.csv")

print(f"\nRaw data shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# Keep only what we need
reviews = df[["UserId", "ProductId", "Score", "Summary", "Text"]].copy()
reviews.columns = ["user_id", "product_id", "rating", "summary", "text"]

# Basic cleaning
reviews = reviews.dropna(subset=["user_id", "product_id", "rating"])
reviews = reviews.drop_duplicates()
reviews["rating"] = reviews["rating"].astype(int)

print(f"\nAfter cleaning: {len(reviews):,} reviews")
print(f"Unique users:    {reviews['user_id'].nunique():,}")
print(f"Unique products: {reviews['product_id'].nunique():,}")
print(f"Rating distribution:\n{reviews['rating'].value_counts().sort_index()}")

# Save cleaned version
reviews.to_csv("data/raw/health_reviews.csv", index=False)
print(f"\nSaved to data/raw/health_reviews.csv")
print("\nAll good! Ready for EDA.")