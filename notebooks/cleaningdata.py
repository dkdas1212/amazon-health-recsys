import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("data/processed", exist_ok=True)

print("Loading raw data...")
df = pd.read_csv("data/raw/health_reviews.csv")
print(f"Raw shape: {df.shape}")

# ── Step 1: Drop missing summaries ─────────────────────────
df = df.dropna(subset=["summary"])
print(f"\nAfter dropping missing summaries: {len(df):,} rows")

# ── Step 2: Remove duplicate reviews ───────────────────────
df = df.drop_duplicates(subset=["user_id", "product_id"])
print(f"After removing duplicates: {len(df):,} rows")

# ── Step 3: Filter out fake/lazy reviewers ─────────────────
# Keep only users with at least 3 reviews (cold start filter)
user_counts = df['user_id'].value_counts()
valid_users = user_counts[user_counts >= 3].index
df = df[df['user_id'].isin(valid_users)]
print(f"After filtering low-activity users: {len(df):,} rows")

# ── Step 4: Filter out products with too few reviews ───────
product_counts = df['product_id'].value_counts()
valid_products = product_counts[product_counts >= 3].index
df = df[df['product_id'].isin(valid_products)]
print(f"After filtering low-review products: {len(df):,} rows")

# ── Step 5: Feature engineering ────────────────────────────
print("\nEngineering features...")

# Review length
df['review_length'] = df['text'].astype(str).apply(len)

# Is it a positive review?
df['is_positive'] = (df['rating'] >= 4).astype(int)

# Average rating per product
product_avg = df.groupby('product_id')['rating'].mean().rename('product_avg_rating')
df = df.merge(product_avg, on='product_id', how='left')

# Number of reviews per user
user_review_count = df.groupby('user_id')['rating'].count().rename('user_review_count')
df = df.merge(user_review_count, on='user_id', how='left')

# Average rating given by each user
user_avg = df.groupby('user_id')['rating'].mean().rename('user_avg_rating')
df = df.merge(user_avg, on='user_id', how='left')

print(f"\nNew columns added:")
print(df[['user_id', 'product_id', 'rating', 'review_length',
          'is_positive', 'product_avg_rating',
          'user_review_count', 'user_avg_rating']].head(5))

# ── Step 6: Save cleaned data ──────────────────────────────
df.to_csv("data/processed/clean_reviews.csv", index=False)
print(f"\nClean data saved to data/processed/clean_reviews.csv")
print(f"Final shape: {df.shape}")
print(f"Unique users:    {df['user_id'].nunique():,}")
print(f"Unique products: {df['product_id'].nunique():,}")

# ── Step 7: Plot review length distribution ────────────────
plt.figure(figsize=(8, 4))
df[df['review_length'] <= 2000]['review_length'].hist(bins=50, color='steelblue')
plt.title('Review Length Distribution')
plt.xlabel('Characters')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.savefig('data/processed/plot4_review_lengths.png')
plt.close()
print("Plot saved!")

print("\n" + "=" * 50)
print("WEEK 2 CLEANING COMPLETE!")
print("=" * 50)