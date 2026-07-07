import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/health_reviews.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Total reviews:   {len(df):,}")
print(f"Unique users:    {df['user_id'].nunique():,}")
print(f"Unique products: {df['product_id'].nunique():,}")
print(f"\nColumn info:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("RATING DISTRIBUTION")
print("=" * 50)
print(df['rating'].value_counts().sort_index())

print("\n" + "=" * 50)
print("TOP 10 MOST REVIEWED PRODUCTS")
print("=" * 50)
print(df['product_id'].value_counts().head(10))

print("\n" + "=" * 50)
print("TOP 10 MOST ACTIVE USERS")
print("=" * 50)
print(df['user_id'].value_counts().head(10))

print("\n" + "=" * 50)
print("REVIEW LENGTH ANALYSIS")
print("=" * 50)
df['review_length'] = df['text'].astype(str).apply(len)
print(f"Average review length: {df['review_length'].mean():.0f} characters")
print(f"Shortest review:       {df['review_length'].min()} characters")
print(f"Longest review:        {df['review_length'].max()} characters")

# ── Plot 1: Rating distribution ────────────────────────────
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='rating', palette='Blues_d')
plt.title('Rating Distribution')
plt.xlabel('Rating (1–5 stars)')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.savefig('data/processed/plot1_rating_distribution.png')
plt.show()
print("\nPlot 1 saved!")

# ── Plot 2: Reviews per user (how active are users?) ───────
plt.figure(figsize=(8, 4))
reviews_per_user = df['user_id'].value_counts()
sns.histplot(reviews_per_user[reviews_per_user <= 20], bins=20, color='steelblue')
plt.title('How many reviews does each user write?')
plt.xlabel('Number of reviews')
plt.ylabel('Number of users')
plt.tight_layout()
plt.savefig('data/processed/plot2_reviews_per_user.png')
plt.show()
print("Plot 2 saved!")

# ── Plot 3: Reviews per product ────────────────────────────
plt.figure(figsize=(8, 4))
reviews_per_product = df['product_id'].value_counts()
sns.histplot(reviews_per_product[reviews_per_product <= 50], bins=30, color='coral')
plt.title('How many reviews does each product get?')
plt.xlabel('Number of reviews')
plt.ylabel('Number of products')
plt.tight_layout()
plt.savefig('data/processed/plot3_reviews_per_product.png')
plt.show()
print("Plot 3 saved!")

print("\n" + "=" * 50)
print("EDA COMPLETE!")
print("3 charts saved to data/processed/")
print("=" * 50)

