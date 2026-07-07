import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/processed/clean_reviews.csv")
print(f"Shape: {df.shape}")

# Rating distribution
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='rating', hue='rating', palette='Blues_d', legend=False)
plt.title('Rating Distribution (Clean Data)')
plt.xlabel('Rating (1–5 stars)')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.savefig('data/processed/plot_clean_ratings.png')
plt.close()
print("Plot 1 saved!")

# User activity
plt.figure(figsize=(8, 4))
df['user_review_count'].clip(upper=50).hist(bins=50, color='steelblue')
plt.title('User Activity — Reviews per User')
plt.xlabel('Number of reviews')
plt.ylabel('Number of users')
plt.tight_layout()
plt.savefig('data/processed/plot_user_activity.png')
plt.close()
print(f"Plot 2 saved!")
print(f"Average reviews per user: {df['user_review_count'].mean():.1f}")
print(f"Most active user: {df['user_review_count'].max()} reviews")

# Product popularity
plt.figure(figsize=(8, 4))
product_counts = df.groupby('product_id')['rating'].count()
product_counts.clip(upper=100).hist(bins=50, color='coral')
plt.title('Product Popularity — Reviews per Product')
plt.xlabel('Number of reviews')
plt.ylabel('Number of products')
plt.tight_layout()
plt.savefig('data/processed/plot_product_popularity.png')
plt.close()
print(f"Plot 3 saved!")
print(f"Average reviews per product: {product_counts.mean():.1f}")
print(f"Most reviewed product: {product_counts.max()} reviews")

# Sparsity
n_users = df['user_id'].nunique()
n_products = df['product_id'].nunique()
n_reviews = len(df)
total_possible = n_users * n_products
sparsity = 1 - (n_reviews / total_possible)

print("\n" + "=" * 50)
print("SPARSITY ANALYSIS")
print("=" * 50)
print(f"Users:          {n_users:,}")
print(f"Products:       {n_products:,}")
print(f"Reviews:        {n_reviews:,}")
print(f"Total possible: {total_possible:,}")
print(f"Sparsity:       {sparsity:.4%}")