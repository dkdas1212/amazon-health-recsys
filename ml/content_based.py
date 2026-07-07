import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

os.makedirs("models", exist_ok=True)

# ── Load data ──────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/processed/clean_reviews.csv")

# ── Build product profiles ─────────────────────────────────
# Combine all reviews for each product into one text document
print("Building product profiles from reviews...")
product_profiles = df.groupby('product_id')['text'].apply(
    lambda reviews: ' '.join(reviews.astype(str))
).reset_index()
product_profiles.columns = ['product_id', 'combined_text']
print(f"Product profiles built: {len(product_profiles):,}")

# ── Add average rating to profiles ────────────────────────
product_stats = df.groupby('product_id').agg(
    avg_rating=('rating', 'mean'),
    review_count=('rating', 'count')
).reset_index()
product_profiles = product_profiles.merge(product_stats, on='product_id')

# ── TF-IDF vectorization ───────────────────────────────────
print("\nRunning TF-IDF vectorization...")
print("This may take 2-3 minutes...")
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf.fit_transform(product_profiles['combined_text'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# ── Compute cosine similarity ──────────────────────────────
print("\nComputing cosine similarity matrix...")
print("This may take a few minutes...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Similarity matrix shape: {cosine_sim.shape}")

# ── Function to get similar products ──────────────────────
def get_similar_products(product_id, n=10):
    if product_id not in product_profiles['product_id'].values:
        return []
    idx = product_profiles[product_profiles['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    similar_products = []
    for i, score in sim_scores:
        pid = product_profiles.iloc[i]['product_id']
        avg_r = product_profiles.iloc[i]['avg_rating']
        r_count = product_profiles.iloc[i]['review_count']
        similar_products.append({
            'product_id': pid,
            'similarity_score': round(score, 4),
            'avg_rating': round(avg_r, 2),
            'review_count': int(r_count)
        })
    return similar_products

# ── Test with most reviewed product ───────────────────────
test_product = df['product_id'].value_counts().index[0]
print(f"\nFinding products similar to: {test_product}")
similar = get_similar_products(test_product, n=10)

print(f"\nTop 10 similar products:")
for i, p in enumerate(similar, 1):
    print(f"  {i}. {p['product_id']} — "
          f"similarity: {p['similarity_score']:.4f} — "
          f"avg rating: {p['avg_rating']} — "
          f"reviews: {p['review_count']}")

# ── Save everything ────────────────────────────────────────
print("\nSaving models...")
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("models/cosine_sim_matrix.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)
with open("models/product_profiles.pkl", "wb") as f:
    pickle.dump(product_profiles, f)

print("Saved:")
print("  models/tfidf_vectorizer.pkl")
print("  models/cosine_sim_matrix.pkl")
print("  models/product_profiles.pkl")

print("\n" + "=" * 50)
print("CONTENT-BASED FILTERING COMPLETE!")
print("=" * 50)