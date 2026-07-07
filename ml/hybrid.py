import pandas as pd
import pickle
import os

os.makedirs("models", exist_ok=True)

# ── Load everything ────────────────────────────────────────
print("Loading models and data...")
df = pd.read_csv("data/processed/clean_reviews.csv")

with open("models/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

with open("models/cosine_sim_matrix.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

with open("models/product_profiles.pkl", "rb") as f:
    product_profiles = pickle.load(f)

print("All models loaded!")

# ── Helper: content-based similar products ─────────────────
def get_content_similar(product_id, n=20):
    if product_id not in product_profiles['product_id'].values:
        return []
    idx = product_profiles[
        product_profiles['product_id'] == product_id
    ].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    return [
        (product_profiles.iloc[i]['product_id'], score)
        for i, score in sim_scores
    ]

# ── Hybrid recommendations ─────────────────────────────────
def get_hybrid_recommendations(user_id, n=10, cf_weight=0.7, cb_weight=0.3):
    """
    Blends collaborative filtering (SVD) and content-based (TF-IDF).
    cf_weight: how much to trust SVD predictions (default 70%)
    cb_weight: how much to trust content similarity (default 30%)
    """
    user_reviews = df[df['user_id'] == user_id]
    if user_reviews.empty:
        print(f"User {user_id} not found — returning popular products")
        return get_popular_products(n)

    reviewed_products = set(user_reviews['product_id'].tolist())
    all_products = set(df['product_id'].unique())
    candidates = all_products - reviewed_products

    # ── Step 1: SVD score for all candidates ──────────────
    cf_scores = {}
    for product_id in candidates:
        pred = svd_model.predict(user_id, product_id)
        cf_scores[product_id] = pred.est

    # ── Step 2: Content-based score ────────────────────────
    # Find products similar to what the user rated highly
    top_rated = user_reviews[user_reviews['rating'] >= 4]['product_id'].tolist()
    cb_scores = {}
    for liked_product in top_rated[:5]:
        similar = get_content_similar(liked_product, n=20)
        for similar_product, sim_score in similar:
            if similar_product in candidates:
                if similar_product not in cb_scores:
                    cb_scores[similar_product] = 0
                cb_scores[similar_product] += sim_score

    # Normalize cb_scores to 1-5 scale
    if cb_scores:
        max_cb = max(cb_scores.values())
        min_cb = min(cb_scores.values())
        for pid in cb_scores:
            if max_cb != min_cb:
                cb_scores[pid] = 1 + 4 * (cb_scores[pid] - min_cb) / (max_cb - min_cb)
            else:
                cb_scores[pid] = 3.0

    # ── Step 3: Blend scores ───────────────────────────────
    hybrid_scores = {}
    for product_id in candidates:
        cf = cf_scores.get(product_id, 3.0)
        cb = cb_scores.get(product_id, 3.0)
        hybrid_scores[product_id] = (cf_weight * cf) + (cb_weight * cb)

    # ── Step 4: Return top N ───────────────────────────────
    top_n = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [
        {
            "product_id": pid,
            "hybrid_score": round(score, 4),
            "cf_score": round(cf_scores.get(pid, 3.0), 4),
            "cb_score": round(cb_scores.get(pid, 3.0), 4)
        }
        for pid, score in top_n
    ]

# ── Fallback: popular products for new users ───────────────
def get_popular_products(n=10):
    popular = df.groupby('product_id').agg(
        review_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    popular = popular[popular['review_count'] >= 10]
    popular['popularity_score'] = (
        0.7 * popular['avg_rating'] +
        0.3 * (popular['review_count'] / popular['review_count'].max() * 5)
    )
    top = popular.nlargest(n, 'popularity_score')
    return [
        {
            "product_id": row['product_id'],
            "hybrid_score": round(row['popularity_score'], 4),
            "cf_score": round(row['avg_rating'], 4),
            "cb_score": round(row['review_count'] / popular['review_count'].max() * 5, 4)
        }
        for _, row in top.iterrows()
    ]

# ── Test the hybrid model ──────────────────────────────────
print("\nTesting hybrid recommendations...")

# Test with most active user
test_user = df['user_id'].value_counts().index[0]
print(f"\nUser: {test_user}")
print(f"Reviews written: {len(df[df['user_id'] == test_user])}")

recs = get_hybrid_recommendations(test_user, n=10)
print(f"\nTop 10 hybrid recommendations:")
for i, rec in enumerate(recs, 1):
    print(f"  {i}. {rec['product_id']} — "
          f"hybrid: {rec['hybrid_score']:.4f} | "
          f"cf: {rec['cf_score']:.4f} | "
          f"cb: {rec['cb_score']:.4f}")

# Test fallback with unknown user
print("\nTesting fallback (unknown user)...")
fallback = get_popular_products(n=5)
print("Top 5 popular products:")
for i, rec in enumerate(fallback, 1):
    print(f"  {i}. {rec['product_id']} — score: {rec['hybrid_score']:.4f}")

# ── Save hybrid function ───────────────────────────────────
print("\nSaving hybrid model components...")
with open("models/product_profiles.pkl", "wb") as f:
    pickle.dump(product_profiles, f)

print("\n" + "=" * 50)
print("HYBRID MODEL COMPLETE!")
print("CF weight: 70% | CB weight: 30%")
print("=" * 50)