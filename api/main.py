import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import json
import time

app = FastAPI(
    title="E-Commerce Recommendation API",
    description="Hybrid recommendation engine using SVD + TF-IDF",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ─────────────────────────────────
print("Loading models...")
df = pd.read_csv("data/processed/clean_reviews.csv")

with open("models/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)
with open("models/cosine_sim_matrix.pkl", "rb") as f:
    cosine_sim = pickle.load(f)
with open("models/product_profiles.pkl", "rb") as f:
    product_profiles = pickle.load(f)

print("All models loaded! API ready.")

# ── In-memory cache (replaces Redis for now) ───────────────
cache = {}

def cache_get(key):
    if key in cache:
        data, timestamp = cache[key]
        if time.time() - timestamp < 3600:
            return data
    return None

def cache_set(key, value):
    cache[key] = (value, time.time())

# ── Helper functions ───────────────────────────────────────
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

def get_popular_products(n=10):
    popular = df.groupby('product_id').agg(
        review_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    popular = popular[popular['review_count'] >= 10]
    popular['score'] = (
        0.7 * popular['avg_rating'] +
        0.3 * (popular['review_count'] / popular['review_count'].max() * 5)
    )
    top = popular.nlargest(n, 'score')
    return [
        {
            "product_id": row['product_id'],
            "hybrid_score": round(row['score'], 4),
            "cf_score": round(row['avg_rating'], 4),
            "cb_score": round(row['review_count'] /
                             popular['review_count'].max() * 5, 4),
            "source": "popularity",
            "sample_summary": get_sample_summary(row['product_id'])
        }
        for _, row in top.iterrows()
    ]

def search_by_keyword(keywords, n=10):
    """Search the FULL dataset (not just top recommendations) for products
    whose reviews mention any of the given keywords, then rank matches by
    rating + review count. This is the real category search, unlike the
    old approach of filtering an already-narrow top-N list."""
    pattern = '|'.join(keywords)
    matches = df[df['summary'].str.contains(pattern, case=False, na=False)]
    if matches.empty:
        return []

    grouped = matches.groupby('product_id').agg(
        review_count=('rating', 'count'),
        avg_rating=('rating', 'mean'),
        sample_summary=('summary', 'first')
    ).reset_index()

    grouped['score'] = (
        0.7 * grouped['avg_rating'] +
        0.3 * (grouped['review_count'] / grouped['review_count'].max() * 5)
    )
    top = grouped.nlargest(n, 'score')
    return [
        {
            "product_id": row['product_id'],
            "avg_rating": round(row['avg_rating'], 2),
            "review_count": int(row['review_count']),
            "match_score": round(row['score'], 4),
            "sample_summary": row['sample_summary']
        }
        for _, row in top.iterrows()
    ]

def get_sample_summary(product_id):
    """Grab one representative review summary for a product, used as a
    weak signal for category (snack/drink/meal) since there's no
    structured category field in this dataset."""
    reviews = df[df['product_id'] == product_id]['summary'].dropna()
    return reviews.iloc[0] if not reviews.empty else ""

def get_hybrid_recommendations(user_id, n=10):
    user_reviews = df[df['user_id'] == user_id]
    if user_reviews.empty:
        return get_popular_products(n)

    reviewed = set(user_reviews['product_id'].tolist())
    candidates = set(df['product_id'].unique()) - reviewed

    cf_scores = {p: svd_model.predict(user_id, p).est for p in candidates}

    top_rated = user_reviews[
        user_reviews['rating'] >= 4
    ]['product_id'].tolist()
    cb_scores = {}
    for liked in top_rated[:5]:
        for similar_product, sim_score in get_content_similar(liked, n=20):
            if similar_product in candidates:
                cb_scores[similar_product] = cb_scores.get(
                    similar_product, 0) + sim_score

    if cb_scores:
        max_cb = max(cb_scores.values())
        min_cb = min(cb_scores.values())
        for pid in cb_scores:
            if max_cb != min_cb:
                cb_scores[pid] = 1 + 4 * (
                    cb_scores[pid] - min_cb) / (max_cb - min_cb)
            else:
                cb_scores[pid] = 3.0

    hybrid = {}
    for pid in candidates:
        cf = cf_scores.get(pid, 3.0)
        cb = cb_scores.get(pid, 3.0)
        hybrid[pid] = (0.7 * cf) + (0.3 * cb)

    top_n = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:n]
    return [
        {
            "product_id": pid,
            "hybrid_score": round(score, 4),
            "cf_score": round(cf_scores.get(pid, 3.0), 4),
            "cb_score": round(cb_scores.get(pid, 3.0), 4),
            "source": "hybrid",
            "sample_summary": get_sample_summary(pid)
        }
        for pid, score in top_n
    ]

# ── API Endpoints ──────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Recommendation API is running!",
        "endpoints": {
            "recommend": "/recommend?user_id=USER_ID&n=10",
            "similar": "/similar/{product_id}?n=10",
            "popular": "/popular?n=10",
            "user_history": "/user/{user_id}/history",
            "stats": "/stats"
        }
    }

@app.get("/recommend")
def recommend(user_id: str, n: int = 10):
    cache_key = f"recs:{user_id}:{n}"
    cached = cache_get(cache_key)
    if cached:
        return {"user_id": user_id, "recommendations": cached,
                "source": "cache", "count": len(cached)}

    user_exists = user_id in df['user_id'].values
    if not user_exists:
        recs = get_popular_products(n)
        return {"user_id": user_id, "recommendations": recs,
                "source": "popularity_fallback", "count": len(recs)}

    recs = get_hybrid_recommendations(user_id, n)
    cache_set(cache_key, recs)
    return {"user_id": user_id, "recommendations": recs,
            "source": "hybrid", "count": len(recs)}

@app.get("/similar/{product_id}")
def similar_products(product_id: str, n: int = 10):
    cache_key = f"similar:{product_id}:{n}"
    cached = cache_get(cache_key)
    if cached:
        return {"product_id": product_id, "similar": cached,
                "source": "cache"}

    if product_id not in product_profiles['product_id'].values:
        raise HTTPException(status_code=404,
                           detail=f"Product {product_id} not found")

    similar = get_content_similar(product_id, n)
    result = [
        {
            "product_id": pid,
            "similarity_score": round(score, 4),
            "avg_rating": round(float(
                product_profiles[
                    product_profiles['product_id'] == pid
                ]['avg_rating'].values[0]), 2),
            "review_count": int(
                product_profiles[
                    product_profiles['product_id'] == pid
                ]['review_count'].values[0]),
            "sample_summary": get_sample_summary(pid)
        }
        for pid, score in similar
    ]
    cache_set(cache_key, result)
    return {"product_id": product_id, "similar": result,
            "source": "content_based"}

@app.get("/popular")
def popular_products(n: int = 10):
    return {"popular": get_popular_products(n), "count": n}

@app.get("/user/{user_id}/history")
def user_history(user_id: str):
    user_reviews = df[df['user_id'] == user_id]
    if user_reviews.empty:
        raise HTTPException(status_code=404,
                           detail=f"User {user_id} not found")
    history = user_reviews[['product_id', 'rating', 'summary']]\
        .sort_values('rating', ascending=False)\
        .head(20)\
        .to_dict(orient='records')
    return {
        "user_id": user_id,
        "total_reviews": len(user_reviews),
        "avg_rating_given": round(user_reviews['rating'].mean(), 2),
        "history": history
    }

@app.get("/stats")
def stats():
    return {
        "total_users": int(df['user_id'].nunique()),
        "total_products": int(df['product_id'].nunique()),
        "total_reviews": len(df),
        "avg_rating": round(float(df['rating'].mean()), 2),
        "sparsity": "99.96%",
        "models": ["SVD Collaborative Filtering",
                   "TF-IDF Content Based",
                   "Hybrid (70% CF + 30% CB)"]
    }

@app.get("/search")
def search(keywords: str, n: int = 10):
    """Search all reviews for keyword matches (comma-separated), ranked by rating.
    Example: /search?keywords=coffee,tea,espresso&n=10"""
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    if not kw_list:
        raise HTTPException(status_code=400, detail="Provide at least one keyword")
    results = search_by_keyword(kw_list, n)
    return {"keywords": kw_list, "results": results, "count": len(results)}