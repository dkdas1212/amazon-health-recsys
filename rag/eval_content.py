"""Content-based comparison: TF-IDF vs sentence embeddings vs popularity.

For each user with >=3 liked products (rating>=4): hide one liked product,
build a profile from the user's other liked products, rank all products by
similarity, and check whether the hidden product is in the top 10 (hit@10).
Products the user already reviewed (other than the hidden one) are excluded.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

K = 10
SEED = 42

def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def evaluate(df, emb, k=K, seed=SEED):
    """Leak-free comparison. A product's vector is the sum of its review vectors,
    but when scoring a user we SUBTRACT that user's own reviews from every
    product vector involved (profile items and the hidden item), so a user's
    own wording cannot match them to the hidden product.
    df: user_id, product_id, rating, content. emb: (len(df), d) normalized."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    prods = df["product_id"].unique()
    pidx = {p: i for i, p in enumerate(prods)}
    codes = df["product_id"].map(pidx).to_numpy()
    n_rev, n_p = len(df), len(prods)

    Xt = TfidfVectorizer(stop_words="english", max_features=20000, min_df=2,
                         sublinear_tf=True).fit_transform(df["content"]).tocsr()
    Agg = csr_matrix((np.ones(n_rev), (codes, np.arange(n_rev))), shape=(n_p, n_rev))
    ST = (Agg @ Xt).tocsr()            # per-product sum of review tf-idf vectors
    SE = np.asarray(Agg @ emb)         # per-product sum of review embeddings
    ST_n, SE_n = sk_normalize(ST), _norm(SE)
    pop = np.bincount(codes, minlength=n_p).astype(float)

    hits = {"popularity": 0, "tfidf": 0, "embeddings": 0}
    n = 0
    for _, idx in df.groupby("user_id").indices.items():
        g = df.iloc[idx]
        liked = g.loc[g.rating >= 4, "product_id"].map(pidx).unique()
        if len(liked) < 3:
            continue
        held = int(rng.choice(liked))
        profile = [int(p) for p in liked if p != held]
        seen = [int(p) for p in g["product_id"].map(pidx).unique() if p != held]
        own = {}
        for pos in idx:
            own.setdefault(int(codes[pos]), []).append(pos)

        def adj_e(p):
            return SE[p] - emb[own[p]].sum(axis=0) if p in own else SE[p]
        def adj_t(p):
            v = ST[p].toarray().ravel()
            return v - np.asarray(Xt[own[p]].sum(axis=0)).ravel() if p in own else v

        pe = [_norm(adj_e(p)) for p in profile]
        pt = [adj_t(p) for p in profile]
        pt = [v / (np.linalg.norm(v) + 1e-12) for v in pt]
        qe, qt = np.mean(pe, axis=0), np.mean(pt, axis=0)
        if np.linalg.norm(qe) < 1e-9 or np.linalg.norm(qt) < 1e-9:
            continue
        qe, qt = _norm(qe), _norm(qt)
        he, ht = adj_e(held), adj_t(held)

        s_e = SE_n @ qe
        s_e[held] = he @ qe / (np.linalg.norm(he) + 1e-12)
        s_t = np.asarray(ST_n @ qt).ravel()
        s_t[held] = ht @ qt / (np.linalg.norm(ht) + 1e-12)
        s_p = pop.copy()
        s_p[held] -= len(own.get(held, []))
        n += 1

        def hit(scores):
            scores = scores.copy()
            scores[seen] = -np.inf
            return held in np.argpartition(-scores, k)[:k]

        hits["popularity"] += hit(s_p)
        hits["tfidf"] += hit(s_t)
        hits["embeddings"] += hit(s_e)
    return n, {m: v / n for m, v in hits.items()}, n_p

def main():
    from common import connect
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT user_id, product_id, rating, content, embedding FROM review_chunks")
        rows = cur.fetchall()
    df = pd.DataFrame([r[:4] for r in rows], columns=["user_id", "product_id", "rating", "content"])
    emb = np.vstack([r[4] for r in rows])
    n, res, n_prod = evaluate(df, emb)
    print(f"Users evaluated: {n:,}   Products: {n_prod:,}   hit@{K}")
    for m, v in res.items():
        print(f"  {m:<11} {v:.4f}")
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002"))
        mlflow.set_experiment("content_based_comparison")
        for m, v in res.items():
            with mlflow.start_run(run_name=m):
                mlflow.log_metric(f"hit_at_{K}", v)
                mlflow.log_param("users_evaluated", n)
    except Exception as e:
        print("MLflow logging skipped:", e)

if __name__ == "__main__":
    main()