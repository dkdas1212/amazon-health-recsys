import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate

os.makedirs("models", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("recommendation-engine")

df = pd.read_csv("data/processed/clean_reviews.csv")
df = df[["user_id", "product_id", "rating"]]
print(f"Loaded {len(df):,} reviews")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# ── Experiment 1: Default SVD ──────────────────────────────
print("\nRunning Experiment 1: Default SVD...")
with mlflow.start_run(run_name="SVD_default"):
    params = {
        "n_factors": 100,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02
    }
    mlflow.log_params(params)

    model = SVD(**params, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("n_users", df['user_id'].nunique())
    mlflow.log_metric("n_products", df['product_id'].nunique())
    mlflow.log_metric("n_reviews", len(df))

    with open("models/svd_model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact("models/svd_model.pkl")

    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# ── Experiment 2: More factors ─────────────────────────────
print("\nRunning Experiment 2: SVD with more factors...")
with mlflow.start_run(run_name="SVD_large_factors"):
    params = {
        "n_factors": 200,
        "n_epochs": 25,
        "lr_all": 0.005,
        "reg_all": 0.02
    }
    mlflow.log_params(params)

    model2 = SVD(**params, random_state=42)
    model2.fit(trainset)
    predictions2 = model2.test(testset)

    rmse2 = accuracy.rmse(predictions2, verbose=False)
    mae2 = accuracy.mae(predictions2, verbose=False)

    mlflow.log_metric("rmse", rmse2)
    mlflow.log_metric("mae", mae2)

    print(f"  RMSE: {rmse2:.4f} | MAE: {mae2:.4f}")

# ── Experiment 3: More epochs ──────────────────────────────
print("\nRunning Experiment 3: SVD with more epochs...")
with mlflow.start_run(run_name="SVD_more_epochs"):
    params = {
        "n_factors": 100,
        "n_epochs": 40,
        "lr_all": 0.008,
        "reg_all": 0.02
    }
    mlflow.log_params(params)

    model3 = SVD(**params, random_state=42)
    model3.fit(trainset)
    predictions3 = model3.test(testset)

    rmse3 = accuracy.rmse(predictions3, verbose=False)
    mae3 = accuracy.mae(predictions3, verbose=False)

    mlflow.log_metric("rmse", rmse3)
    mlflow.log_metric("mae", mae3)

    print(f"  RMSE: {rmse3:.4f} | MAE: {mae3:.4f}")

# ── Summary ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("ALL EXPERIMENTS COMPLETE!")
print(f"Experiment 1 — Default:      RMSE {rmse:.4f}")
print(f"Experiment 2 — More factors: RMSE {rmse2:.4f}")
print(f"Experiment 3 — More epochs:  RMSE {rmse3:.4f}")
best_rmse = min(rmse, rmse2, rmse3)
print(f"\nBest RMSE: {best_rmse:.4f}")
print("=" * 50)
print("\nRun 'mlflow ui' to see the experiment dashboard!")