import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pickle
import os

os.makedirs("models", exist_ok=True)

# ── Load data ──────────────────────────────────────────────
print("Loading clean data...")
df = pd.read_csv("data/processed/clean_reviews.csv")
df = df[["user_id", "product_id", "rating"]]
print(f"Loaded {len(df):,} reviews")

# ── Prepare data for Surprise ──────────────────────────────
print("\nPreparing data for SVD...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

# ── Train/test split ───────────────────────────────────────
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
print(f"Training on: {trainset.n_ratings:,} reviews")
print(f"Testing on:  {len(testset):,} reviews")

# ── Train SVD model ────────────────────────────────────────
print("\nTraining SVD model...")
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
model.fit(trainset)
print("Training complete!")

# ── Evaluate ───────────────────────────────────────────────
print("\nEvaluating on test set...")
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# ── Cross validation ───────────────────────────────────────
print("\nRunning 3-fold cross validation...")
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print(f"\nCross validation RMSE: {cv_results['test_rmse'].mean():.4f}")
print(f"Cross validation MAE:  {cv_results['test_mae'].mean():.4f}")

# ── Generate recommendations for a sample user ────────────
print("\nGenerating sample recommendations...")
sample_user = df['user_id'].value_counts().index[0]
print(f"Sample user: {sample_user}")

already_reviewed = df[df['user_id'] == sample_user]['product_id'].tolist()
print(f"Products already reviewed: {len(already_reviewed)}")

all_products = df['product_id'].unique()
not_reviewed = [p for p in all_products if p not in already_reviewed]

predictions_list = [(p, model.predict(sample_user, p).est) for p in not_reviewed]
predictions_list.sort(key=lambda x: x[1], reverse=True)
top_10 = predictions_list[:10]

print(f"\nTop 10 recommendations for {sample_user}:")
for i, (product, score) in enumerate(top_10, 1):
    print(f"  {i}. {product} — predicted rating: {score:.2f}")

# ── Save model ─────────────────────────────────────────────
with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to models/svd_model.pkl")

print("\n" + "=" * 50)
print("COLLABORATIVE FILTERING COMPLETE!")
print(f"RMSE: {rmse:.4f} — MAE: {mae:.4f}")
print("=" * 50)