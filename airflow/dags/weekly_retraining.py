from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import pickle
from sqlalchemy import create_engine

default_args = {
    'owner': 'dibya',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def retrain_model():
    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split

    engine = create_engine(
        "postgresql://recsys_user:recsys123@recsys_postgres:5432/recsys"
    )

    print("Loading training data...")
    df = pd.read_sql("SELECT user_id, product_id, rating FROM clean_reviews", engine)
    print(f"Training on {len(df):,} reviews")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df, reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(n_factors=100, n_epochs=40, lr_all=0.008, reg_all=0.02)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"Retrained RMSE: {rmse:.4f}")

    with open("/opt/airflow/models/svd_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved! RMSE: {rmse:.4f}")
    return rmse

with DAG(
    'weekly_model_retraining',
    default_args=default_args,
    description='Weekly model retraining pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'retraining', 'weekly']
) as dag:

    retrain_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model
    )