from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

default_args = {
    'owner': 'dibya',
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

def refresh_user_features():
    engine = create_engine(
        "postgresql://recsys_user:recsys123@recsys_postgres:5432/recsys"
    )
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_features AS
            SELECT
                user_id,
                COUNT(*) as total_events,
                COUNT(DISTINCT product_id) as unique_products,
                COUNT(CASE WHEN event_type='purchase' THEN 1 END) as purchases,
                COUNT(CASE WHEN event_type='view' THEN 1 END) as views,
                MAX(created_at) as last_active
            FROM clickstream_events
            GROUP BY user_id
        """))
        conn.commit()
    print("User features refreshed!")

with DAG(
    'hourly_feature_refresh',
    default_args=default_args,
    description='Hourly user feature refresh',
    schedule_interval='@hourly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['features', 'hourly']
) as dag:

    refresh_task = PythonOperator(
        task_id='refresh_user_features',
        python_callable=refresh_user_features
    )