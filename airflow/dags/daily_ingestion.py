from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text

default_args = {
    'owner': 'dibya',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def ingest_clickstream_events():
    engine = create_engine(
        "postgresql://recsys_user:recsys123@recsys_postgres:5432/recsys"
    )
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM clickstream_events
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """))
        count = result.fetchone()[0]
        print(f"New events in last 24 hours: {count}")

        conn.execute(text("""
            INSERT INTO processed_events
            SELECT
                user_id,
                product_id,
                event_type,
                COUNT(*) as event_count,
                MAX(created_at) as last_seen
            FROM clickstream_events
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY user_id, product_id, event_type
            ON CONFLICT (user_id, product_id, event_type)
            DO UPDATE SET
                event_count = processed_events.event_count + EXCLUDED.event_count,
                last_seen = EXCLUDED.last_seen
        """))
        conn.commit()
        print("Events processed and saved!")

def refresh_popular_products():
    engine = create_engine(
        "postgresql://recsys_user:recsys123@recsys_postgres:5432/recsys"
    )
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS popular_products AS
            SELECT
                product_id,
                COUNT(*) as interaction_count,
                NOW() as updated_at
            FROM clickstream_events
            GROUP BY product_id
            ORDER BY interaction_count DESC
            LIMIT 100
        """))
        conn.commit()
        print("Popular products refreshed!")

with DAG(
    'daily_data_ingestion',
    default_args=default_args,
    description='Daily clickstream ingestion pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ingestion', 'daily']
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_clickstream_events',
        python_callable=ingest_clickstream_events
    )

    refresh_task = PythonOperator(
        task_id='refresh_popular_products',
        python_callable=refresh_popular_products
    )

    ingest_task >> refresh_task