from confluent_kafka import Consumer
from sqlalchemy import create_engine, text
import json
import pandas as pd

DB_URL = "postgresql://recsys_user:recsys123@localhost:5433/recsys"
engine = create_engine(DB_URL)

# Create events table if not exists
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS clickstream_events (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            product_id TEXT,
            event_type TEXT,
            session_id TEXT,
            timestamp FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """))
    conn.commit()
print("Table ready!")

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'recsys-consumer-group',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['user-events'])

print("Consumer listening on topic: user-events")
print("Writing events to PostgreSQL...")
print("Press Ctrl+C to stop\n")

count = 0
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Error: {msg.error()}")
            continue

        event = json.loads(msg.value().decode('utf-8'))

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO clickstream_events
                (user_id, product_id, event_type, session_id, timestamp)
                VALUES (:user_id, :product_id, :event_type,
                        :session_id, :timestamp)
            """), event)
            conn.commit()

        count += 1
        print(f"[{count}] Saved → {event['event_type'].upper()} | "
              f"user: {event['user_id'][:10]}... | "
              f"product: {event['product_id']}")

except KeyboardInterrupt:
    print(f"\nStopped. Total events saved: {count}")
finally:
    consumer.close()