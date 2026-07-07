from confluent_kafka import Producer
import json
import random
import time
import pandas as pd

# Load real user and product IDs from your dataset
df = pd.read_csv("data/processed/clean_reviews.csv")
users = df['user_id'].unique().tolist()
products = df['product_id'].unique().tolist()
events = ["view", "add_to_cart", "purchase", "wishlist"]

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def delivery_report(err, msg):
    if err:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Event sent → topic: {msg.topic()} | '
              f'partition: {msg.partition()}')

print("Starting clickstream producer...")
print("Sending real user events to Kafka topic: user-events")
print("Press Ctrl+C to stop\n")

count = 0
while True:
    event = {
        "user_id": random.choice(users),
        "product_id": random.choice(products),
        "event_type": random.choice(events),
        "timestamp": time.time(),
        "session_id": f"session_{random.randint(1000, 9999)}"
    }
    producer.produce(
        'user-events',
        key=event['user_id'],
        value=json.dumps(event),
        callback=delivery_report
    )
    producer.flush()
    count += 1
    print(f"[{count}] {event['event_type'].upper()} — "
          f"user: {event['user_id'][:10]}... "
          f"product: {event['product_id']}")
    time.sleep(1)