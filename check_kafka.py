import sys, json
sys.path.insert(0, 'D:/data-streaming/crypto-data-streaming')
from kafka import KafkaConsumer

c = KafkaConsumer(
    'crypto-prices',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    consumer_timeout_ms=5000,
    value_deserializer=lambda v: json.loads(v)
)
msgs = list(c)
c.close()
print(f"Total messages consumed: {len(msgs)}")
for m in msgs[:5]:
    print(f"{m.key.decode()}: ${m.value['price_usd']} at {m.value['fetched_at']}")
