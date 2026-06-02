#!/bin/bash
set -euo pipefail

echo "Starting Kafka stack..."
docker-compose up -d zookeeper kafka
echo "Waiting for Kafka to be ready..."

for i in $(seq 1 30); do
    if docker exec crypto-kafka kafka-topics --bootstrap-server localhost:9092 --list &>/dev/null; then
        echo "Kafka is ready!"
        break
    fi
    sleep 2
done

# Create topic
docker exec crypto-kafka kafka-topics \
    --bootstrap-server localhost:9092 \
    --create \
    --topic crypto-prices \
    --partitions 3 \
    --replication-factor 1 \
    --if-not-exists

echo "Topic 'crypto-prices' ready"
