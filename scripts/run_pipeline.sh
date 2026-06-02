#!/bin/bash
set -euo pipefail

echo "=== Running full Crypto Data Pipeline ==="

# 1. Start Kafka
echo "[1/5] Starting Kafka..."
docker-compose up -d zookeeper kafka
sleep 10

# 2. Start crypto producer
echo "[2/5] Starting crypto producer..."
python api/crypto_producer.py &
PRODUCER_PID=$!
sleep 5

# 3. Start Spark streaming (bronze → silver → gold)
echo "[3/5] Starting Spark streaming..."
python spark/streaming_job.py &
SPARK_PID=$!
sleep 30

# 4. Train ML model
echo "[4/5] Training ML model..."
python ml/train.py

# 5. Launch dashboard & app
echo "[5/5] Launching dashboard and app..."
streamlit run viz/dashboard.py --server.port=8501 &
streamlit run app/main.py --server.port=8502 &

echo "=== Pipeline running ==="
echo "Kafka:      localhost:9092"
echo "Dashboard:  http://localhost:8501"
echo "App:         http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop."

trap "kill $PRODUCER_PID $SPARK_PID 2>/dev/null; docker-compose down" EXIT
wait
