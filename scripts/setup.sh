#!/bin/bash
set -euo pipefail

echo "=== Crypto Data Streaming — Setup ==="

# Create Python venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Create local dirs
mkdir -p /tmp/crypto-dwh/{bronze,silver,gold}
mkdir -p /tmp/spark-checkpoints/{bronze,silver,gold}
mkdir -p /tmp/crypto-model

# Copy .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Copied .env.example -> .env — edit as needed"
fi

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Ensure Kafka & Zookeeper are running (docker-compose up -d zookeeper kafka)"
echo "  2. Start the crypto producer: python api/crypto_producer.py"
echo "  3. Start Spark streaming:   python spark/streaming_job.py"
echo "  4. Train model:             python ml/train.py"
echo "  5. Launch dashboard:        streamlit run viz/dashboard.py"
echo "  6. Launch app:              streamlit run app/main.py"
