import os
import sys

from config.logging_config import setup_logger

logger = setup_logger("config_validation")

REQUIRED = {
    "KAFKA_BOOTSTRAP_SERVERS": "localhost:9093",
    "KAFKA_TOPIC": "crypto-prices",
}

DEPENDENT = {
    "spark-streaming": {
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        "OUTPUT_PATH": "/tmp/crypto-dwh",
        "CHECKPOINT_DIR": "/tmp/spark-checkpoints",
    },
    "ml-train": {
        "OUTPUT_PATH": "/tmp/crypto-dwh",
        "ML_MODEL_PATH": "/tmp/crypto-model",
    },
    "dashboard": {
        "OUTPUT_PATH": "/tmp/crypto-dwh",
        "ML_MODEL_PATH": "/tmp/crypto-model",
    },
    "ws-gateway": {
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9093",
        "WS_PORT": "8765",
    },
}

VALID_COINS = {
    "bitcoin", "ethereum", "solana", "cardano", "polkadot",
    "dogecoin", "ripple", "litecoin", "chainlink", "avalanche-2",
}


def validate() -> bool:
    ok = True

    for var, default in REQUIRED.items():
        val = os.getenv(var, default)
        if not val:
            logger.error(f"Missing required env var: {var}")
            ok = False

    service = os.getenv("SERVICE_NAME", "")
    if service in DEPENDENT:
        expected = DEPENDENT[service]
        for var, default in expected.items():
            val = os.getenv(var)
            if not val:
                logger.warning(f"{service}: {var} is not set, will use default '{default}'")

    coins_str = os.getenv("CRYPTO_IDS", "")
    if coins_str:
        for coin in coins_str.split(","):
            coin = coin.strip()
            if coin and coin not in VALID_COINS:
                logger.warning(f"Unknown coin ID '{coin}' - may not be recognized by CoinGecko")

    kafka = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9093")
    parts = kafka.split(":")
    if len(parts) != 2 or not parts[1].isdigit():
        logger.error(f"Invalid KAFKA_BOOTSTRAP_SERVERS format: {kafka} (expected host:port)")
        ok = False

    return ok


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
