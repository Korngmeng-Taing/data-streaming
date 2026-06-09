import json
import time
from datetime import datetime
from config.timezone import PHNOM_PENH_TZ
import random
import os

import requests
from kafka import KafkaProducer

from api.api_config import APIConfig, KafkaConfig
from config.logging_config import setup_logger

logger = setup_logger("crypto_producer")


def generate_mock_prices() -> list[dict]:
    """Generate mock crypto price data for testing without API calls"""
    mock_data = {
        "bitcoin": {
            "price_range": (35000, 45000),
            "volume_range": (20e9, 30e9),
        },
        "ethereum": {
            "price_range": (1800, 2500),
            "volume_range": (10e9, 20e9),
        },
        "solana": {
            "price_range": (60, 150),
            "volume_range": (1e9, 3e9),
        },
        "cardano": {
            "price_range": (0.4, 1.0),
            "volume_range": (0.5e9, 2e9),
        },
        "polkadot": {
            "price_range": (5, 15),
            "volume_range": (0.5e9, 2e9),
        },
    }

    records = []
    ts = datetime.now(PHNOM_PENH_TZ).isoformat()
    
    for coin_id in APIConfig.coin_ids:
        if coin_id in mock_data:
            config = mock_data[coin_id]
            price = random.uniform(*config["price_range"])
            volume = random.uniform(*config["volume_range"])
            change = random.uniform(-5, 5)
            
            records.append({
                "coin_id": coin_id,
                "price_usd": round(price, 2),
                "market_cap_usd": round(price * random.uniform(1e6, 1e8), 0),
                "volume_24h_usd": round(volume, 0),
                "change_24h_pct": round(change, 2),
                "last_updated": ts,
                "fetched_at": ts,
            })
    
    logger.info(f"Generated {len(records)} mock coin prices")
    return records


def fetch_prices() -> list[dict]:
    # Check if mock mode is enabled
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    
    if use_mock:
        logger.info("Using mock data mode")
        return generate_mock_prices()
    
    url = f"{APIConfig.base_url}/simple/price"
    params = {
        "ids": ",".join(APIConfig.coin_ids),
        "vs_currencies": APIConfig.vs_currency,
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "include_market_cap": "true",
        "include_last_updated_at": "true",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            logger.warning("Rate limited (429). Falling back to mock data...")
            return generate_mock_prices()
        else:
            logger.error(f"API fetch failed (HTTP {resp.status_code}): {e}")
        return []
    except requests.RequestException as e:
        logger.warning(f"API fetch failed: {e}. Falling back to mock data...")
        return generate_mock_prices()

    records = []
    ts = datetime.now(PHNOM_PENH_TZ).isoformat()
    for coin_id, values in data.items():
        records.append(
            {
                "coin_id": coin_id,
                "price_usd": values.get(f"{APIConfig.vs_currency}"),
                "market_cap_usd": values.get(f"{APIConfig.vs_currency}_market_cap"),
                "volume_24h_usd": values.get(f"{APIConfig.vs_currency}_24h_vol"),
                "change_24h_pct": values.get(f"{APIConfig.vs_currency}_24h_change"),
                "last_updated": values.get("last_updated_at"),
                "fetched_at": ts,
            }
        )

    logger.info(f"Fetched {len(records)} coin prices")
    return records


def produce():
    producer = KafkaProducer(
        bootstrap_servers=KafkaConfig.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )

    logger.info(
        f"Starting producer -> {KafkaConfig.bootstrap_servers} "
        f"topic={KafkaConfig.topic}"
    )

    consecutive_failures = 0
    base_sleep = 60
    max_sleep = 600

    while True:
        records = fetch_prices()
        if not records:
            consecutive_failures += 1
            sleep_time = min(base_sleep * (2 ** (consecutive_failures - 1)), max_sleep)
            jitter_sleep = sleep_time + random.uniform(0, sleep_time * 0.1)
            logger.warning(
                f"No records (failure #{consecutive_failures}), "
                f"backing off {jitter_sleep:.1f}s..."
            )
            time.sleep(jitter_sleep)
            continue

        consecutive_failures = 0
        for rec in records:
            try:
                future = producer.send(
                    KafkaConfig.topic,
                    key=rec["coin_id"].encode("utf-8"),
                    value=rec,
                )
                future.get(timeout=10)
                logger.debug(f"Sent record for {rec['coin_id']}")
            except Exception as e:
                logger.error(f"Failed to send record: {e}")

        producer.flush()
        logger.info(
            f"Produced {len(records)} messages "
            f"({', '.join([r['coin_id'] for r in records])}), "
            f"sleeping {base_sleep}s..."
        )
        time.sleep(base_sleep)


if __name__ == "__main__":
    produce()
