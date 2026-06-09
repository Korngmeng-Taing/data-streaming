import json
import os
import random
import time
from datetime import datetime, timezone

import requests

from api.api_config import APIConfig
from config.logging_config import setup_logger

logger = setup_logger("crypto_producer")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh")
RAW_PATH = f"{OUTPUT_PATH}/raw"


def generate_mock_prices() -> list[dict]:
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
    now = datetime.now(timezone.utc)
    ts_iso = now.isoformat()
    ts_unix = int(now.timestamp())

    for coin_id in APIConfig.coin_ids:
        if coin_id in mock_data:
            config = mock_data[coin_id]
            price = random.uniform(*config["price_range"])
            volume = random.uniform(*config["volume_range"])
            change = random.uniform(-5, 5)

            records.append(
                {
                    "coin_id": coin_id,
                    "price_usd": round(price, 2),
                    "market_cap_usd": round(price * random.uniform(1e6, 1e8), 0),
                    "volume_24h_usd": round(volume, 0),
                    "change_24h_pct": round(change, 2),
                    "last_updated": ts_unix,
                    "fetched_at": ts_iso,
                }
            )

    logger.info(f"Generated {len(records)} mock coin prices")
    return records


def fetch_prices() -> list[dict]:
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
        status = resp.status_code if hasattr(resp, "status_code") else None
        if status == 429:
            logger.warning("Rate limited (429). Falling back to mock data...")
            return generate_mock_prices()
        logger.error(f"API fetch failed (HTTP {status}): {e}")
        return []
    except requests.RequestException as e:
        logger.warning(f"API fetch failed: {e}. Falling back to mock data...")
        return generate_mock_prices()

    records = []
    ts = datetime.now(timezone.utc).isoformat()
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


def write_raw(records: list[dict]):
    if not records:
        return None
    os.makedirs(RAW_PATH, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(RAW_PATH, f"prices_{ts}.jsonl")
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def produce():
    logger.info(f"Starting producer -> {RAW_PATH}")

    consecutive_failures = 0
    base_sleep = 10
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
        path = write_raw(records)

        coins = ", ".join([r["coin_id"] for r in records])
        logger.info(
            f"Wrote {len(records)} records to {path} ({coins}), sleeping {base_sleep}s..."
        )
        time.sleep(base_sleep)


if __name__ == "__main__":
    produce()
