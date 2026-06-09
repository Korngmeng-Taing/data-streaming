import json
import os
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config.logging_config import setup_logger

logger = setup_logger("dash_app")

# Track when the app started for "this session" filtering (Unix epoch, tz-agnostic)
APP_START_TIME = time.time()

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh")
GOLD_PATH = f"{OUTPUT_PATH}/gold"
SILVER_PATH = f"{OUTPUT_PATH}/silver"
SESSIONS_PATH = f"{OUTPUT_PATH}/sessions"

_parquet_cache: dict[str, tuple[pd.DataFrame, float]] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 9.0


def load_parquet(path: str) -> pd.DataFrame:
    now = time.time()
    with _cache_lock:
        cached = _parquet_cache.get(path)
        if cached is not None and (now - cached[1]) < CACHE_TTL:
            return cached[0]
    try:
        df = pd.read_parquet(path)
        with _cache_lock:
            _parquet_cache[path] = (df, now)
        return df
    except Exception as e:
        logger.warning(f"Cannot read {path}: {e}")
        return pd.DataFrame()


def prepare_df(
    df: pd.DataFrame, time_col: str, value_col: str, extra_cols: list[str]
) -> pd.DataFrame:
    for c in df.select_dtypes(include=["object"]):
        if c == time_col:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    num_cols = [value_col] + [c for c in extra_cols if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[value_col, time_col]).sort_values(time_col)
    return df


TIMEFRAMES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}


def resample_df(df: pd.DataFrame, interval: str, time_col: str) -> pd.DataFrame:
    if interval == "1m" or df.empty or time_col not in df.columns:
        return df
    rule = TIMEFRAMES.get(interval)
    if not rule:
        return df
    df = df.copy()
    if time_col in df.columns and df[time_col].dtype == object:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    price_cols = [
        "avg_price",
        "min_price",
        "max_price",
        "avg_volume",
        "avg_change_pct",
        "price_volatility",
    ]
    for c in price_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    agg = {}
    for col in num_cols:
        if col in ("coin_id", time_col):
            continue
        if col == "min_price":
            agg[col] = "min"
        elif col == "max_price":
            agg[col] = "max"
        elif col in ("avg_volume", "record_count"):
            agg[col] = "sum"
        else:
            agg[col] = "mean"
    df = df.set_index(time_col)
    resampled = df.groupby("coin_id").resample(rule).agg(agg)
    resampled = resampled.dropna(
        subset=[c for c in resampled.columns if c.startswith("avg_")][:1]
    )
    resampled = resampled.reset_index()
    return resampled


MOCK_COINS = [
    {"id": "bitcoin", "base_price": 65000, "volatility": 800},
    {"id": "ethereum", "base_price": 3200, "volatility": 120},
    {"id": "solana", "base_price": 140, "volatility": 8},
    {"id": "cardano", "base_price": 0.6, "volatility": 0.05},
    {"id": "polkadot", "base_price": 7.5, "volatility": 0.4},
]


def list_session_files() -> list[str]:
    if not os.path.isdir(SESSIONS_PATH):
        return []
    try:
        return sorted(
            (f for f in os.listdir(SESSIONS_PATH) if f.endswith(".csv")),
            reverse=True,
        )
    except Exception:
        return []


def load_session_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(SESSIONS_PATH, filename)
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Cannot load session CSV {filename}: {e}")
        return pd.DataFrame()


def _generate_mock_data():
    now = time.time()
    span = max(1.0, now - APP_START_TIME)
    n_points = max(10, min(60, int(span / 60) + 1))
    if n_points <= 1:
        timestamps = [pd.Timestamp.fromtimestamp(now, tz="UTC")]
    else:
        timestamps = [
            pd.Timestamp.fromtimestamp(now - span + span * i / (n_points - 1), tz="UTC")
            for i in range(n_points)
        ]
    gold_rows = []
    silver_rows = []
    for coin in MOCK_COINS:
        base = coin["base_price"]
        vol = coin["volatility"]
        for i, ts in enumerate(timestamps):
            price = (
                base
                + np.sin(i / 10 * np.pi) * vol * 0.3
                + np.random.randn() * vol * 0.05
            )
            change = np.random.randn() * 2
            volume = base * np.random.uniform(50000, 200000)
            gold_rows.append(
                {
                    "coin_id": coin["id"],
                    "window_start": ts,
                    "avg_price": round(price, 6),
                    "min_price": round(price - vol * 0.02, 6),
                    "max_price": round(price + vol * 0.02, 6),
                    "avg_volume": round(volume, 2),
                    "avg_change_pct": round(change, 4),
                    "price_volatility": round(vol * 0.01, 6),
                    "record_count": np.random.randint(5, 30),
                }
            )
            silver_rows.append(
                {
                    "coin_id": coin["id"],
                    "fetched_at": ts,
                    "price_usd": round(price, 6),
                    "volume_24h_usd": round(volume * 24, 2),
                    "change_24h_pct": round(change, 4),
                    "market_cap_usd": round(price * np.random.uniform(1e6, 1e8), 2),
                    "data_quality_flag": "good",
                    "last_updated": int(time.time()),
                }
            )
    return pd.DataFrame(gold_rows), pd.DataFrame(silver_rows)


def _load_all_data(interval: str) -> str:
    gold = load_parquet(GOLD_PATH)
    silver = load_parquet(SILVER_PATH)
    use_mock = gold.empty and silver.empty
    result = {}

    if use_mock:
        gold, silver = _generate_mock_data()

    if not gold.empty:
        df = prepare_df(
            gold,
            "window_start",
            "avg_price",
            [
                "avg_volume",
                "avg_change_pct",
                "min_price",
                "max_price",
                "price_volatility",
                "record_count",
            ],
        )
        result["gold"] = df.to_dict("records") if not df.empty else []
        result["gold_time_col"] = "window_start"
        result["gold_value_col"] = "avg_price"
        result["gold_vol_col"] = "avg_volume"
        result["gold_chg_col"] = "avg_change_pct"
        result["gold_extra"] = [
            "min_price",
            "max_price",
            "price_volatility",
            "record_count",
        ]

    if not silver.empty:
        df = prepare_df(
            silver,
            "fetched_at",
            "price_usd",
            ["volume_24h_usd", "change_24h_pct", "market_cap_usd"],
        )
        result["silver"] = df.to_dict("records") if not df.empty else []
        result["silver_time_col"] = "fetched_at"
        result["silver_value_col"] = "price_usd"
        result["silver_vol_col"] = "volume_24h_usd"
        result["silver_chg_col"] = "change_24h_pct"
        result["silver_extra"] = ["market_cap_usd"]

    coins = set()
    for key in ["gold", "silver"]:
        records = result.get(key, [])
        if isinstance(records, list):
            for r in records:
                if isinstance(r, dict) and "coin_id" in r:
                    coins.add(r["coin_id"])
    result["coins"] = sorted(coins)
    result["updated_at"] = datetime.now().isoformat()

    return json.dumps(result, default=str)


def _parse_store_json(data_json: str) -> dict:
    return json.loads(data_json) if isinstance(data_json, str) else {}


def save_session_csv():
    try:
        gold = load_parquet(GOLD_PATH)
        silver = load_parquet(SILVER_PATH)
        df = gold if not gold.empty else silver
        if df.empty:
            logger.info("No data to save for session CSV")
            return
        os.makedirs(SESSIONS_PATH, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SESSIONS_PATH, f"session_{timestamp}.csv")
        df.to_csv(path, index=False)
        logger.info(f"Session data saved to {path} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Failed to save session CSV: {e}")


def df_from_store(data_json: str) -> tuple:
    data = _parse_store_json(data_json)
    gold = pd.DataFrame(data.get("gold", []))
    silver = pd.DataFrame(data.get("silver", []))
    coins = data.get("coins", [])
    return gold, silver, coins, data
