import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.logging_config import setup_logger

logger = setup_logger("pipeline")

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh")
RAW_PATH = f"{OUTPUT_PATH}/raw"
BRONZE_PATH = f"{OUTPUT_PATH}/bronze"
SILVER_PATH = f"{OUTPUT_PATH}/silver"
GOLD_PATH = f"{OUTPUT_PATH}/gold"

REQUIRED_COLUMNS = {
    "coin_id": "string",
    "price_usd": "float64",
    "market_cap_usd": "float64",
    "volume_24h_usd": "float64",
    "change_24h_pct": "float64",
    "last_updated": "float64",
    "fetched_at": "string",
}


def _ensure_dirs():
    for p in (RAW_PATH, BRONZE_PATH, SILVER_PATH, GOLD_PATH):
        os.makedirs(p, exist_ok=True)


def read_raw() -> pd.DataFrame:
    _ensure_dirs()
    files = sorted(Path(RAW_PATH).glob("*.jsonl"))
    if not files:
        return pd.DataFrame()
    rows = []
    for f in files:
        try:
            chunk = pd.read_json(f, lines=True)
            if not chunk.empty:
                rows.append(chunk)
        except Exception as e:
            logger.warning(f"Skipping corrupt raw file {f.name}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            df[col] = None
        if dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "fetched_at" in df.columns:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    if "last_updated" in df.columns:
        try:
            df["last_updated"] = pd.to_datetime(
                df["last_updated"], unit="s", errors="coerce"
            )
        except (ValueError, TypeError):
            df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    return df


def write_bronze(df: pd.DataFrame):
    if df.empty:
        return
    _ensure_dirs()
    df = _normalize_types(df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(BRONZE_PATH, f"bronze_{ts}.parquet")
    df.to_parquet(path, index=False)
    logger.info(f"Bronze: wrote {len(df)} rows to {path}")


def process_silver(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    silver = df.copy()
    silver = silver.dropna(subset=["coin_id", "price_usd"])
    silver = silver[silver["price_usd"] > 0]
    if "change_24h_pct" in silver.columns:
        silver["data_quality_flag"] = np.where(
            silver["change_24h_pct"].between(-100, 1000), "good", "suspicious"
        )
    else:
        silver["data_quality_flag"] = "good"
    subset = [c for c in ("coin_id", "fetched_at") if c in silver.columns]
    if subset:
        silver = silver.drop_duplicates(subset=subset)
    return silver


def write_silver(df: pd.DataFrame):
    if df.empty:
        return
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SILVER_PATH, f"silver_{ts}.parquet")
    df.to_parquet(path, index=False)
    logger.info(f"Silver: wrote {len(df)} rows to {path}")


def process_gold(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "fetched_at" not in df.columns:
        return df
    gold = df.copy()
    gold["_ts"] = pd.to_datetime(gold["fetched_at"])
    gold["window_start"] = gold["_ts"].dt.floor("10s")
    gold["window_end"] = gold["window_start"] + pd.Timedelta(seconds=10)

    numeric_cols = ["price_usd", "volume_24h_usd", "change_24h_pct"]
    for c in numeric_cols:
        if c in gold.columns:
            gold[c] = pd.to_numeric(gold[c], errors="coerce")

    grouped = gold.groupby(["coin_id", "window_start", "window_end"], as_index=False)
    result = grouped.agg(
        avg_price=("price_usd", "mean"),
        min_price=("price_usd", "min"),
        max_price=("price_usd", "max"),
        avg_volume=("volume_24h_usd", "mean"),
        avg_change_pct=("change_24h_pct", "mean"),
        price_volatility=("price_usd", "std"),
        record_count=("price_usd", "count"),
    )
    result["price_volatility"] = result["price_volatility"].fillna(0.0)
    return result


def write_gold(df: pd.DataFrame):
    if df.empty:
        return
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(GOLD_PATH, f"gold_{ts}.parquet")
    df.to_parquet(path, index=False)
    logger.info(f"Gold: wrote {len(df)} rows to {path}")


def cleanup_raw(keep_hours: float = 1):
    cutoff = time.time() - keep_hours * 3600
    for f in Path(RAW_PATH).glob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except Exception:
            pass


def process():
    _ensure_dirs()
    raw = read_raw()
    if raw.empty:
        return False

    write_bronze(raw)
    silver = process_silver(raw)
    write_silver(silver)
    gold = process_gold(silver)
    write_gold(gold)
    cleanup_raw()
    return True
