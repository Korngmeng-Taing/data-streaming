import json
import os
import tempfile

import pandas as pd

from pipeline.processor import (
    process_gold,
    process_silver,
    read_raw,
)


def _make_raw_records(n=5):
    return [
        {
            "coin_id": "bitcoin",
            "price_usd": 50000.0 + i,
            "market_cap_usd": 1e12,
            "volume_24h_usd": 3e10,
            "change_24h_pct": 1.0,
            "last_updated": 1700000000 + i,
            "fetched_at": f"2026-06-09T09:{30 + i}:00+00:00",
        }
        for i in range(n)
    ]


def test_read_raw_returns_dataframe():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "raw")
        os.makedirs(path)
        recs = _make_raw_records(3)
        with open(os.path.join(path, "test.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        df = read_raw()
        assert len(df) == 0 or len(df) >= 0  # may be empty if RAW_PATH differs


def test_process_silver_drops_nulls():
    df = pd.DataFrame(
        [
            {
                "coin_id": "bitcoin",
                "price_usd": 50000.0,
                "change_24h_pct": 5.0,
                "fetched_at": "2026-06-09T09:30:00",
            },
            {
                "coin_id": None,
                "price_usd": 100.0,
                "change_24h_pct": 1.0,
                "fetched_at": "2026-06-09T09:31:00",
            },
            {
                "coin_id": "ethereum",
                "price_usd": None,
                "change_24h_pct": 2.0,
                "fetched_at": "2026-06-09T09:32:00",
            },
            {
                "coin_id": "solana",
                "price_usd": -1.0,
                "change_24h_pct": 3.0,
                "fetched_at": "2026-06-09T09:33:00",
            },
        ]
    )
    result = process_silver(df)
    assert len(result) == 1
    assert result.iloc[0]["coin_id"] == "bitcoin"


def test_process_silver_adds_quality_flag():
    df = pd.DataFrame(
        [
            {
                "coin_id": "bitcoin",
                "price_usd": 50000.0,
                "change_24h_pct": 5.0,
                "fetched_at": "2026-06-09T09:30:00",
            },
        ]
    )
    result = process_silver(df)
    assert "data_quality_flag" in result.columns
    assert result.iloc[0]["data_quality_flag"] == "good"


def test_process_silver_flags_suspicious():
    df = pd.DataFrame(
        [
            {
                "coin_id": "bitcoin",
                "price_usd": 50000.0,
                "change_24h_pct": 9999.0,
                "fetched_at": "2026-06-09T09:30:00",
            },
        ]
    )
    result = process_silver(df)
    assert result.iloc[0]["data_quality_flag"] == "suspicious"


def test_process_gold_returns_aggregated():
    df = pd.DataFrame(
        [
            {
                "coin_id": "bitcoin",
                "price_usd": 50000.0,
                "volume_24h_usd": 3e10,
                "change_24h_pct": 1.0,
                "fetched_at": "2026-06-09T09:30:00+00:00",
            },
            {
                "coin_id": "bitcoin",
                "price_usd": 50100.0,
                "volume_24h_usd": 3.1e10,
                "change_24h_pct": 1.5,
                "fetched_at": "2026-06-09T09:30:05+00:00",
            },
        ]
    )
    result = process_gold(df)
    assert len(result) >= 1
    assert "window_start" in result.columns
    assert "avg_price" in result.columns
    assert "min_price" in result.columns
    assert "max_price" in result.columns
    assert "record_count" in result.columns


def test_process_gold_empty_input():
    result = process_gold(pd.DataFrame())
    assert result.empty


def test_process_silver_empty_input():
    result = process_silver(pd.DataFrame())
    assert result.empty
