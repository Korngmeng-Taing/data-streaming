import pandas as pd

from ml.features import build_features


def make_sample_gold(n: int = 12) -> pd.DataFrame:
    rows = []
    for coin in ["bitcoin", "ethereum"]:
        for i in range(n):
            h = 8 + i
            rows.append({
                "coin_id": coin,
                "window_start": pd.Timestamp(f"2025-01-01 {h:02d}:00:00"),
                "window_end": pd.Timestamp(f"2025-01-01 {h:02d}:01:00"),
                "avg_price": float(40000 + i * 100 + (0 if coin == "bitcoin" else -50)),
                "avg_volume": float(1e9 + i * 1e7),
                "avg_change_pct": float(0.5 + i * 0.1),
                "price_volatility": float(100.0 + i * 5),
                "record_count": 10 + i,
                "min_price": float(39500),
                "max_price": float(40500),
            })
    return pd.DataFrame(rows)


class TestBuildFeatures:
    def test_returns_feature_columns(self):
        df_in = make_sample_gold()
        result, cols = build_features(df_in)
        assert set(cols).issubset(result.columns)
        assert "avg_price" in result.columns
        assert "coin_id" in result.columns
        assert "window_start" in result.columns

    def test_creates_lag_features(self):
        df_in = make_sample_gold()
        result, cols = build_features(df_in)
        for lag in [1, 2, 3]:
            assert f"avg_price_lag_{lag}" in result.columns
            assert f"avg_volume_lag_{lag}" in result.columns

    def test_creates_rolling_features(self):
        df_in = make_sample_gold()
        result, cols = build_features(df_in)
        for w in [3, 6]:
            assert f"price_ma_{w}" in result.columns
            assert f"price_std_{w}" in result.columns

    def test_creates_price_change(self):
        df_in = make_sample_gold()
        result, cols = build_features(df_in)
        assert "price_change" in result.columns

    def test_drops_nan_rows_from_lags(self):
        df_in = make_sample_gold()
        result, cols = build_features(df_in)
        first_coin = df_in["coin_id"].unique()[0]
        n_coin = len(df_in[df_in["coin_id"] == first_coin])
        n_result_coin = len(result[result["coin_id"] == first_coin])
        assert n_result_coin < n_coin

    def test_sorts_by_coin_and_time(self):
        df_in = make_sample_gold()
        result, cols = build_features(df_in)
        for coin in result["coin_id"].unique():
            subset = result[result["coin_id"] == coin]["window_start"]
            assert subset.is_monotonic_increasing

    def test_empty_input(self):
        df_in = pd.DataFrame(columns=["coin_id", "window_start", "avg_price", "avg_volume",
                                       "avg_change_pct", "price_volatility", "record_count",
                                       "min_price", "max_price"])
        result, cols = build_features(df_in)
        assert result.empty

    def test_single_coin(self):
        df_in = make_sample_gold()
        df_in = df_in[df_in["coin_id"] == "bitcoin"].copy()
        result, cols = build_features(df_in)
        assert not result.empty
        assert (result["coin_id"] == "bitcoin").all()
