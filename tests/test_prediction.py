import pandas as pd
import numpy as np

from prediction.arima import predict_prices as predict_arima
from prediction.prophet_model import predict_prices_prophet


def _make_gold_df(n=30):
    dates = pd.date_range("2025-01-01", periods=n, freq="1min")
    prices = np.linspace(100, 110, n) + np.random.randn(n) * 2
    return pd.DataFrame({
        "coin_id": ["bitcoin"] * n,
        "window_start": dates,
        "avg_price": prices,
    })


class TestARIMA:
    def test_returns_forecast_dataframe(self):
        gold = _make_gold_df(30)
        result = predict_arima(gold)
        assert "bitcoin" in result
        df = result["bitcoin"]
        assert "predicted_price" in df.columns
        assert len(df) == 12

    def test_returns_fallback_for_short_series(self):
        gold = _make_gold_df(5)
        result = predict_arima(gold)
        assert "bitcoin" in result
        assert len(result["bitcoin"]) == 12

    def test_returns_fallback_for_single_point(self):
        gold = _make_gold_df(1)
        result = predict_arima(gold)
        assert "bitcoin" in result
        assert len(result["bitcoin"]) == 12
        assert (result["bitcoin"]["predicted_price"] == gold["avg_price"].iloc[0]).all()

    def test_returns_empty_for_empty_df(self):
        result = predict_arima(pd.DataFrame())
        assert result == {}


class TestProphet:
    def test_returns_forecast_dataframe(self):
        gold = _make_gold_df(30)
        result = predict_prices_prophet(gold)
        if "bitcoin" in result:
            df = result["bitcoin"]
            assert "predicted_price" in df.columns
            assert len(df) == 12

    def test_returns_fallback_for_short_series(self):
        gold = _make_gold_df(5)
        result = predict_prices_prophet(gold)
        assert "bitcoin" in result
        assert len(result["bitcoin"]) == 12

    def test_returns_empty_for_empty_df(self):
        result = predict_prices_prophet(pd.DataFrame())
        assert result == {}
