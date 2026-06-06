import pandas as pd

from dash_app.backtest import backtest


def make_preds(direction: str = "UP") -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows.append({
            "coin_id": "bitcoin",
            "window_start": pd.Timestamp(f"2025-01-01 {10 + i}:00:00"),
            "direction": direction,
            "actual_price": 40000 + i * 100,
        })
    return pd.DataFrame(rows)


class TestBacktest:
    def test_returns_expected_keys(self):
        preds = make_preds()
        result = backtest(preds, initial_capital=1000.0)
        assert "results" in result
        assert len(result["results"]) == 1

    def test_stats_have_required_fields(self):
        preds = make_preds()
        result = backtest(preds)
        stats = result["results"][0]["stats"]
        for key in ["total_return_pct", "num_trades", "win_rate_pct",
                     "max_drawdown_pct", "sharpe_ratio"]:
            assert key in stats, f"Missing key: {key}"

    def test_trades_list(self):
        preds = make_preds()
        result = backtest(preds)
        trades = result["results"][0]["trades"]
        assert isinstance(trades, list)

    def test_has_curve(self):
        preds = make_preds()
        result = backtest(preds)
        curve = result["results"][0]["curve"]
        for col in ["window_start", "cum_strategy", "buy_hold"]:
            assert col in curve.columns

    def test_empty_preds_returns_error(self):
        result = backtest(pd.DataFrame())
        assert "error" in result

    def test_missing_direction_returns_error(self):
        preds = pd.DataFrame({"coin_id": ["bitcoin"], "window_start": ["2025-01-01"], "actual_price": [40000]})
        result = backtest(preds)
        assert "error" in result

    def test_multiple_coins(self):
        rows = []
        for coin in ["bitcoin", "ethereum"]:
            for i in range(10):
                rows.append({
                    "coin_id": coin,
                    "window_start": pd.Timestamp(f"2025-01-01 {10 + i}:00:00"),
                    "direction": "UP",
                    "actual_price": 40000 + i * 100,
                })
        preds = pd.DataFrame(rows)
        result = backtest(preds)
        assert len(result["results"]) == 2

    def test_down_direction_trades(self):
        preds = make_preds(direction="DOWN")
        result = backtest(preds)
        trades = result["results"][0]["trades"]
        for t in trades:
            assert t["direction"] == "SHORT"
