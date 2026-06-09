import pandas as pd

from dash_app.charts import build_main_price_chart, build_technical_chart


class TestCandlestickCharts:
    def test_build_main_price_chart_candlestick(self):
        # Create dummy data where price goes up and down
        data = {
            "coin_id": ["btc", "btc", "btc"],
            "time": pd.to_datetime(
                ["2026-06-07 00:00:00", "2026-06-07 00:01:00", "2026-06-07 00:02:00"]
            ),
            "price": [100.0, 105.0, 95.0],
            "max_price": [102.0, 107.0, 98.0],
            "min_price": [98.0, 103.0, 92.0],
        }
        df = pd.DataFrame(data)

        # Build chart
        fig = build_main_price_chart(
            df, ["btc"], "time", "price", chart_type="candlestick"
        )

        # Verify fig has trace
        assert len(fig.data) == 1
        trace = fig.data[0]

        # Verify it's a candlestick chart
        assert trace.type == "candlestick"

        # Verify shift logic for open/close
        # Expected open: [100.0, 100.0, 105.0]
        # Expected close: [100.0, 105.0, 95.0]
        assert list(trace.open) == [100.0, 100.0, 105.0]
        assert list(trace.close) == [100.0, 105.0, 95.0]

        # Verify colors are configured correctly
        assert trace.increasing.line.color == "#4CAF50"
        assert trace.decreasing.line.color == "#F44336"
        assert trace.increasing.fillcolor == "#4CAF50"
        assert trace.decreasing.fillcolor == "#F44336"

    def test_build_technical_chart_candlestick(self):
        # Create dummy data for technical chart
        # Technical analysis needs at least 20 points for bollinger bands, otherwise it might skip/fill
        # Let's provide a series of 25 points
        data = {
            "coin_id": ["btc"] * 25,
            "time": pd.date_range(start="2026-06-07 00:00:00", periods=25, freq="1min"),
            "price": [100.0 + i for i in range(25)],
            "max_price": [101.0 + i for i in range(25)],
            "min_price": [99.0 + i for i in range(25)],
            "avg_volume": [1000.0] * 25,
            "avg_change_pct": [1.0] * 25,
        }
        df = pd.DataFrame(data)

        fig = build_technical_chart(
            df, "btc", "time", "price", "avg_volume", "avg_change_pct"
        )

        # Find the candlestick trace (it should be the first trace)
        candlestick_traces = [t for t in fig.data if t.type == "candlestick"]
        assert len(candlestick_traces) == 1
        trace = candlestick_traces[0]

        # Verify shift logic for open/close
        # price went from 100 to 124
        # open should be shifted price: first is 100, second is 100, third is 101, etc.
        assert trace.open[0] == 100.0
        assert trace.open[1] == 100.0
        assert trace.open[2] == 101.0

        assert trace.close[0] == 100.0
        assert trace.close[1] == 101.0
        assert trace.close[2] == 102.0

        # Verify colors are configured correctly
        assert trace.increasing.line.color == "#4CAF50"
        assert trace.decreasing.line.color == "#F44336"
        assert trace.increasing.fillcolor == "#4CAF50"
        assert trace.decreasing.fillcolor == "#F44336"


class TestAutoInterpretation:
    def test_get_auto_interpretation_bullish(self):
        from dash_app.pages import _get_auto_interpretation

        # Create bullish series (price keeps going up)
        price_series = pd.Series([100.0 + i * 2 for i in range(30)])
        cdf = pd.DataFrame(
            {
                "coin_id": ["btc"] * 30,
                "price": price_series,
                "max_price": price_series + 1,
                "min_price": price_series - 1,
            }
        )
        last_price = price_series.iloc[-1]

        card = _get_auto_interpretation(price_series, cdf, last_price)
        assert card is not None
        # Render check: should contain card body and card header
        assert hasattr(card, "children")

    def test_get_auto_interpretation_bearish(self):
        from dash_app.pages import _get_auto_interpretation

        # Create bearish series (price keeps going down)
        price_series = pd.Series([200.0 - i * 2 for i in range(30)])
        cdf = pd.DataFrame(
            {
                "coin_id": ["btc"] * 30,
                "price": price_series,
                "max_price": price_series + 1,
                "min_price": price_series - 1,
            }
        )
        last_price = price_series.iloc[-1]

        card = _get_auto_interpretation(price_series, cdf, last_price)
        assert card is not None
        assert hasattr(card, "children")
