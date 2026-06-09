import numpy as np
import pandas as pd

from viz.utils import bollinger, rsi, sma


class TestSMA:
    def test_basic_average(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = sma(s, 3)
        assert result.iloc[2] == 2.0
        assert result.iloc[4] == 4.0

    def test_short_series(self):
        s = pd.Series([1, 2])
        result = sma(s, 5)
        assert result.isna().all()


class TestBollinger:
    def test_returns_middle_upper_lower(self):
        s = pd.Series(np.random.randn(50) * 10 + 100)
        middle, upper, lower = bollinger(s, 20, 2)
        assert len(middle) == 50
        valid = middle.notna()
        assert (upper[valid] >= middle[valid]).all()
        assert (lower[valid] <= middle[valid]).all()

    def test_constant_series(self):
        s = pd.Series([100.0] * 30)
        middle, upper, lower = bollinger(s, 20, 2)
        assert (middle.dropna() == 100.0).all()
        assert (upper.dropna() == 100.0).all()
        assert (lower.dropna() == 100.0).all()


class TestRSI:
    def test_rsi_between_0_and_100(self):
        s = pd.Series(np.random.randn(100) * 5 + 100)
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_50_for_flat_series(self):
        s = pd.Series([100.0] * 30)
        result = rsi(s, 14)
        valid = result.dropna()
        assert ((valid - 50).abs() < 1e-6).all()

    def test_rsi_100_for_consistently_up(self):
        s = pd.Series(list(range(100, 200)))
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid > 90).all()
