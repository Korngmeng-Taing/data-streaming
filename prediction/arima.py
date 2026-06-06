from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = __import__("logging").getLogger("prediction.arima")

FORECAST_STEPS = 12
ARIMA_ORDER = (5, 1, 0)


def predict_prices(
    gold: pd.DataFrame,
    steps: int = FORECAST_STEPS,
) -> dict[str, pd.DataFrame]:
    if gold.empty or "coin_id" not in gold.columns or "avg_price" not in gold.columns:
        return {}

    result = {}
    for coin in gold["coin_id"].unique():
        coin_df = gold[gold["coin_id"] == coin].copy()
        coin_df = coin_df.sort_values("window_start")
        series = coin_df["avg_price"].dropna()
        if len(series) < 10:
            logger.warning(f"Not enough data for {coin}: {len(series)} points")
            continue

        try:
            model = ARIMA(series.values, order=ARIMA_ORDER)
            fitted = model.fit()
            forecast = fitted.forecast(steps=steps)
            last_ts = coin_df["window_start"].iloc[-1]
            if isinstance(last_ts, str):
                last_ts = pd.Timestamp(last_ts)
            freq = coin_df["window_start"].diff().median()
            pred_timestamps = [
                last_ts + (i + 1) * freq for i in range(steps)
            ]
            result[coin] = pd.DataFrame({
                "coin_id": coin,
                "window_start": pred_timestamps,
                "predicted_price": forecast,
            })
        except Exception as e:
            logger.warning(f"ARIMA failed for {coin}: {e}")
            continue

    return result
