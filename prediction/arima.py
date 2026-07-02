import numpy as np
import pandas as pd

logger = __import__("logging").getLogger("prediction.arima")

FORECAST_STEPS = 12
ARIMA_ORDER = (5, 1, 0)


def _fallback_forecast(series: pd.Series, steps: int) -> np.ndarray:
    """Uses linear regression if points >= 2, else constant projection."""
    n = len(series)
    if n >= 2:
        x = np.arange(n)
        y = series.values
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(n, n + steps)
        return slope * future_x + intercept
    if n == 1:
        return np.full(steps, series.iloc[0])
    return np.zeros(steps)


def predict_prices(
    gold: pd.DataFrame,
    steps: int = FORECAST_STEPS,
) -> dict[str, pd.DataFrame]:
    if gold.empty or "coin_id" not in gold.columns or "avg_price" not in gold.columns:
        return {}

    result = {}
    for coin in gold["coin_id"].unique():
        coin_df = gold[gold["coin_id"] == coin].copy()
        coin_df["window_start"] = pd.to_datetime(coin_df["window_start"])
        coin_df = coin_df.sort_values("window_start")
        series = coin_df["avg_price"].dropna()

        if series.empty:
            continue

        last_ts = coin_df["window_start"].iloc[-1]
        if isinstance(last_ts, str):
            last_ts = pd.Timestamp(last_ts)

        # Calculate frequency safely
        if len(coin_df) >= 2:
            freq = coin_df["window_start"].diff().median()
        else:
            freq = pd.Timedelta(minutes=1)

        if freq is None or pd.isna(freq):
            freq = pd.Timedelta(minutes=1)

        pred_timestamps = [last_ts + (i + 1) * freq for i in range(steps)]

        if len(series) < 10:
            logger.info(f"Using fallback forecast for {coin} ({len(series)} points)")
            forecast = _fallback_forecast(series, steps)
        else:
            try:
                aligned = coin_df.loc[series.index].dropna(
                    subset=["avg_price", "avg_volume"]
                )
                if aligned.empty or len(aligned) < 5:
                    forecast = _fallback_forecast(series, steps)
                else:
                    exog = aligned["avg_volume"].values
                    last_exog = exog[-1]
                    future_exog = np.full((steps, 1), last_exog)

                    from statsmodels.tsa.statespace.sarimax import SARIMAX

                    model = SARIMAX(
                        aligned["avg_price"].values, exog=exog, order=ARIMA_ORDER
                    )
                    fitted = model.fit(disp=False)
                    forecast = fitted.forecast(steps=steps, exog=future_exog)
            except Exception as e:
                logger.warning(f"SARIMAX failed for {coin}, falling back: {e}")
                forecast = _fallback_forecast(series, steps)

        result[coin] = pd.DataFrame(
            {
                "coin_id": coin,
                "window_start": pred_timestamps,
                "predicted_price": forecast,
            }
        )

    return result
