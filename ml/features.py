import pandas as pd


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if df.empty or "coin_id" not in df.columns:
        return df.copy(), []

    for coin in df["coin_id"].unique():
        mask = df["coin_id"] == coin

        # Lag features (past values only, no target leakage)
        for lag in [1, 2, 3]:
            df.loc[mask, f"avg_price_lag_{lag}"] = (
                df.loc[mask, "avg_price"].shift(lag)
            )
            df.loc[mask, f"avg_volume_lag_{lag}"] = (
                df.loc[mask, "avg_volume"].shift(lag)
            )

        # Rolling statistics of past values
        for window in [3, 6]:
            df.loc[mask, f"price_ma_{window}"] = (
                df.loc[mask, "avg_price"].rolling(window).mean()
            )
            df.loc[mask, f"price_std_{window}"] = (
                df.loc[mask, "avg_price"].rolling(window).std()
            )

        # Price change from previous window
        df.loc[mask, "price_change"] = df.loc[mask, "avg_price"].pct_change()

    feature_cols = [
        "avg_volume", "avg_change_pct",
        "price_volatility", "record_count",
        "avg_price_lag_1", "avg_price_lag_2", "avg_price_lag_3",
        "avg_volume_lag_1", "avg_volume_lag_2", "avg_volume_lag_3",
        "price_ma_3", "price_ma_6", "price_std_3", "price_std_6",
        "price_change",
    ]

    df = df.fillna({"price_volatility": 0.0, "avg_change_pct": 0.0})
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df[["coin_id", "window_start", "avg_price"] + feature_cols], feature_cols
