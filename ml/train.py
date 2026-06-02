import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config.logging_config import setup_logger
from ml.features import build_features

logger = setup_logger("ml_trainer")

MIN_FEATURE_ROWS = 100


_spark = None


def spark_session():
    global _spark
    if _spark is None:
        from pyspark.sql import SparkSession
        _spark = SparkSession.builder.appName("CryptoML").master("local[*]").getOrCreate()
    return _spark


def read_gold_data(spark, path: str) -> pd.DataFrame:
    return spark.read.parquet(path).toPandas()


def train_for_interval(gold_path: str, model_dir: str, interval: str):
    os.makedirs(model_dir, exist_ok=True)
    interval_model_path = os.path.join(model_dir, f"model_{interval}.joblib")
    logger.info(f"Training model for interval={interval} -> {interval_model_path}")

    pdf = read_gold_data(spark_session(), gold_path)

    TIMEFRAMES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}
    rule = TIMEFRAMES.get(interval, "1min")
    if rule != "1min":
        num_cols = pdf.select_dtypes(include=[np.number]).columns
        price_cols = ["avg_price", "min_price", "max_price", "avg_volume", "avg_change_pct", "price_volatility"]
        for c in price_cols:
            if c in pdf.columns and pdf[c].dtype == object:
                pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
        agg = {}
        for c in num_cols:
            if c in ("coin_id", "window_start"): continue
            if c == "min_price": agg[c] = "min"
            elif c == "max_price": agg[c] = "max"
            elif c in ("avg_volume", "record_count"): agg[c] = "sum"
            else: agg[c] = "mean"
        pdf = pdf.set_index("window_start")
        pdf = pdf.groupby("coin_id").resample(rule).agg(agg).dropna().reset_index()

    logger.info(f"Loaded {len(pdf)} rows for {interval}")
    features_df, feature_cols = build_features(pdf)
    logger.info(f"Feature rows for {interval}: {len(features_df)}")

    if len(features_df) < MIN_FEATURE_ROWS:
        logger.warning(f"Only {len(features_df)} feature rows for {interval} — need {MIN_FEATURE_ROWS}, skipping")
        return

    X = features_df[feature_cols].copy()
    y = features_df["avg_price"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    ridge_r2 = r2_score(y_test, ridge.predict(X_test))

    rf = RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))

    if ridge_r2 >= rf_r2:
        model, y_pred, r2, model_name = ridge, ridge.predict(X_test), ridge_r2, "Ridge"
    else:
        model, y_pred, r2, model_name = rf, rf.predict(X_test), rf_r2, "RandomForest"

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    logger.info(f"{interval} best: {model_name}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "model_name": model_name}},
        interval_model_path,
    )
    logger.info(f"Model for {interval} saved to {interval_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-path", default=os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh") + "/gold")
    parser.add_argument("--model-dir", default=os.getenv("MODEL_PATH", "/tmp/crypto-model"))
    args = parser.parse_args()

    spark = spark_session()
    for interval in ["1m", "5m", "15m", "30m", "1h"]:
        try:
            train_for_interval(args.gold_path, args.model_dir, interval)
        except Exception as e:
            logger.warning(f"Training for {interval} failed: {e}")
