import os

import pandas as pd

from config.logging_config import setup_logger
from ml.features import build_features
from ml.model import load_model

logger = setup_logger("predictor")


def generate_predictions(spark, gold_path: str, model_path: str | None = None):
    if model_path is None:
        model_dir = os.getenv("ML_MODEL_PATH", os.getenv("MODEL_PATH", "/tmp/crypto-model"))
        model_path = f"{model_dir}/model_1m.joblib"
    logger.info("Loading gold data for predictions...")
    pdf = spark.read.parquet(f"{gold_path}/gold").toPandas()

    features_df, _ = build_features(pdf)
    model, feature_cols, metrics = load_model(model_path)

    preds = model.predict(features_df[feature_cols])

    result = features_df[["coin_id", "window_start"]].copy()
    result["predicted_price"] = preds
    result["metrics"] = str(metrics)

    logger.info(f"Generated {len(result)} predictions")
    return result
