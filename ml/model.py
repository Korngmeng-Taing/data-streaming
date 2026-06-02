import joblib
import pandas as pd

from config.logging_config import setup_logger

logger = setup_logger("model_loader")


def load_model(path: str = "/tmp/crypto-model/model.joblib"):
    artifact = joblib.load(path)
    return artifact["model"], artifact["feature_cols"], artifact.get("metrics", {})


def predict(model, features: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    return model.predict(features[feature_cols])
