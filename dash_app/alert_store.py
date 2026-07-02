import json
import os

ALERTS_PATH = os.getenv("ALERTS_PATH", "/data/dwh/alerts.json")
HISTORY_PATH = os.getenv("HISTORY_PATH", "/data/dwh/alert_history.json")


def _load_json(path: str) -> list:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_alerts():
    return _load_json(ALERTS_PATH)


def save_alerts(alerts):
    _save_json(alerts, ALERTS_PATH)


def load_history():
    return _load_json(HISTORY_PATH)


def save_history(history):
    _save_json(history, HISTORY_PATH)
