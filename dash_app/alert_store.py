import json
import os

ALERTS_PATH = os.getenv("ALERTS_PATH", "/data/dwh/alerts.json")
HISTORY_PATH = os.getenv("HISTORY_PATH", "/data/dwh/alert_history.json")


def load_alerts() -> list:
    try:
        with open(ALERTS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_alerts(alerts: list):
    os.makedirs(os.path.dirname(ALERTS_PATH), exist_ok=True)
    with open(ALERTS_PATH, "w") as f:
        json.dump(alerts, f, indent=2)


def load_history() -> list:
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(history: list):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
