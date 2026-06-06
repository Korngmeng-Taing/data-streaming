import os
import tempfile

from dash_app.alert_store import load_alerts, save_alerts, load_history, save_history


class TestAlertStore:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.alerts_path = os.path.join(self.tmp, "alerts.json")
        self.history_path = os.path.join(self.tmp, "history.json")
        import dash_app.alert_store as store
        store.ALERTS_PATH = self.alerts_path
        store.HISTORY_PATH = self.history_path

    def test_save_and_load_alerts(self):
        alerts = [{"id": 1, "coin": "bitcoin", "type": "price", "active": True}]
        save_alerts(alerts)
        loaded = load_alerts()
        assert loaded == alerts

    def test_load_alerts_empty_when_file_missing(self):
        loaded = load_alerts()
        assert loaded == []

    def test_load_alerts_empty_when_invalid_json(self):
        with open(self.alerts_path, "w") as f:
            f.write("not json")
        loaded = load_alerts()
        assert loaded == []

    def test_save_and_load_history(self):
        history = [{"time": "12:00", "coin": "BTC", "message": "test"}]
        save_history(history)
        loaded = load_history()
        assert loaded == history

    def test_load_history_empty_when_file_missing(self):
        loaded = load_history()
        assert loaded == []

    def test_creates_directory_on_save(self):
        nested = os.path.join(self.tmp, "nested", "dir")
        os.environ["ALERTS_PATH"] = os.path.join(nested, "alerts.json")
        os.environ["HISTORY_PATH"] = os.path.join(nested, "history.json")
        import dash_app.alert_store as store
        store.ALERTS_PATH = os.path.join(nested, "alerts.json")
        store.HISTORY_PATH = os.path.join(nested, "history.json")
        save_alerts([{"id": 1}])
        assert os.path.isfile(os.path.join(nested, "alerts.json"))
