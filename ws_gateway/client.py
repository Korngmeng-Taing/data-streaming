import json
import os
import time
from urllib.request import urlopen, URLError

WS_GATEWAY_HOST = os.getenv("WS_GATEWAY_HOST", "localhost")
HEALTH_URL = f"http://{WS_GATEWAY_HOST}:8766"
_cache: dict[str, tuple[float, float | None]] = {}


def get_last_update() -> float | None:
    cached = _cache.get("last_update")
    if cached and time.time() - cached[0] < 2.0:
        return cached[1]
    try:
        resp = urlopen(f"{HEALTH_URL}", timeout=1.0)
        data = json.loads(resp.read().decode())
        ts = data.get("last_update")
        _cache["last_update"] = (time.time(), ts)
        return ts
    except (URLError, OSError, json.JSONDecodeError):
        return _cache.get("last_update", (0, None))[1]
