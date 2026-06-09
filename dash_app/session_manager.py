import os
import atexit
import signal
import json
import glob
from datetime import datetime

import pandas as pd

from config.logging_config import setup_logger
from config.timezone import PHNOM_PENH_TZ

logger = setup_logger("session_manager")

SESSIONS_DIR = os.path.join(os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh"), "sessions")
_session_data: list[dict] = []
_session_start = datetime.now(PHNOM_PENH_TZ)


def _ensure_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def record_data(data_json: str):
    if not data_json:
        return
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else {}
        gold = data.get("gold", [])
        silver = data.get("silver", [])
        now = datetime.now(PHNOM_PENH_TZ).isoformat()
        for rec in gold:
            rec["_layer"] = "gold"
            rec["_recorded_at"] = now
            _session_data.append(rec)
        for rec in silver:
            rec["_layer"] = "silver"
            rec["_recorded_at"] = now
            _session_data.append(rec)
    except Exception as e:
        logger.warning(f"Failed to record session data: {e}")


def _deduplicate(records: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for r in records:
        key = (r.get("coin_id", ""), r.get("_layer", ""),
               str(r.get("window_start", "")), str(r.get("fetched_at", "")))
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result


def save_session():
    if not _session_data:
        logger.info("No session data to save")
        return
    _ensure_dir()
    deduped = _deduplicate(_session_data)
    if not deduped:
        return
    df = pd.DataFrame(deduped)
    ts = _session_start.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SESSIONS_DIR, f"session_{ts}.csv")
    df.to_csv(path, index=False)
    logger.info(f"Session saved: {path} ({len(df)} rows)")


def list_sessions() -> list[dict]:
    _ensure_dir()
    files = sorted(glob.glob(os.path.join(SESSIONS_DIR, "session_*.csv")), reverse=True)
    sessions = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=0)
            size = os.path.getsize(f)
            total = sum(1 for _ in open(f, encoding="utf-8")) - 1
            fname = os.path.basename(f)
            ts_str = fname.replace("session_", "").replace(".csv", "")
            sessions.append({
                "path": f,
                "filename": fname,
                "timestamp": ts_str,
                "columns": list(df.columns),
                "rows": max(0, total),
                "size_kb": round(size / 1024, 1),
            })
        except Exception as e:
            logger.warning(f"Failed to read session file {f}: {e}")
    return sessions


def load_session_csv(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"Failed to load session CSV {filepath}: {e}")
        return pd.DataFrame()


def _cleanup():
    save_session()


atexit.register(_cleanup)

for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, lambda s, f: save_session())
    except (ValueError, OSError):
        pass
