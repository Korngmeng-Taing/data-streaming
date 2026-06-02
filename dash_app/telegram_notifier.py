import logging
import os
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("telegram")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


def send_telegram(message: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    payload = f"chat_id={CHAT_ID}&text={message}&parse_mode=HTML"
    req = Request(API_URL, data=payload.encode(), headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        urlopen(req, timeout=5)
        return True
    except URLError as e:
        logger.warning(f"Telegram send failed: {e}")
        return False
