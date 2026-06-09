import os
import sys

from config.logging_config import setup_logger

logger = setup_logger("config_validation")

REQUIRED = {
    "OUTPUT_PATH": "/tmp/crypto-dwh",
}

VALID_COINS = {
    "bitcoin",
    "ethereum",
    "solana",
    "cardano",
    "polkadot",
    "dogecoin",
    "ripple",
    "litecoin",
    "chainlink",
    "avalanche-2",
}


def validate() -> bool:
    ok = True

    for var, default in REQUIRED.items():
        val = os.getenv(var, default)
        if not val:
            logger.error(f"Missing required env var: {var}")
            ok = False

    coins_str = os.getenv("CRYPTO_IDS", "")
    if coins_str:
        for coin in coins_str.split(","):
            coin = coin.strip()
            if coin and coin not in VALID_COINS:
                logger.warning(
                    f"Unknown coin ID '{coin}' - may not be recognized by CoinGecko"
                )

    return ok


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
