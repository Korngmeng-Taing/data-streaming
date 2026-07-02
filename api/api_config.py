import os

from dotenv import load_dotenv

load_dotenv()


class APIConfig:
    base_url: str = os.getenv("CRYPTO_API_BASE_URL", "https://api.coingecko.com/api/v3")
    coin_ids: list[str] = os.getenv(
        "CRYPTO_IDS", "bitcoin,ethereum,solana,cardano,polkadot"
    ).split(",")
    vs_currency: str = os.getenv("VS_CURRENCY", "usd")
