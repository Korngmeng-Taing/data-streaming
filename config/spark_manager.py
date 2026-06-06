from pyspark.sql import SparkSession

from config.logging_config import setup_logger
from config.spark_config import get_spark_config

logger = setup_logger("spark_manager")

_spark = None


def get_spark(app_name: str = "CryptoApp", master: str | None = None) -> SparkSession:
    global _spark
    if _spark is not None:
        return _spark

    master = master or "local[*]"
    builder = SparkSession.builder.appName(app_name).master(master)

    for k, v in get_spark_config().items():
        if k not in ("spark.master", "spark.app.name"):
            builder = builder.config(k, v)

    _spark = builder.getOrCreate()
    _spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Spark session created: app={app_name}, master={master}")
    return _spark


def stop_spark():
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
        logger.info("Spark session stopped")
