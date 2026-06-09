import os
import signal
import sys
import time
import threading

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQuery

from config.logging_config import setup_logger
from config.spark_config import get_spark_config
from dwh.schema.bronze_schema import bronze_schema
from dwh.schema.silver_schema import silver_schema
from spark.bronze_layer import write_to_bronze
from spark.gold_layer import agg_and_write
from spark.silver_layer import clean_and_write

load_dotenv()

logger = setup_logger("streaming_job")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "crypto-prices")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/tmp/crypto-dwh")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/tmp/spark-checkpoints")
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

queries: list[StreamingQuery] = []
_start_barrier = threading.Barrier(3, timeout=120)


def signal_handler(sig, frame):
    logger.info("Shutdown signal received, stopping streams...")
    for q in queries:
        try:
            q.stop()
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def wait_for_kafka(max_retries=30, delay=5):
    logger.info(f"Waiting for Kafka at {KAFKA_BOOTSTRAP}...")
    import socket
    host, port = KAFKA_BOOTSTRAP.split(":")
    port = int(port)
    for attempt in range(max_retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((host, port))
            s.close()
            logger.info("Kafka is reachable")
            return True
        except Exception:
            if attempt < max_retries - 1:
                logger.info(f"Waiting for Kafka (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
    logger.error("Kafka not reachable after max retries")
    return False


def _wait_for_path(path: str, max_wait: int = 30):
    for _ in range(max_wait):
        if os.path.exists(path) and any(os.scandir(path)):
            return True
        time.sleep(1)
    return False


def start_bronze(spark: SparkSession) -> StreamingQuery:
    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )
    q = write_to_bronze(df, CHECKPOINT_DIR, OUTPUT_PATH)
    queries.append(q)
    logger.info("Bronze stream started")
    return q


def start_silver(spark: SparkSession) -> StreamingQuery:
    bronze_path = f"{OUTPUT_PATH}/bronze"
    if not _wait_for_path(bronze_path):
        logger.warning("Bronze data not found, silver may start with no initial data")
    bronze_stream = (
        spark.readStream
        .schema(bronze_schema)
        .parquet(bronze_path)
    )
    q = clean_and_write(bronze_stream, OUTPUT_PATH, CHECKPOINT_DIR)
    queries.append(q)
    logger.info("Silver stream started")
    return q


def start_gold(spark: SparkSession) -> StreamingQuery:
    silver_path = f"{OUTPUT_PATH}/silver"
    if not _wait_for_path(silver_path):
        logger.warning("Silver data not found, gold may start with no initial data")
    silver_stream = (
        spark.readStream
        .schema(silver_schema)
        .parquet(silver_path)
    )
    q = agg_and_write(silver_stream, OUTPUT_PATH, CHECKPOINT_DIR)
    queries.append(q)
    logger.info("Gold stream started")
    return q


def main():
    if not wait_for_kafka():
        sys.exit(1)

    spark = (
        SparkSession.builder
        .appName("CryptoBronzeSilverGold")
        .master(SPARK_MASTER)
    )

    for k, v in get_spark_config().items():
        if k not in ("spark.master", "spark.app.name"):
            spark = spark.config(k, v)

    spark = spark.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Connecting to Kafka: {KAFKA_BOOTSTRAP}, topic: {TOPIC}")

    bronze_q = start_bronze(spark)
    start_silver(spark)
    start_gold(spark)

    logger.info("All streaming queries started. Waiting for termination...")
    bronze_q.awaitTermination()


if __name__ == "__main__":
    main()
