import os
import signal
import sys
import time

from dotenv import load_dotenv
from pyspark.sql import SparkSession

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

queries = []


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

    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    bronze_query = write_to_bronze(df, CHECKPOINT_DIR, OUTPUT_PATH)
    queries.append(bronze_query)

    logger.info("Starting bronze stream...")
    time.sleep(5)

    logger.info("Starting silver stream (reading from bronze)...")
    bronze_stream = (
        spark.readStream
        .schema(bronze_schema)
        .parquet(f"{OUTPUT_PATH}/bronze")
    )
    silver_query = clean_and_write(bronze_stream, OUTPUT_PATH, CHECKPOINT_DIR)
    queries.append(silver_query)

    time.sleep(5)

    logger.info("Starting gold stream (reading from silver)...")
    silver_stream = (
        spark.readStream
        .schema(silver_schema)
        .parquet(f"{OUTPUT_PATH}/silver")
    )
    gold_query = agg_and_write(silver_stream, OUTPUT_PATH, CHECKPOINT_DIR)
    queries.append(gold_query)

    logger.info("All streaming queries started. Waiting for termination...")
    bronze_query.awaitTermination()


if __name__ == "__main__":
    main()
