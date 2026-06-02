from pathlib import Path

from pyspark.sql import SparkSession

from config.logging_config import setup_logger
from spark.silver_layer import clean_and_write

logger = setup_logger("bronze_to_silver")

SILVER_ETL_CHECKPOINT = "/tmp/checkpoints/bronze_to_silver"


def run(spark: SparkSession, bronze_path: str, silver_path: str):
    logger.info(f"Reading bronze from {bronze_path}")
    bronze_df = spark.readStream.parquet(bronze_path)
    query = clean_and_write(bronze_df, silver_path, SILVER_ETL_CHECKPOINT)
    query.awaitTermination()
