from pyspark.sql import SparkSession

from config.logging_config import setup_logger
from spark.gold_layer import agg_and_write

logger = setup_logger("silver_to_gold")

GOLD_ETL_CHECKPOINT = "/tmp/checkpoints/silver_to_gold"


def run(spark: SparkSession, silver_path: str, gold_path: str):
    logger.info(f"Reading silver from {silver_path}")
    silver_df = spark.readStream.parquet(silver_path)
    query = agg_and_write(silver_df, gold_path, GOLD_ETL_CHECKPOINT)
    query.awaitTermination()
