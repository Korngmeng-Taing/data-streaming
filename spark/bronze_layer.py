import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, from_json, schema_of_json, to_timestamp, lit, from_unixtime

from config.logging_config import setup_logger

logger = setup_logger("bronze_layer")

RAW_SCHEMA_JSON = """
{
    "coin_id": "bitcoin",
    "price_usd": 50000.0,
    "market_cap_usd": 1000000000000.0,
    "volume_24h_usd": 30000000000.0,
    "change_24h_pct": 2.5,
    "last_updated": 1700000000,
    "fetched_at": "2025-01-01T00:00:00+00:00"
}
"""


def write_to_bronze(df: DataFrame, checkpoint_dir: str, output_path: str):
    inferred_schema = schema_of_json(RAW_SCHEMA_JSON)
    bronze_path = f"{output_path}/bronze"
    os.makedirs(bronze_path, exist_ok=True)
    os.makedirs(f"{checkpoint_dir}/bronze", exist_ok=True)

    stream = (
        df.selectExpr(
            "CAST(key AS STRING) as coin_id_raw",
            "CAST(value AS STRING) as json_str",
        )
        .withColumn("parsed", from_json(col("json_str"), inferred_schema))
        .select("parsed.*")
        .withColumn("fetched_at", to_timestamp(col("fetched_at")))
        .withColumn("event_time", from_unixtime(col("last_updated")).cast("timestamp"))
    )

    query = (
        stream.writeStream
        .format("parquet")
        .option("path", bronze_path)
        .option("checkpointLocation", f"{checkpoint_dir}/bronze")
        .partitionBy("coin_id")
        .trigger(processingTime="10 seconds")
        .start()
    )

    logger.info(f"Bronze streaming -> {bronze_path}")
    return query


def read_bronze(spark: SparkSession, path: str) -> DataFrame:
    bronze_path = f"{path}/bronze"
    return spark.read.parquet(bronze_path)
