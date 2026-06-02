import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    avg, coalesce, col, count, lit, max as spark_max, min as spark_min,
    stddev, window, to_timestamp,
)

from config.logging_config import setup_logger

logger = setup_logger("gold_layer")


def agg_and_write(df: DataFrame, output_path: str, checkpoint_dir: str):
    gold_path = f"{output_path}/gold"
    os.makedirs(gold_path, exist_ok=True)
    os.makedirs(f"{checkpoint_dir}/gold", exist_ok=True)

    gold = (
        df.withColumn("timestamp", to_timestamp(col("event_time")))
        .withWatermark("timestamp", "1 minute")
        .groupBy(
            col("coin_id"),
            window(col("timestamp"), "1 minute"),
        )
        .agg(
            avg("price_usd").alias("avg_price"),
            spark_min("price_usd").alias("min_price"),
            spark_max("price_usd").alias("max_price"),
            avg("volume_24h_usd").alias("avg_volume"),
            avg("change_24h_pct").alias("avg_change_pct"),
            coalesce(stddev("price_usd"), lit(0.0)).alias("price_volatility"),
            count("*").alias("record_count"),
        )
        .select(
            col("coin_id"),
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "avg_price", "min_price", "max_price",
            "avg_volume", "avg_change_pct",
            "price_volatility", "record_count",
        )
    )

    query = (
        gold.writeStream
        .format("parquet")
        .option("path", gold_path)
        .option("checkpointLocation", f"{checkpoint_dir}/gold")
        .partitionBy("coin_id")
        .trigger(processingTime="15 seconds")
        .outputMode("append")
        .start()
    )

    logger.info(f"Gold streaming -> {gold_path}")
    return query


def read_gold(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(f"{path}/gold")
