import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, when

from config.logging_config import setup_logger

logger = setup_logger("silver_layer")


def clean_and_write(df: DataFrame, output_path: str, checkpoint_dir: str):
    silver_path = f"{output_path}/silver"
    os.makedirs(silver_path, exist_ok=True)
    os.makedirs(f"{checkpoint_dir}/silver", exist_ok=True)

    silver = (
        df
        .filter(col("coin_id").isNotNull())
        .filter(col("price_usd").isNotNull() & (col("price_usd") > 0))
        .withColumn("price_usd", col("price_usd").cast("decimal(18,6)"))
        .withColumn("market_cap_usd", col("market_cap_usd").cast("decimal(24,2)"))
        .withColumn("volume_24h_usd", col("volume_24h_usd").cast("decimal(24,2)"))
        .withColumn("change_24h_pct", col("change_24h_pct").cast("decimal(8,4)"))
        .withColumn(
            "data_quality_flag",
            when(col("change_24h_pct").between(-100, 1000), lit("good"))
            .otherwise(lit("suspicious")),
        )
        .dropDuplicates(["coin_id", "event_time"])
    )

    query = (
        silver.writeStream
        .format("parquet")
        .option("path", silver_path)
        .option("checkpointLocation", f"{checkpoint_dir}/silver")
        .partitionBy("coin_id", "data_quality_flag")
        .trigger(processingTime="10 seconds")
        .outputMode("append")
        .start()
    )

    logger.info(f"Silver streaming -> {silver_path}")
    return query


def read_silver(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(f"{path}/silver")
