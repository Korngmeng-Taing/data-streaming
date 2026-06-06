from pyspark.sql.types import (
    DoubleType, LongType, StringType, StructField, StructType, TimestampType,
)

bronze_schema = StructType(
    [
        StructField("coin_id", StringType(), False),
        StructField("price_usd", DoubleType(), True),
        StructField("market_cap_usd", DoubleType(), True),
        StructField("volume_24h_usd", DoubleType(), True),
        StructField("change_24h_pct", DoubleType(), True),
        StructField("last_updated", LongType(), True),
        StructField("event_time", TimestampType(), True),
        StructField("fetched_at", TimestampType(), True),
    ]
)
