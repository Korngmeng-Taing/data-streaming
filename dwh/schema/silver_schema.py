from pyspark.sql.types import (
    DecimalType, StringType, StructField, StructType, TimestampType,
)

silver_schema = StructType(
    [
        StructField("coin_id", StringType(), False),
        StructField("price_usd", DecimalType(18, 6), True),
        StructField("market_cap_usd", DecimalType(24, 2), True),
        StructField("volume_24h_usd", DecimalType(24, 2), True),
        StructField("change_24h_pct", DecimalType(8, 4), True),
        StructField("data_quality_flag", StringType(), True),
        StructField("event_time", TimestampType(), True),
        StructField("fetched_at", TimestampType(), True),
    ]
)
