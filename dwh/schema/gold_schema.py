from pyspark.sql.types import (
    DecimalType, IntegerType, StringType, StructField, StructType, TimestampType,
)

gold_schema = StructType(
    [
        StructField("coin_id", StringType(), False),
        StructField("window_start", TimestampType(), False),
        StructField("window_end", TimestampType(), False),
        StructField("avg_price", DecimalType(18, 6), True),
        StructField("min_price", DecimalType(18, 6), True),
        StructField("max_price", DecimalType(18, 6), True),
        StructField("avg_volume", DecimalType(24, 2), True),
        StructField("avg_change_pct", DecimalType(10, 4), True),
        StructField("price_volatility", DecimalType(18, 6), True),
        StructField("record_count", IntegerType(), True),
    ]
)
