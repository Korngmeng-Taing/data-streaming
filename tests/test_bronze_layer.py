import os
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType, LongType, StringType, StructField, StructType, TimestampType,
)
from pyspark.sql import Row

from spark.bronze_layer import write_to_bronze

TEST_SCHEMA = StructType([
    StructField("coin_id", StringType(), True),
    StructField("price_usd", DoubleType(), True),
    StructField("market_cap_usd", DoubleType(), True),
    StructField("volume_24h_usd", DoubleType(), True),
    StructField("change_24h_pct", DoubleType(), True),
    StructField("last_updated", LongType(), True),
    StructField("fetched_at", TimestampType(), True),
])


class TestBronzeLayer:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.checkpoint = os.path.join(self.tmp, "checkpoints", "bronze")
        self.output = os.path.join(self.tmp, "dwh")
        self.spark = SparkSession.builder \
            .appName("test_bronze") \
            .master("local[1]") \
            .config("spark.sql.streaming.schemaInference", "true") \
            .config("spark.sql.shuffle.partitions", "1") \
            .getOrCreate()

    def teardown_method(self):
        self.spark.stop()

    def test_write_to_bronze_starts_query(self):
        import time
        data = [Row(coin_id="bitcoin", price_usd=50000.0, market_cap_usd=1e12,
                    volume_24h_usd=3e10, change_24h_pct=2.5, last_updated=1700000000,
                    fetched_at="2025-01-01T00:00:00")]
        df = self.spark.createDataFrame(data, TEST_SCHEMA)
        json_strs = df.toJSON().collect()
        rows = [Row(key=None, value=s) for s in json_strs]
        kafka_df = self.spark.createDataFrame(rows)

        query = write_to_bronze(kafka_df, self.checkpoint, self.output)
        assert query is not None
        assert query.isActive
        time.sleep(2)
        query.stop()
        assert os.path.exists(os.path.join(self.output, "bronze"))
