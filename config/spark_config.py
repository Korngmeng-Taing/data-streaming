import os


def get_spark_config():
    return {
        "spark.master": os.getenv("SPARK_MASTER", "local[*]"),
        "spark.app.name": "CryptoDataStreaming",
        "spark.sql.shuffle.partitions": "4",
        "spark.streaming.backpressure.enabled": "true",
        "spark.streaming.kafka.maxRatePerPartition": "100",
        "spark.jars.packages": "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        "spark.sql.streaming.schemaInference": "true",
        "spark.hadoop.fs.defaultFS": "file:///",
    }
