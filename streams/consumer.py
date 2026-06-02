import json
from collections.abc import Callable

from kafka import KafkaConsumer

from api.api_config import KafkaConfig
from config.logging_config import setup_logger

logger = setup_logger("kafka_consumer")


def create_consumer(
    topic: str | None = None,
    group_id: str = "crypto-streaming-group",
    auto_offset_reset: str = "latest",
) -> KafkaConsumer:
    return KafkaConsumer(
        topic or KafkaConfig.topic,
        bootstrap_servers=KafkaConfig.bootstrap_servers,
        group_id=group_id,
        auto_offset_reset=auto_offset_reset,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
    )


def consume(handler: Callable[[str, dict], None], topic: str | None = None):
    consumer = create_consumer(topic)
    logger.info(f"Listening on topic: {consumer.topics()}")

    try:
        for msg in consumer:
            handler(msg.key, msg.value)
    except KeyboardInterrupt:
        logger.info("Shutting down consumer")
    finally:
        consumer.close()
