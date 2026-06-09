import asyncio
import json
import os
import time
from datetime import datetime
from config.timezone import PHNOM_PENH_TZ

import aiokafka
import websockets
from dotenv import load_dotenv

from config.logging_config import setup_logger

load_dotenv()

logger = setup_logger("ws_gateway")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9093")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "crypto-prices")
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))

CLIENTS: set[websockets.WebSocketServerProtocol] = set()
last_update: float = 0.0


async def kafka_worker():
    global last_update
    consumer = aiokafka.AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
    )
    await consumer.start()
    logger.info(f"Kafka consumer started: {KAFKA_BOOTSTRAP}/{KAFKA_TOPIC}")

    try:
        async for msg in consumer:
            record = msg.value
            record["_timestamp"] = time.time()
            payload = json.dumps(record)
            last_update = time.time()
            if CLIENTS:
                await asyncio.gather(
                    *[c.send(payload) for c in CLIENTS],
                    return_exceptions=True,
                )
    except asyncio.CancelledError:
        pass
    finally:
        await consumer.stop()


async def ws_handler(websocket: websockets.WebSocketServerProtocol):
    CLIENTS.add(websocket)
    logger.info(f"Client connected ({len(CLIENTS)} total)")
    try:
        async for _ in websocket:
            pass
    except websockets.ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(websocket)
        logger.info(f"Client disconnected ({len(CLIENTS)} total)")


async def health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    status = json.dumps({
        "ok": True,
        "clients": len(CLIENTS),
        "last_update": last_update,
        "last_update_iso": (
            datetime.fromtimestamp(last_update, tz=PHNOM_PENH_TZ).isoformat()
            if last_update else None
        ),
        "topic": KAFKA_TOPIC,
    })
    response = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(status)}\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n"
        f"{status}"
    )
    writer.write(response.encode())
    await writer.drain()
    writer.close()


async def main():
    logger.info(f"Starting WS gateway on {WS_HOST}:{WS_PORT}")
    kafka_task = asyncio.create_task(kafka_worker())

    async with (
        websockets.serve(ws_handler, WS_HOST, WS_PORT, ping_interval=20),
        await asyncio.start_server(health_handler, WS_HOST, 8766),
    ):
        logger.info(f"WS: ws://{WS_HOST}:{WS_PORT}, Health: http://{WS_HOST}:8766")
        await asyncio.gather(kafka_task)


if __name__ == "__main__":
    asyncio.run(main())
