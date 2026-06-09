import signal
import time

from config.logging_config import setup_logger
from pipeline.processor import process

logger = setup_logger("pipeline_runner")

running = True


def _signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received, stopping pipeline...")
    running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def main():
    logger.info("Pipeline runner started (interval=10s)")
    while running:
        try:
            processed = process()
            if processed:
                logger.debug("Pipeline cycle completed")
        except Exception as e:
            logger.error(f"Pipeline cycle failed: {e}")
        for _ in range(10):
            if not running:
                break
            time.sleep(1)
    logger.info("Pipeline runner stopped")


if __name__ == "__main__":
    main()
