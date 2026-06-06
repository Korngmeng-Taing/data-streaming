.PHONY: help install test lint run-producer run-dash \
        run-streaming run-ws docker-up docker-down clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

lint: ## Check syntax of all Python files
	python -m py_compile config/validate.py
	python -m py_compile config/spark_manager.py
	python -m py_compile config/spark_config.py
	python -m py_compile config/logging_config.py
	python -m py_compile api/crypto_producer.py
	python -m py_compile api/api_config.py
	python -m py_compile spark/streaming_job.py
	python -m py_compile spark/bronze_layer.py
	python -m py_compile spark/silver_layer.py
	python -m py_compile spark/gold_layer.py
	python -m py_compile dash_app/app.py
	python -m py_compile dash_app/alert_store.py
	python -m py_compile viz/utils.py
	python -m py_compile ws_gateway/server.py
	python -m py_compile ws_gateway/client.py

run-producer: ## Run the Kafka producer
	python -m api.crypto_producer

run-dash: ## Run the Dash dashboard
	python -m dash_app.app

run-streaming: ## Run the Spark streaming job
	python -m spark.streaming_job

run-ws: ## Run the WebSocket gateway
	python -m ws_gateway.server

validate: ## Validate configuration
	python -m config.validate

docker-up: ## Build and start all services
	docker-compose up --build -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## Tail logs from all services
	docker-compose logs -f

clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
