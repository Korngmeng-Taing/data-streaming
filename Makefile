.PHONY: help install test lint run-producer run-pipeline run-dash \
        clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

lint: ## Check syntax of all Python files
	python -m py_compile config/validate.py
	python -m py_compile config/logging_config.py
	python -m py_compile api/crypto_producer.py
	python -m py_compile api/api_config.py
	python -m py_compile pipeline/processor.py
	python -m py_compile pipeline/runner.py
	python -m py_compile dash_app/app.py
	python -m py_compile dash_app/alert_store.py
	python -m py_compile dash_app/data_utils.py
	python -m py_compile dash_app/pages.py
	python -m py_compile dash_app/charts.py
	python -m py_compile dash_app/callbacks.py
	python -m py_compile viz/utils.py

run-producer: ## Run the crypto producer
	python -m api.crypto_producer

run-pipeline: ## Run the data pipeline processor
	python -m pipeline.runner

run-dash: ## Run the Dash dashboard
	python -m dash_app.app

validate: ## Validate configuration
	python -m config.validate

clean: ## Remove Python cache files
	if exist .\__pycache__ rmdir /s /q .\__pycache__
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
