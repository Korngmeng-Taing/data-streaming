FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=300
RUN pip install --no-cache-dir --retries 5 -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
