ARG SPARK_VERSION=3.5.0
FROM python:3.10-slim

ARG SPARK_VERSION

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps netcat-openbsd wget \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    (apt-get install -y --no-install-recommends openjdk-17-jre-headless || \
     apt-get install -y --no-install-recommends openjdk-21-jre-headless) && \
    rm -rf /var/lib/apt/lists/* && \
    java -version 2>&1 | head -1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import pyspark; print(f'PySpark {pyspark.__version__} ready')"

COPY . .

ENV PYTHONPATH=/app
