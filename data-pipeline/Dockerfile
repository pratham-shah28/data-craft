FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    g++ \
    gcc \
    cmake \
    libpq-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Airflow FIRST (explicitly with constraints)
RUN pip install --no-cache-dir \
    "apache-airflow==2.9.2" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-3.9.txt"

# Then install other packages, skipping google-re2 and apache-airflow
RUN sed '/google-re2/Id; /apache-airflow/Id' requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt || true && \
    rm requirements_filtered.txt

COPY . .

RUN git init && \
    git config --global user.email "airflow@datacraft.com" && \
    git config --global user.name "Airflow Pipeline" && \
    git add -A && \
    git commit -m "Initial commit" || true


RUN mkdir -p airflow/logs config/dataset_profiles

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV AIRFLOW_HOME=/app/airflow

CMD ["tail", "-f", "/dev/null"]