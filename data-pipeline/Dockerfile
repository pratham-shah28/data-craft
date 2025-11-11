FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies--> # Added packages for building certain Python libraries
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

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

## Install dependencies with fallback for google-re2

RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "⚠️ Some packages failed. Retrying without google-re2..." && \
     grep -v "google-re2" requirements.txt > requirements_temp.txt && \
     pip install --no-cache-dir -r requirements_temp.txt && \
     rm requirements_temp.txt)

# Copy project files
COPY . .

# Create necessary directories --> Creates only the missing directories
RUN mkdir -p \
    data/raw \
    data/processed \
    data/validated \
    data/unstructured \
    logs \
    reports \
    config/dataset_profiles \
    airflow/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV AIRFLOW_HOME=/app/airflow

# Default command (keeps container running)
CMD ["tail", "-f", "/dev/null"]