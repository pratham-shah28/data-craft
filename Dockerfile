FROM apache/airflow:2.7.3-python3.9

USER root

# Install essential system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories (unified structure)
RUN mkdir -p /opt/airflow/dags/data-pipeline \
             /opt/airflow/dags/model-training \
             /opt/airflow/data-pipeline/scripts \
             /opt/airflow/data-pipeline/config \
             /opt/airflow/model-training/scripts \
             /opt/airflow/model-training/data \
             /opt/airflow/shared/utils \
             /opt/airflow/logs \
             /opt/airflow/outputs \
             /opt/airflow/gcp

USER airflow

# Copy and install Python dependencies
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Environment configuration
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow:/opt/airflow/data-pipeline/scripts:/opt/airflow/model-training/scripts:/opt/airflow/shared
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
