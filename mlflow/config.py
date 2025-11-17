import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "datacraft-data-pipeline")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "isha-retail-data")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db")
MLFLOW_ARTIFACT_ROOT = f"gs://{GCS_BUCKET}/mlflow-artifacts"

EXPERIMENTS = {
    "llm2_text_to_sql": "LLM2-Text-to-SQL-Pipeline",
    "llm1_document_extraction": "LLM1-Document-Extraction"
}

DEFAULT_TAGS = {
    "project": "DataCraft-MLOps",
    "team": "datacraft-team"
}
