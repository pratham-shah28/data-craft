"""
MLflow Configuration Settings
Centralized configuration for MLflow tracking in DataCraft Platform
"""

import os
from pathlib import Path
from typing import Optional


class MLflowConfig:
    """
    MLflow configuration manager
    Handles all MLflow-related settings and paths
    """
    
    # Project Settings
    PROJECT_NAME = "datacraft-mlops"
    EXPERIMENT_NAME = "LLM2_Model_Evaluation"
    
    # Storage Settings - Local File-Based (No Database Needed!)
    MLFLOW_HOME = Path(__file__).parent.resolve()
    TRACKING_URI = f"file://{MLFLOW_HOME}/mlruns"
    ARTIFACT_LOCATION = f"{MLFLOW_HOME}/mlruns"
    
    # Optional: GCS Storage (if you want cloud artifacts)
    USE_GCS_ARTIFACTS = os.environ.get("MLFLOW_USE_GCS", "false").lower() == "true"
    GCS_BUCKET = os.environ.get("MLFLOW_GCS_BUCKET", "isha-retail-data")
    GCS_ARTIFACT_PATH = "mlflow-artifacts"
    
    # Model Settings - BOTH MODELS TRACKED
    MODELS_TO_TRACK = [
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    ]
    
    # Logging Settings
    LOG_PARAMS = True
    LOG_METRICS = True
    LOG_ARTIFACTS = True
    AUTO_LOG = False  # We'll do manual logging for better control
    
    # Run Settings
    RUN_NAME_PREFIX = "model_eval"
    NESTED_RUNS = True  # Enable nested runs for comparing models
    
    @classmethod
    def get_tracking_uri(cls) -> str:
        """Get the MLflow tracking URI"""
        if cls.USE_GCS_ARTIFACTS:
            return f"gs://{cls.GCS_BUCKET}/{cls.GCS_ARTIFACT_PATH}"
        return cls.TRACKING_URI
    
    @classmethod
    def get_artifact_location(cls, run_id: Optional[str] = None) -> str:
        """Get artifact storage location"""
        base_path = cls.ARTIFACT_LOCATION
        if run_id:
            return f"{base_path}/{run_id}/artifacts"
        return base_path
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories for MLflow"""
        mlruns_dir = cls.MLFLOW_HOME / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created MLflow directory: {mlruns_dir}")
        
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get configuration summary for logging"""
        return {
            "project_name": cls.PROJECT_NAME,
            "experiment_name": cls.EXPERIMENT_NAME,
            "tracking_uri": cls.get_tracking_uri(),
            "artifact_location": cls.ARTIFACT_LOCATION,
            "models_tracked": cls.MODELS_TO_TRACK,
            "use_gcs": cls.USE_GCS_ARTIFACTS,
            "log_params": cls.LOG_PARAMS,
            "log_metrics": cls.LOG_METRICS,
            "log_artifacts": cls.LOG_ARTIFACTS,
        }


# Initialize directories on import
MLflowConfig.setup_directories()


if __name__ == "__main__":
    """Test configuration"""
    print("\n" + "=" * 60)
    print("MLFLOW CONFIGURATION SUMMARY")
    print("=" * 60)
    
    config = MLflowConfig.get_config_summary()
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    
    print("\n✓ Configuration loaded successfully!")
