import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


class MLflowTracker:
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{timestamp}"
        
        mlflow.start_run(run_name=run_name)
        
        if tags:
            mlflow.set_tags(tags)
        
        return mlflow.active_run()
    
    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        if os.path.exists(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_dict(self, dictionary: Dict, filename: str):
        temp_file = f"/tmp/{filename}"
        with open(temp_file, 'w') as f:
            json.dump(dictionary, f, indent=2)
        mlflow.log_artifact(temp_file)
        os.remove(temp_file)
    
    def end_run(self, status: str = "FINISHED"):
        mlflow.end_run(status=status)
    
    def log_model_info(self, model_name: str, model_version: str, model_params: Dict[str, Any]):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_version", model_version)
        self.log_params(model_params)
    
    @staticmethod
    def get_experiment_runs(experiment_name: str, max_results: int = 100):
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            return runs
        return []
