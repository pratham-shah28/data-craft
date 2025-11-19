"""
MLflow Tracker - Core Tracking Wrapper
Provides easy-to-use interface for logging experiments, metrics, and artifacts
"""

import mlflow
import mlflow.entities
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime
import logging

from mlflow_config import MLflowConfig


class MLflowTracker:
    """
    Core MLflow tracking wrapper
    Simplifies experiment tracking for the DataCraft Platform
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the experiment (default: from config)
            tracking_uri: MLflow tracking URI (default: from config)
        """
        self.experiment_name = experiment_name or MLflowConfig.EXPERIMENT_NAME
        self.tracking_uri = tracking_uri or MLflowConfig.get_tracking_uri()
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create/get experiment
        self.experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = self.experiment.experiment_id
        
        # Initialize client
        self.client = MlflowClient()
        
        # Current run context
        self.current_run = None
        self.current_run_id = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"✓ MLflow Tracker initialized")
        self.logger.info(f"  Experiment: {self.experiment_name}")
        self.logger.info(f"  Tracking URI: {self.tracking_uri}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for tracker"""
        logger = logging.getLogger("MLflowTracker")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for this run
            tags: Dictionary of tags to add
            nested: Whether this is a nested run
            
        Returns:
            Active MLflow run context
        """
        # Generate run name if not provided
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{MLflowConfig.RUN_NAME_PREFIX}_{timestamp}"
        
        # Start run
        self.current_run = mlflow.start_run(
            run_name=run_name,
            nested=nested
        )
        self.current_run_id = self.current_run.info.run_id
        
        # Add tags
        if tags:
            mlflow.set_tags(tags)
        
        # Add default tags
        mlflow.set_tag("project", MLflowConfig.PROJECT_NAME)
        mlflow.set_tag("experiment", self.experiment_name)
        
        self.logger.info(f"✓ Started run: {run_name} (ID: {self.current_run_id})")
        
        return self.current_run
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.current_run:
            mlflow.end_run(status=status)
            self.logger.info(f"✓ Ended run: {self.current_run_id} ({status})")
            self.current_run = None
            self.current_run_id = None
        else:
            self.logger.warning("No active run to end")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to current run
        
        Args:
            params: Dictionary of parameters
        """
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.log_params(params)
        self.logger.info(f"✓ Logged {len(params)} parameters")
    
    def log_param(self, key: str, value: Any):
        """Log a single parameter"""
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.log_param(key, value)
        self.logger.debug(f"✓ Logged param: {key}={value}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.log_metrics(metrics, step=step)
        self.logger.info(f"✓ Logged {len(metrics)} metrics")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric"""
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.log_metric(key, value, step=step)
        self.logger.debug(f"✓ Logged metric: {key}={value}")
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None):
        """
        Log a single artifact file
        
        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifact store
        """
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        self.logger.info(f"✓ Logged artifact: {local_path}")
    
    def log_artifacts(self, local_dir: Union[str, Path], artifact_path: Optional[str] = None):
        """
        Log all files in a directory as artifacts
        
        Args:
            local_dir: Path to local directory
            artifact_path: Optional path within artifact store
        """
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
        self.logger.info(f"✓ Logged artifacts from: {local_dir}")
    
    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as a JSON artifact
        
        Args:
            dictionary: Dictionary to log
            filename: Name of the JSON file
        """
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.log_dict(dictionary, filename)
        self.logger.info(f"✓ Logged dict as artifact: {filename}")
    
    def log_text(self, text: str, filename: str):
        """
        Log text as an artifact
        
        Args:
            text: Text content
            filename: Name of the text file
        """
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.log_text(text, filename)
        self.logger.info(f"✓ Logged text as artifact: {filename}")
    
    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags on current run"""
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.set_tags(tags)
        self.logger.info(f"✓ Set {len(tags)} tags")
    
    def set_tag(self, key: str, value: str):
        """Set a single tag on current run"""
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        mlflow.set_tag(key, value)
        self.logger.debug(f"✓ Set tag: {key}={value}")
    
    def get_run_info(self) -> Optional[Dict]:
        """Get information about current run"""
        if not self.current_run:
            self.logger.warning("No active run")
            return None
        
        return {
            "run_id": self.current_run.info.run_id,
            "run_name": self.current_run.info.run_name,
            "experiment_id": self.current_run.info.experiment_id,
            "status": self.current_run.info.status,
            "start_time": self.current_run.info.start_time,
            "artifact_uri": self.current_run.info.artifact_uri
        }
    
    def log_model_comparison(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        artifact_name: str = "model_comparison.json"
    ):
        """
        Log comparison between multiple models
        
        Args:
            model_metrics: Dictionary mapping model names to their metrics
            artifact_name: Name for the comparison artifact
        """
        if not self.current_run:
            self.logger.error("No active run! Start a run first.")
            return
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models": model_metrics,
            "best_model": max(
                model_metrics.items(),
                key=lambda x: x[1].get('composite_score', 0)
            )[0] if model_metrics else None
        }
        
        self.log_dict(comparison, artifact_name)
        self.logger.info(f"✓ Logged model comparison: {artifact_name}")


if __name__ == "__main__":
    """Test MLflow tracker"""
    print("\n" + "=" * 60)
    print("TESTING MLFLOW TRACKER")
    print("=" * 60)
    
    # Initialize tracker
    tracker = MLflowTracker()
    
    # Test run
    print("\n1. Starting test run...")
    tracker.start_run("test_run", tags={"test": "true"})
    
    print("\n2. Logging parameters...")
    tracker.log_params({
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "top_p": 0.8
    })
    
    print("\n3. Logging metrics...")
    tracker.log_metrics({
        "accuracy": 85.3,
        "response_time": 2.1,
        "bias_score": 25.0
    })
    
    print("\n4. Logging test artifact...")
    tracker.log_dict(
        {"test": "data", "status": "success"},
        "test_artifact.json"
    )
    
    print("\n5. Getting run info...")
    info = tracker.get_run_info()
    print(f"   Run ID: {info['run_id']}")
    print(f"   Artifact URI: {info['artifact_uri']}")
    
    print("\n6. Ending run...")
    tracker.end_run()
    
    print("\n" + "=" * 60)
    print("✓ MLflow Tracker test completed successfully!")
    print("=" * 60)
