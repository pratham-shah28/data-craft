import sys
import os
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'mlflow'))

from tracker import MLflowTracker
from config import EXPERIMENTS, MLFLOW_TRACKING_URI, DEFAULT_TAGS


class LLM2Tracker:
    
    def __init__(self, dataset_name, model_name, generation_config):
        self.tracker = MLflowTracker(
            experiment_name=EXPERIMENTS["llm2_text_to_sql"],
            tracking_uri=MLFLOW_TRACKING_URI
        )
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.generation_config = generation_config
        self.start_time = None
        self.query_times = []
        
    def start_run(self, run_name=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"llm2_run_{timestamp}"
        
        tags = {
            **DEFAULT_TAGS,
            "model": self.model_name,
            "dataset": self.dataset_name,
            "pipeline_type": "text-to-sql"
        }
        
        self.tracker.start_run(run_name=run_name, tags=tags)
        self.start_time = time.time()
        
        params = {
            "model_name": self.model_name,
            "temperature": self.generation_config.get("temperature", 0.2),
            "top_p": self.generation_config.get("top_p", 0.8),
            "top_k": self.generation_config.get("top_k", 40),
            "max_output_tokens": self.generation_config.get("max_output_tokens", 2048),
            "dataset_name": self.dataset_name,
        }
        
        self.tracker.log_params(params)
        
    def log_query_processing(self, query_num, success, response_time):
        self.query_times.append(response_time)
        
        metrics = {
            f"query_{query_num}_success": 1 if success else 0,
            f"query_{query_num}_response_time": response_time
        }
        
        self.tracker.log_metrics(metrics, step=query_num)
    
    def log_final_metrics(self, total_queries, successful_queries, failed_queries):
        total_time = time.time() - self.start_time
        avg_response_time = sum(self.query_times) / len(self.query_times) if self.query_times else 0
        
        final_metrics = {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "avg_response_time": avg_response_time,
            "total_pipeline_time": total_time,
            "queries_per_minute": (total_queries / total_time * 60) if total_time > 0 else 0
        }
        
        self.tracker.log_metrics(final_metrics)
    
    def log_artifacts(self, output_dir):
        if os.path.exists(output_dir):
            self.tracker.log_artifacts(output_dir, artifact_path="query_results")
    
    def end_run(self, status="FINISHED"):
        self.tracker.end_run(status=status)
