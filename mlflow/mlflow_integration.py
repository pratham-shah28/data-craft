"""
MLflow Integration for Model Pipeline DAG
Simple helper functions to add MLflow tracking to existing DAG
NO MODIFICATIONS to existing task logic - just logging wrapper
"""

import mlflow
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add mlflow directory to path
sys.path.insert(0, str(Path(__file__).parent))
from mlflow_tracker import MLflowTracker
from mlflow_config import MLflowConfig


class SimpleMLflowLogger:
    """
    Simple MLflow logger for model pipeline
    Tracks BOTH models: gemini-2.5-flash AND gemini-2.5-pro
    """
    
    def __init__(self):
        self.tracker = MLflowTracker()
        self.experiment_name = MLflowConfig.EXPERIMENT_NAME
        self.parent_run_id = None
        self.model_run_ids = {}
        
    def start_experiment(self, **context):
        """Start MLflow experiment - call at DAG start"""
        dag_run = context.get('dag_run', {})
        execution_date = context.get('execution_date', 'unknown')
        
        run_name = f"pipeline_{execution_date}"
        
        # Start parent run
        run = self.tracker.start_run(run_name=run_name, nested=False)
        self.parent_run_id = run.info.run_id  # Get actual run ID
        
        # Log basic info
        self.tracker.set_tag("dag_id", "model_pipeline_with_evaluation")
        self.tracker.set_tag("models", "gemini-2.5-flash,gemini-2.5-pro")
        self.tracker.log_param("dataset", "walmart_retail")
        
        # End the run immediately (we'll resume it later)
        self.tracker.end_run()
        
        # Store run ID in XCom for later tasks
        context['ti'].xcom_push(key='mlflow_parent_run_id', value=self.parent_run_id)
        
        print(f"\n✓ MLflow experiment started: {run_name}")
        print(f"  Run ID: {self.parent_run_id}\n")
        
        return {"status": "started", "run_id": self.parent_run_id}
    
    def log_all_results(self, **context):
        """Extract all results from XCom and log to MLflow - call at DAG end"""
        ti = context['ti']
        
        # Get parent run ID from XCom
        parent_run_id = ti.xcom_pull(key='mlflow_parent_run_id')
        if not parent_run_id:
            # Try to use stored ID
            parent_run_id = self.parent_run_id
        
        if not parent_run_id:
            print("⚠ No MLflow run found, skipping logging")
            return {"status": "skipped"}
        
        print(f"\n{'='*60}")
        print("📊 LOGGING RESULTS TO MLFLOW")
        print(f"   Parent Run ID: {parent_run_id}")
        print(f"{'='*60}\n")
        
        # Extract data from XCom
        all_model_responses = ti.xcom_pull(task_ids='process_queries_with_all_models', 
                                           key='all_model_responses') or {}
        eval_reports = ti.xcom_pull(task_ids='evaluate_all_models', 
                                    key='evaluation_reports') or {}
        bias_reports = ti.xcom_pull(task_ids='detect_bias_in_all_models', 
                                    key='bias_reports') or {}
        model_selection = ti.xcom_pull(task_ids='select_best_model', 
                                      key='model_selection') or {}
        
        # Set parent run as active
        mlflow.start_run(run_id=parent_run_id)
        
        # Log each model's results
        for model_name in ["gemini-2.5-flash", "gemini-2.5-pro"]:  # BOTH MODELS
            if model_name not in all_model_responses:
                continue
                
            print(f"\n🤖 Logging {model_name}...")
            
            # Start nested run for this model
            with mlflow.start_run(run_name=model_name, nested=True) as model_run:
                # Log model config
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("temperature", 0.2)
                mlflow.log_param("top_p", 0.8)
                mlflow.log_param("top_k", 40)
                mlflow.log_param("max_output_tokens", 2048)
                
                # Log query metrics
                responses = all_model_responses[model_name]
                successful = sum(1 for r in responses if r.get("status") == "success")
                total = len(responses)
                avg_time = sum(r.get("response_time", 0) for r in responses if r.get("status") == "success") / successful if successful > 0 else 0
                
                mlflow.log_metric("total_queries", total)
                mlflow.log_metric("successful_queries", successful)
                mlflow.log_metric("success_rate", (successful/total*100) if total > 0 else 0)
                mlflow.log_metric("avg_response_time", avg_time)
                
                # Log evaluation metrics
                if model_name in eval_reports:
                    report = eval_reports[model_name]
                    mlflow.log_metric("syntax_valid_percent", report.get("syntax_valid_percent", 0))
                    mlflow.log_metric("executable_percent", report.get("executable_percent", 0))
                    mlflow.log_metric("intent_match_score", report.get("intent_match_score", 0))
                    mlflow.log_metric("overall_score", report.get("overall_score", 0))
                
                # Log bias metrics
                if model_name in bias_reports:
                    report = bias_reports[model_name]
                    mlflow.log_metric("bias_score", report.get("overall_bias_score", 0))
                    mlflow.log_metric("fairness_score", report.get("fairness_score", 100))
                    mlflow.log_metric("bias_flags", report.get("total_flags", 0))
                
                print(f"   ✓ Logged metrics for {model_name}")
        
        # Log model selection in parent run
        if model_selection and "best_model" in model_selection:
            best = model_selection["best_model"]
            mlflow.log_param("best_model_selected", best.get("name", "unknown"))
            mlflow.log_metric("best_composite_score", best.get("composite_score", 0))
            mlflow.log_metric("best_performance_score", best.get("performance_score", 0))
            mlflow.log_metric("best_bias_score", best.get("bias_score", 0))
            
            print(f"\n🏆 Best Model: {best.get('name')}")
            print(f"   Composite Score: {best.get('composite_score', 0):.2f}")
        
        # End parent run
        mlflow.end_run()
        
        print(f"\n{'='*60}")
        print("✅ MLFLOW LOGGING COMPLETE")
        print(f"   View: http://localhost:5000")
        print(f"{'='*60}\n")
        
        return {"status": "logged", "models": ["gemini-2.5-flash", "gemini-2.5-pro"]}


# Global logger instance
_logger = SimpleMLflowLogger()


# Airflow task functions
def start_mlflow_experiment(**context):
    """Airflow task: Start MLflow tracking"""
    return _logger.start_experiment(**context)


def log_to_mlflow(**context):
    """Airflow task: Log all results to MLflow"""
    return _logger.log_all_results(**context)


if __name__ == "__main__":
    """Test the logger"""
    print("\n" + "="*60)
    print("TESTING SIMPLE MLFLOW LOGGER")
    print("="*60 + "\n")
    
    logger = SimpleMLflowLogger()
    
    # Mock context
    class MockTI:
        def xcom_push(self, key, value):
            print(f"  XCom push: {key} = {value}")
        
        def xcom_pull(self, task_ids=None, key=None):
            # Return mock data
            if key == 'all_model_responses':
                return {
                    "gemini-2.5-flash": [
                        {"status": "success", "response_time": 2.5},
                        {"status": "success", "response_time": 3.0}
                    ],
                    "gemini-2.5-pro": [
                        {"status": "success", "response_time": 3.2},
                        {"status": "success", "response_time": 3.8}
                    ]
                }
            elif key == 'evaluation_reports':
                return {
                    "gemini-2.5-flash": {
                        "syntax_valid_percent": 90.0,
                        "executable_percent": 85.0,
                        "intent_match_score": 88.0,
                        "overall_score": 87.5
                    },
                    "gemini-2.5-pro": {
                        "syntax_valid_percent": 95.0,
                        "executable_percent": 90.0,
                        "intent_match_score": 92.0,
                        "overall_score": 91.0
                    }
                }
            elif key == 'bias_reports':
                return {
                    "gemini-2.5-flash": {"overall_bias_score": 25.0, "fairness_score": 75.0, "total_flags": 8},
                    "gemini-2.5-pro": {"overall_bias_score": 22.0, "fairness_score": 78.0, "total_flags": 6}
                }
            elif key == 'model_selection':
                return {
                    "best_model": {
                        "name": "gemini-2.5-pro",
                        "composite_score": 91.2,
                        "performance_score": 92.0,
                        "bias_score": 22.0
                    }
                }
            return None
    
    context = {
        'ti': MockTI(),
        'execution_date': '2025-11-18',
        'dag_run': type('obj', (object,), {})()
    }
    
    # Test start
    print("1. Testing start_experiment...")
    result = logger.start_experiment(**context)
    print(f"   Result: {result}")
    
    # Test logging (with mock data)
    print("\n2. Testing log_all_results...")
    result = logger.log_all_results(**context)
    print(f"   Result: {result}")
    
    print("\n✓ Test complete!")
