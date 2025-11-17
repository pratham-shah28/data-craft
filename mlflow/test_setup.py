#!/usr/bin/env python3

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tracker import MLflowTracker
from config import MLFLOW_TRACKING_URI, EXPERIMENTS, DEFAULT_TAGS

def test_mlflow_setup():
    print("\n" + "="*60)
    print("Testing MLflow Setup")
    print("="*60 + "\n")
    
    print(f"1. Tracking URI: {MLFLOW_TRACKING_URI}")
    
    tracker = MLflowTracker(
        experiment_name=EXPERIMENTS["llm2_text_to_sql"],
        tracking_uri=MLFLOW_TRACKING_URI
    )
    print(f"2. Created tracker for: {EXPERIMENTS['llm2_text_to_sql']}")
    
    tracker.start_run(run_name="test_run", tags=DEFAULT_TAGS)
    print("3. Started test run")
    
    test_params = {
        "model": "test-model",
        "temperature": 0.5,
        "max_tokens": 100
    }
    tracker.log_params(test_params)
    print(f"4. Logged parameters: {test_params}")
    
    test_metrics = {
        "accuracy": 0.95,
        "latency": 1.2
    }
    tracker.log_metrics(test_metrics)
    print(f"5. Logged metrics: {test_metrics}")
    
    tracker.end_run()
    print("6. Ended run successfully")
    
    print("\n" + "="*60)
    print("MLflow Setup Test: PASSED")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        test_mlflow_setup()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
