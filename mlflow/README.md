# MLflow Experiment Tracking


## Overview

Tracks experiments for gemini-2.5-flash and gemini-2.5-pro models. Logs parameters, metrics, and model selection results.

## Setup

1. Install dependencies:
```bash
cd mlflow
pip3 install -r requirements-mlflow.txt
```

2. Start MLflow UI:
```bash
python3 -m mlflow ui --backend-store-uri file:///Users/viraj/Desktop/mlops-project/mlflow/mlruns --port 5000
```

3. Access dashboard:
```
http://127.0.0.1:5000
```

## Integration

Modified `model-training/dags/model_pipeline_dag.py` with 23 lines:
- Added start_mlflow_tracking task (runs first)
- Added log_to_mlflow task (runs last)
- Both models preserved in MODELS_TO_EVALUATE

## What Gets Logged

### Parameters (per model)
- model_name
- temperature (0.2)
- top_p (0.8)
- top_k (40)
- max_output_tokens (2048)

### Metrics (per model)
- total_queries
- successful_queries
- success_rate
- avg_response_time
- syntax_valid_percent
- executable_percent
- intent_match_score
- overall_score
- bias_score
- fairness_score
- bias_flags

### Model Selection (parent run)
- best_model_selected
- best_composite_score
- best_performance_score
- best_bias_score

## Files Created

Core integration:
- mlflow_config.py - Configuration
- mlflow_tracker.py - MLflow wrapper
- mlflow_integration.py - DAG integration functions
- requirements-mlflow.txt - Dependencies

Documentation:
- TASK.md - Progress tracker
- CHANGES_APPLIED.md - Change log
- MLFLOW_DASHBOARD_GUIDE.md - UI guide
- COMPLETION_SUMMARY.md - Final summary

Testing:
- test_mlflow_integration.py - Integration test

## Usage

After DAG completes:
1. Open MLflow UI at http://127.0.0.1:5000
2. Click on "LLM2_Model_Evaluation" experiment
3. View runs with both model evaluations
4. Select two model runs and click "Compare" to see side-by-side metrics
5. Check parent run for model selection decision

## Testing

Run integration test:
```bash
cd mlflow
python3 test_mlflow_integration.py
```

Expected output: Both models logged with all metrics.

## Storage

Local file-based storage in mlflow/mlruns/. No database required.

## Status

Implementation complete. Testing verified. Ready for merge.
