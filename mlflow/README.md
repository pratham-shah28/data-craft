# MLflow Experiment Tracking

This directory contains MLflow configuration and tracking utilities for the DataCraft MLOps project.

## Structure

```
mlflow/
├── config.py          # Configuration for tracking URI, experiments, artifacts
├── tracker.py         # MLflowTracker class for experiment tracking
├── test_setup.py      # Test script to verify MLflow setup
├── mlflow.db          # SQLite database (created after first run)
└── README.md          # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test MLflow setup:
```bash
cd /Users/viraj/Desktop/mlops-project
python3 mlflow/test_setup.py
```

3. Start MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```

4. Access UI at: http://localhost:5000

## Configuration

Key settings in `config.py`:
- Tracking URI: Local SQLite database
- Artifact Storage: Google Cloud Storage bucket
- Experiments: LLM2 and LLM1 tracking

## Usage

```python
from mlflow.tracker import MLflowTracker
from mlflow.config import EXPERIMENTS, MLFLOW_TRACKING_URI

tracker = MLflowTracker(
    experiment_name=EXPERIMENTS["llm2_text_to_sql"],
    tracking_uri=MLFLOW_TRACKING_URI
)

tracker.start_run(run_name="my_experiment")
tracker.log_params({"model": "gemini-2.5-pro", "temperature": 0.2})
tracker.log_metrics({"accuracy": 0.95})
tracker.end_run()
```
