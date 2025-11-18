# MLflow Testing Commands

## Test MLflow Integration

```bash
python3 test_mlflow_integration.py
```

Expected output: Both models logged successfully.

## Start MLflow UI

```bash
python3 -m mlflow ui --backend-store-uri file://./mlflow/mlruns --port 5000
```

Then open: http://127.0.0.1:5000

## View Logged Data

1. Open http://127.0.0.1:5000
2. Click "LLM2_Model_Evaluation"
3. See runs with both models
4. Click any run to see details
5. Select two model runs and click "Compare"

## Stop MLflow UI

```bash
# Find process
ps aux | grep mlflow | grep -v grep

# Kill it (replace PID)
kill <PID>
```

## Test with Actual Airflow DAG

```bash
# Start Docker
docker-compose up -d

# Wait 60 seconds for Airflow to start
sleep 60

# Open Airflow UI
# http://localhost:8080
# Username: admin
# Password: admin

# Trigger DAG: model_pipeline_with_evaluation

# After DAG completes, check MLflow UI for new run
```
