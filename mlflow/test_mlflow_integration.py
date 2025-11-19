"""
Quick MLflow Integration Test
Simulates a complete DAG run with BOTH models to verify MLflow logging
"""

import sys
from pathlib import Path

# Add mlflow directory to path
sys.path.insert(0, str(Path(__file__).parent))
from mlflow_integration import SimpleMLflowLogger

print("\n" + "="*70)
print("🧪 MLFLOW INTEGRATION TEST - SIMULATING FULL DAG RUN")
print("="*70 + "\n")

# Create logger
logger = SimpleMLflowLogger()

# ========================================
# SIMULATE AIRFLOW CONTEXT
# ========================================

class MockTaskInstance:
    """Mock Airflow TaskInstance"""
    def __init__(self):
        self.xcom_data = {}
    
    def xcom_push(self, key, value):
        self.xcom_data[key] = value
        print(f"  📤 XCom Push: {key}")
    
    def xcom_pull(self, task_ids=None, key=None):
        """Return mock data based on key"""
        
        if key == 'all_model_responses':
            # Simulate responses from BOTH models
            return {
                "gemini-2.5-flash": [
                    {"query_number": 1, "user_query": "What are total sales?", 
                     "status": "success", "response_time": 2.1,
                     "sql_query": "SELECT SUM(sales) FROM dataset",
                     "visualization": {"type": "number", "title": "Total Sales"}},
                    {"query_number": 2, "user_query": "Top 5 products?", 
                     "status": "success", "response_time": 2.5,
                     "sql_query": "SELECT product, sales FROM dataset ORDER BY sales DESC LIMIT 5",
                     "visualization": {"type": "bar_chart", "title": "Top Products"}},
                    {"query_number": 3, "user_query": "Sales by region?", 
                     "status": "success", "response_time": 2.8},
                    {"query_number": 4, "user_query": "Failed query", 
                     "status": "failed", "response_time": 0, "error": "Invalid syntax"},
                ],
                "gemini-2.5-pro": [
                    {"query_number": 1, "user_query": "What are total sales?", 
                     "status": "success", "response_time": 3.2,
                     "sql_query": "SELECT SUM(sales) FROM dataset",
                     "visualization": {"type": "number", "title": "Total Sales"}},
                    {"query_number": 2, "user_query": "Top 5 products?", 
                     "status": "success", "response_time": 3.5,
                     "sql_query": "SELECT product, sales FROM dataset ORDER BY sales DESC LIMIT 5",
                     "visualization": {"type": "bar_chart", "title": "Top Products"}},
                    {"query_number": 3, "user_query": "Sales by region?", 
                     "status": "success", "response_time": 3.8},
                    {"query_number": 4, "user_query": "Monthly trends?", 
                     "status": "success", "response_time": 4.0},
                ]
            }
        
        elif key == 'evaluation_reports':
            # Simulate evaluation reports from BOTH models
            return {
                "gemini-2.5-flash": {
                    "model_name": "gemini-2.5-flash",
                    "total_queries": 4,
                    "syntax_valid": 3,
                    "syntax_valid_percent": 75.0,
                    "executable": 3,
                    "executable_percent": 75.0,
                    "intent_match_score": 85.5,
                    "avg_response_time": 2.47,
                    "success_rate_percent": 75.0,
                    "overall_score": 78.8
                },
                "gemini-2.5-pro": {
                    "model_name": "gemini-2.5-pro",
                    "total_queries": 4,
                    "syntax_valid": 4,
                    "syntax_valid_percent": 100.0,
                    "executable": 4,
                    "executable_percent": 100.0,
                    "intent_match_score": 92.3,
                    "avg_response_time": 3.63,
                    "success_rate_percent": 100.0,
                    "overall_score": 96.2
                }
            }
        
        elif key == 'bias_reports':
            # Simulate bias reports from BOTH models
            return {
                "gemini-2.5-flash": {
                    "model_name": "gemini-2.5-flash",
                    "overall_bias_score": 28.5,
                    "fairness_score": 71.5,
                    "total_flags": 8,
                    "high_severity_flags": 2,
                    "medium_severity_flags": 6,
                    "total_queries": 4
                },
                "gemini-2.5-pro": {
                    "model_name": "gemini-2.5-pro",
                    "overall_bias_score": 22.0,
                    "fairness_score": 78.0,
                    "total_flags": 6,
                    "high_severity_flags": 1,
                    "medium_severity_flags": 5,
                    "total_queries": 4
                }
            }
        
        elif key == 'model_selection':
            # Simulate model selection
            return {
                "best_model": {
                    "name": "gemini-2.5-pro",
                    "composite_score": 91.2,
                    "performance_score": 96.2,
                    "bias_score": 22.0
                },
                "ranking": [
                    {"model": "gemini-2.5-pro", "score": 91.2},
                    {"model": "gemini-2.5-flash", "score": 78.8}
                ]
            }
        
        return None

class MockDagRun:
    """Mock Airflow DagRun"""
    pass

class MockDag:
    """Mock Airflow DAG"""
    dag_id = "model_pipeline_with_evaluation"

# Create mock context
mock_context = {
    'ti': MockTaskInstance(),
    'execution_date': '2025-11-18_test_run',
    'dag_run': MockDagRun(),
    'dag': MockDag()
}

# ========================================
# RUN TEST
# ========================================

print("📊 STEP 1: Starting MLflow Experiment...")
print("-" * 70)
start_result = logger.start_experiment(**mock_context)
print(f"✅ Experiment started!")
print(f"   Run ID: {start_result['run_id']}")
print()

print("📊 STEP 2: Logging All Results to MLflow...")
print("-" * 70)
log_result = logger.log_all_results(**mock_context)
print(f"\n✅ Logging complete!")
print(f"   Status: {log_result['status']}")
print(f"   Models logged: {log_result.get('models', [])}")

print("\n" + "="*70)
print("✅ TEST COMPLETE!")
print("="*70)

print(f"""
🎯 NEXT STEPS:

1. Open MLflow UI in your browser:
   → http://127.0.0.1:5000

2. Click on "LLM2_Model_Evaluation" experiment

3. You should see the test run with:
   ✓ 1 parent run (pipeline_2025-11-18_test_run)
   ✓ 2 nested runs (gemini-2.5-flash, gemini-2.5-pro)

4. Click on the parent run to see:
   ✓ Parameters: dataset, best_model_selected
   ✓ Metrics: best_composite_score, best_performance_score, best_bias_score

5. Click on each nested model run to see:
   ✓ Parameters: model_name, temperature, top_p, top_k, max_output_tokens
   ✓ Metrics: 
      - Query metrics: total_queries, success_rate, avg_response_time
      - Evaluation metrics: syntax_valid_percent, executable_percent, intent_match_score
      - Bias metrics: bias_score, fairness_score, bias_flags

6. Use the "Compare" button to compare both models side-by-side!

📊 The dashboard will show you which model performed better!
""")
