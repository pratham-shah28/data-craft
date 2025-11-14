"""
Complete Model Pipeline DAG with Evaluation, Bias Detection & Best Model Selection
✅ Uses modular scripts: model_evaluator.py, bias_detector.py, model_selector.py, response_saver.py
✅ Evaluates multiple Gemini models and selects the best one
✅ Saves only the best model's responses
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable
import json
import os
import sys
from pathlib import Path
import time

# ✅ Use unified Docker paths
sys.path.insert(0, '/opt/airflow/model-training/scripts')
sys.path.insert(0, '/opt/airflow/shared')

# Import existing modules
from data_loader import ModelDataLoader
from feature_engineering import FeatureEngineer
from metadata_manager import MetadataManager
from prompts import build_prompt, FEW_SHOT_EXAMPLES

# ✅ Import new evaluation modules
from model_evaluator import ModelEvaluator
from bias_detector import BiasDetector
from model_selector import ModelSelector
from response_saver import ResponseSaver
from query_executor import QueryExecutor  

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "datacraft-data-pipeline")
REGION = Variable.get("REGION", default_var="us-central1")
BUCKET_NAME = Variable.get("BUCKET_NAME", default_var="isha-retail-data")
DATASET_ID = Variable.get("BQ_DATASET", default_var="datacraft_ml")

# ✅ Multiple models for evaluation
MODELS_TO_EVALUATE = [
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

# Paths
QUERIES_FILE = "/opt/airflow/model-training/data/user_queries.txt"
OUTPUT_BASE_DIR = "/opt/airflow/outputs/model-training"
EVALUATION_DIR = "/opt/airflow/outputs/evaluation"
BIAS_DIR = "/opt/airflow/outputs/bias"
SELECTION_DIR = "/opt/airflow/outputs/model-selection"
BEST_MODEL_DIR = "/opt/airflow/outputs/best-model-responses"
DATASET_NAME = "orders"

# Default args
default_args = {
    'owner': 'datacraft-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['mlops0242@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# DAG definition
dag = DAG(
    'model_pipeline_with_evaluation',
    default_args=default_args,
    description='Complete pipeline: Data → Features → Multi-Model Evaluation → Bias Detection → Best Model Selection → Save Responses',
    schedule_interval=None,
    catchup=False,
    tags=['model', 'gemini', 'evaluation', 'bias-detection', 'mlops']
)


# ========================================
# PHASE 1: DATA PREPARATION
# ========================================

def load_data_to_bigquery(**context):
    """STEP 1: Load data from GCS to BigQuery"""
    ti = context['ti']
    
    print("=" * 60)
    print("STEP 1: LOADING DATA TO BIGQUERY")
    print("=" * 60)
    
    loader = ModelDataLoader(
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID
    )
    
    table_name = f"{DATASET_NAME}_processed"
    
    if loader.table_exists(table_name):
        print(f"✓ Table {table_name} already exists")
        table_info = loader.get_table_info(table_name)
        ti.xcom_push(key='row_count', value=table_info.get('num_rows', 0))
        return {"status": "exists", "table": table_name, "rows": table_info.get('num_rows', 0)}
    
    df = loader.load_processed_data_from_gcs(DATASET_NAME, stage='validated')
    table_id = loader.load_to_bigquery(df, DATASET_NAME, table_suffix="_processed")
    ti.xcom_push(key='row_count', value=len(df))
    
    print(f"\n✓ Data loaded: {len(df):,} rows")
    return {"status": "loaded", "table": table_id, "rows": len(df)}


def generate_features(**context):
    """STEP 2: Generate features and metadata"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 2: GENERATING FEATURES & METADATA")
    print("=" * 60)
    
    loader = ModelDataLoader(BUCKET_NAME, PROJECT_ID, DATASET_ID)
    df = loader.query_bigquery_table(f"{DATASET_NAME}_processed", limit=10000)
    print(f"✓ Loaded {len(df):,} rows")
    
    schema = loader.load_dataset_profile_from_gcs(DATASET_NAME)
    engineer = FeatureEngineer(df, schema, DATASET_NAME)
    
    metadata = engineer.generate_metadata()
    llm_context = engineer.create_llm_context()
    summary = engineer.get_feature_summary()
    
    ti.xcom_push(key='metadata', value=metadata)
    ti.xcom_push(key='llm_context', value=llm_context)
    ti.xcom_push(key='feature_summary', value=summary)
    
    return summary


def store_metadata_in_bigquery(**context):
    """STEP 3: Store metadata"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 3: STORING METADATA IN BIGQUERY")
    print("=" * 60)
    
    metadata = ti.xcom_pull(task_ids='generate_features', key='metadata')
    llm_context = ti.xcom_pull(task_ids='generate_features', key='llm_context')
    
    manager = MetadataManager(PROJECT_ID, DATASET_ID)
    manager.store_metadata(DATASET_NAME, metadata, llm_context)
    
    retrieved = manager.get_metadata(DATASET_NAME)
    
    if retrieved:
        print(f"✓ Metadata stored and verified")
        return {"status": "success", "dataset": DATASET_NAME}
    else:
        raise ValueError("Failed to verify metadata!")


def read_user_queries(**context):
    """STEP 4: Read user queries"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 4: READING USER QUERIES")
    print("=" * 60)
    
    if not os.path.exists(QUERIES_FILE):
        raise FileNotFoundError(f"Query file not found: {QUERIES_FILE}")
    
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    if not queries:
        raise ValueError("No queries found!")
    
    print(f"\n✓ Read {len(queries)} queries")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")
    
    ti.xcom_push(key='user_queries', value=queries)
    return {"total_queries": len(queries)}


# Task definitions for Phase 1
load_data_task = PythonOperator(
    task_id='load_data_to_bigquery',
    python_callable=load_data_to_bigquery,
    dag=dag
)

generate_features_task = PythonOperator(
    task_id='generate_features',
    python_callable=generate_features,
    dag=dag
)

store_metadata_task = PythonOperator(
    task_id='store_metadata',
    python_callable=store_metadata_in_bigquery,
    dag=dag
)

read_queries_task = PythonOperator(
    task_id='read_user_queries',
    python_callable=read_user_queries,
    dag=dag
)


# ========================================
# PHASE 2: MULTI-MODEL QUERY PROCESSING
# ========================================

def process_queries_with_all_models(**context):
    """STEP 5: Process queries with ALL models"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 5: PROCESSING QUERIES WITH MULTIPLE MODELS")
    print("=" * 60)
    
    # Get queries and metadata
    queries = ti.xcom_pull(task_ids='read_user_queries', key='user_queries')
    
    manager = MetadataManager(PROJECT_ID, DATASET_ID)
    metadata_result = manager.get_metadata(DATASET_NAME)
    
    if not metadata_result:
        raise ValueError(f"No metadata found for: {DATASET_NAME}")
    
    llm_context = metadata_result['llm_context']
    
    # Store all model responses
    all_model_responses = {}
    
    print(f"\nEvaluating {len(MODELS_TO_EVALUATE)} models on {len(queries)} queries...")
    
    # Process each model
    for model_name in MODELS_TO_EVALUATE:
        print(f"\n{'='*60}")
        print(f"PROCESSING WITH MODEL: {model_name}")
        print(f"{'='*60}")
        
        model_responses = []
        
        for idx, user_query in enumerate(queries, 1):
            print(f"\n[{idx}/{len(queries)}] {user_query}")
            
            try:
                start_time = time.time()
                
                # Build prompt
                prompt = build_prompt(user_query, llm_context, FEW_SHOT_EXAMPLES)
                
                # Call Gemini with specific model
                response_text = _call_gemini_model(prompt, model_name)
                
                response_time = time.time() - start_time
                
                # Parse response
                parsed = _parse_gemini_response(response_text)
                
                # Store result
                model_responses.append({
                    "query_number": idx,
                    "user_query": user_query,
                    "sql_query": parsed['sql_query'],
                    "visualization": parsed['visualization'],
                    "explanation": parsed['explanation'],
                    "raw_response": response_text,
                    "response_time": response_time,
                    "status": "success"
                })
                
                print(f"  ✓ Success (took {response_time:.2f}s)")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                model_responses.append({
                    "query_number": idx,
                    "user_query": user_query,
                    "status": "failed",
                    "error": str(e),
                    "response_time": 0
                })
        
        all_model_responses[model_name] = model_responses
        
        print(f"\n✓ Completed {model_name}: " +
              f"{sum(1 for r in model_responses if r['status'] == 'success')}/{len(queries)} successful")
    
    # Store in XCom
    ti.xcom_push(key='all_model_responses', value=all_model_responses)
    
    return {
        "models_processed": len(MODELS_TO_EVALUATE),
        "queries_per_model": len(queries)
    }


def _call_gemini_model(prompt: str, model_name: str) -> str:
    """Call specific Gemini model"""
    vertexai.init(project=PROJECT_ID, location=REGION)
    
    model = GenerativeModel(
        model_name,
        generation_config=GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
    )
    
    response = model.generate_content(prompt)
    return response.text


def _parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini response"""
    import re
    
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    response_text = response_text.strip()
    
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
            except:
                result = {
                    "sql_query": "SELECT * FROM dataset LIMIT 100;",
                    "visualization": {"type": "table", "title": "Results"},
                    "explanation": "Fallback response"
                }
        else:
            result = {
                "sql_query": "SELECT * FROM dataset LIMIT 100;",
                "visualization": {"type": "table", "title": "Results"},
                "explanation": "Fallback response"
            }
    
    # Ensure required fields
    if "sql_query" not in result:
        result["sql_query"] = "SELECT * FROM dataset LIMIT 100;"
    if "visualization" not in result:
        result["visualization"] = {"type": "table", "title": "Results"}
    if "explanation" not in result:
        result["explanation"] = "Generated query"
    
    return result


process_queries_task = PythonOperator(
    task_id='process_queries_with_all_models',
    python_callable=process_queries_with_all_models,
    dag=dag
)


# ========================================
# PHASE 3: MODEL EVALUATION
# ========================================

def evaluate_all_models(**context):
    """STEP 6: Evaluate all models using ModelEvaluator"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 6: EVALUATING ALL MODELS")
    print("=" * 60)
    
    # Get all model responses
    all_model_responses = ti.xcom_pull(
        task_ids='process_queries_with_all_models',
        key='all_model_responses'
    )
    
    # Load test dataframe for SQL execution testing
    loader = ModelDataLoader(BUCKET_NAME, PROJECT_ID, DATASET_ID)
    test_df = loader.query_bigquery_table(f"{DATASET_NAME}_processed", limit=1000)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(PROJECT_ID, DATASET_ID, EVALUATION_DIR)
    
    # Evaluate each model
    all_evaluation_reports = {}
    
    for model_name, responses in all_model_responses.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        metrics = evaluator.evaluate_model_responses(
            model_name=model_name,
            responses=responses,
            test_dataframe=test_df
        )
        
        # Save individual report
        report_file = evaluator.save_evaluation_report(metrics, model_name)
        
        all_evaluation_reports[model_name] = metrics
    
    # Generate comparison report
    comparison_file = evaluator.generate_comparison_report(all_evaluation_reports)
    
    # Store in XCom
    ti.xcom_push(key='evaluation_reports', value=all_evaluation_reports)
    ti.xcom_push(key='evaluation_comparison_file', value=comparison_file)
    
    print(f"\n✓ Evaluation complete for {len(all_evaluation_reports)} models")
    
    return {
        "models_evaluated": len(all_evaluation_reports),
        "comparison_file": comparison_file
    }


evaluate_models_task = PythonOperator(
    task_id='evaluate_all_models',
    python_callable=evaluate_all_models,
    dag=dag
)


# ========================================
# NEW PHASE: EXECUTE & VALIDATE BEST MODEL QUERIES
# ========================================

def execute_and_validate_best_model_queries(**context):
    """
    STEP 9.5: Execute SQL queries from best model and validate results
    This ensures the queries actually return correct answers
    """
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 9.5: EXECUTING & VALIDATING BEST MODEL QUERIES")
    print("=" * 60)
    
    # Get best model name and responses
    best_model_name = ti.xcom_pull(
        task_ids='select_best_model',
        key='best_model_name'
    )
    
    all_model_responses = ti.xcom_pull(
        task_ids='process_queries_with_all_models',
        key='all_model_responses'
    )
    
    best_model_responses = all_model_responses.get(best_model_name, [])
    
    # Initialize query executor
    executor = QueryExecutor(PROJECT_ID, DATASET_ID)
    
    # Prepare queries for execution
    queries_to_execute = []
    for response in best_model_responses:
        if response.get('status') == 'success':
            queries_to_execute.append({
                'query_number': response['query_number'],
                'user_query': response['user_query'],
                'sql_query': response['sql_query'],
                'visualization': response['visualization']
            })
    
    # Execute and validate all queries
    print(f"\nExecuting {len(queries_to_execute)} queries from {best_model_name}...")
    
    execution_results = executor.execute_batch_queries(
        queries_and_sql=queries_to_execute,
        table_name=f"{DATASET_NAME}_processed"
    )
    
    # Save execution results
    results_file = executor.save_execution_results(
        results=execution_results,
        output_dir=os.path.join(BEST_MODEL_DIR, "execution_results")
    )
    
    # Calculate accuracy metrics
    total = len(execution_results)
    executed = sum(1 for r in execution_results if r['execution_status'] == 'success')
    valid = sum(1 for r in execution_results if r['results_valid'])
    
    accuracy_metrics = {
        "total_queries": total,
        "successfully_executed": executed,
        "valid_results": valid,
        "execution_success_rate": (executed / total * 100) if total > 0 else 0,
        "result_validity_rate": (valid / executed * 100) if executed > 0 else 0,
        "overall_accuracy": (valid / total * 100) if total > 0 else 0
    }
    
    # Store in XCom
    ti.xcom_push(key='execution_results', value=execution_results)
    ti.xcom_push(key='accuracy_metrics', value=accuracy_metrics)
    ti.xcom_push(key='execution_results_file', value=results_file)
    
    print(f"\n{'='*60}")
    print(f"Execution & Validation Summary:")
    print(f"  Total Queries: {total}")
    print(f"  Successfully Executed: {executed} ({accuracy_metrics['execution_success_rate']:.1f}%)")
    print(f"  Valid Results: {valid} ({accuracy_metrics['result_validity_rate']:.1f}%)")
    print(f"  Overall Accuracy: {accuracy_metrics['overall_accuracy']:.1f}%")
    print(f"  Results File: {results_file}")
    print(f"{'='*60}\n")
    
    return accuracy_metrics


execute_validate_task = PythonOperator(
    task_id='execute_validate_queries',
    python_callable=execute_and_validate_best_model_queries,
    dag=dag
)


# ========================================
# PHASE 4: BIAS DETECTION
# ========================================

def detect_bias_in_all_models(**context):
    """STEP 7: Detect bias in all models using BiasDetector"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 7: DETECTING BIAS IN ALL MODELS")
    print("=" * 60)
    
    # Get all model responses
    all_model_responses = ti.xcom_pull(
        task_ids='process_queries_with_all_models',
        key='all_model_responses'
    )
    
    # Get dataset metadata for bias detection
    metadata = ti.xcom_pull(task_ids='generate_features', key='metadata')
    
    # Initialize bias detector
    detector = BiasDetector(BIAS_DIR)
    
    # Detect bias in each model
    all_bias_reports = {}
    
    for model_name, responses in all_model_responses.items():
        print(f"\n{'='*60}")
        print(f"BIAS DETECTION: {model_name}")
        print(f"{'='*60}")
        
        bias_report = detector.detect_all_biases(
            model_name=model_name,
            responses=responses,
            dataset_metadata=metadata
        )
        
        # Save individual report
        report_file = detector.save_bias_report(bias_report, model_name)
        
        all_bias_reports[model_name] = bias_report
    
    # Generate bias comparison
    comparison_file = detector.generate_bias_comparison(all_bias_reports)
    
    # Store in XCom
    ti.xcom_push(key='bias_reports', value=all_bias_reports)
    ti.xcom_push(key='bias_comparison_file', value=comparison_file)
    
    print(f"\n✓ Bias detection complete for {len(all_bias_reports)} models")
    
    return {
        "models_analyzed": len(all_bias_reports),
        "comparison_file": comparison_file
    }


detect_bias_task = PythonOperator(
    task_id='detect_bias_in_all_models',
    python_callable=detect_bias_in_all_models,
    dag=dag
)


# ========================================
# PHASE 5: BEST MODEL SELECTION
# ========================================

def select_best_model(**context):
    """STEP 8: Select best model using ModelSelector"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 8: SELECTING BEST MODEL")
    print("=" * 60)
    
    # Get evaluation and bias reports
    evaluation_reports = ti.xcom_pull(
        task_ids='evaluate_all_models',
        key='evaluation_reports'
    )
    
    bias_reports = ti.xcom_pull(
        task_ids='detect_bias_in_all_models',
        key='bias_reports'
    )
    
    # Initialize selector
    selector = ModelSelector(SELECTION_DIR)
    
    # Select best model
    selection_report = selector.select_best_model(
        evaluation_reports,
        bias_reports
    )
    
    # Save selection report
    report_file = selector.save_selection_report(selection_report)
    
    # Print summary
    summary = selector.generate_selection_summary(selection_report)
    print(summary)
    
    # Get best model name
    best_model_name = selection_report['best_model']['name']
    
    # Store in XCom
    ti.xcom_push(key='selection_report', value=selection_report)
    ti.xcom_push(key='best_model_name', value=best_model_name)
    ti.xcom_push(key='selection_report_file', value=report_file)
    
    print(f"\n✓ Best model selected: {best_model_name}")
    
    return {
        "best_model": best_model_name,
        "composite_score": selection_report['best_model']['composite_score'],
        "report_file": report_file
    }


select_model_task = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    dag=dag
)


# ========================================
# PHASE 6: SAVE BEST MODEL RESPONSES
# ========================================

def save_best_model_responses(**context):
    """STEP 10: Save responses from best model with execution results"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 10: SAVING BEST MODEL RESPONSES WITH EXECUTION RESULTS")
    print("=" * 60)
    
    # Get best model name
    best_model_name = ti.xcom_pull(
        task_ids='select_best_model',
        key='best_model_name'
    )
    
    # Get all model responses
    all_model_responses = ti.xcom_pull(
        task_ids='process_queries_with_all_models',
        key='all_model_responses'
    )
    
    # Get selection report
    selection_report = ti.xcom_pull(
        task_ids='select_best_model',
        key='selection_report'
    )
    
    # NEW: Get execution results
    execution_results = ti.xcom_pull(
        task_ids='execute_validate_queries',
        key='execution_results'
    )
    
    accuracy_metrics = ti.xcom_pull(
        task_ids='execute_validate_queries',
        key='accuracy_metrics'
    )
    
    # Get best model's responses
    best_model_responses = all_model_responses.get(best_model_name, [])
    
    if not best_model_responses:
        raise ValueError(f"No responses found for best model: {best_model_name}")
    
    # Merge execution results with model responses
    merged_responses = []
    for response in best_model_responses:
        query_num = response.get('query_number')
        
        # Find matching execution result
        exec_result = next(
            (r for r in execution_results if r.get('query_number') == query_num),
            None
        )
        
        # Merge
        merged = response.copy()
        if exec_result:
            merged['execution_status'] = exec_result.get('execution_status')
            merged['results_valid'] = exec_result.get('results_valid')
            merged['result_count'] = exec_result.get('result_count')
            merged['natural_language_answer'] = exec_result.get('natural_language_answer')
            merged['validation_checks'] = exec_result.get('validation_checks')
        
        merged_responses.append(merged)
    
    # Initialize response saver
    saver = ResponseSaver(PROJECT_ID, BUCKET_NAME, BEST_MODEL_DIR)
    
    save_result = saver.save_best_model_responses(
        model_name=best_model_name,
        responses=merged_responses,
        selection_report=selection_report
    )
    
    # Save accuracy metrics
    accuracy_file = Path(BEST_MODEL_DIR) / "accuracy_metrics.json"
    accuracy_file.parent.mkdir(parents=True, exist_ok=True)
    with open(accuracy_file, 'w') as f:
        json.dump(accuracy_metrics, f, indent=2)
    
    # Save deployment metadata
    metadata_file = saver.save_metadata_for_deployment(
        model_name=best_model_name,
        selection_report=selection_report
    )
    
    print(f"\n✓ Saved {save_result['files_saved']} files")
    print(f"✓ Local directory: {save_result['local_directory']}")
    print(f"✓ Deployment metadata: {metadata_file}")
    print(f"✓ Query Accuracy: {accuracy_metrics['overall_accuracy']:.1f}%")
    
    # Store in XCom
    ti.xcom_push(key='save_result', value=save_result)
    ti.xcom_push(key='deployment_metadata_file', value=metadata_file)
    
    return save_result


save_responses_task = PythonOperator(
    task_id='save_best_model_responses',
    python_callable=save_best_model_responses,
    dag=dag
)


# ========================================
# PHASE 7: FINAL SUMMARY
# ========================================

def generate_final_summary(**context):
    """STEP 11: Generate final pipeline summary"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    # Collect all results
    best_model_name = ti.xcom_pull(task_ids='select_best_model', key='best_model_name')
    save_result = ti.xcom_pull(task_ids='save_best_model_responses', key='save_result')
    selection_report = ti.xcom_pull(task_ids='select_best_model', key='selection_report')
    
    # ✅ NEW: Get accuracy metrics
    accuracy_metrics = ti.xcom_pull(
        task_ids='execute_validate_queries',
        key='accuracy_metrics'
    )
    
    summary = {
        "pipeline_completion_time": datetime.now().isoformat(),
        "dataset": DATASET_NAME,
        "models_evaluated": len(MODELS_TO_EVALUATE),
        "models_list": MODELS_TO_EVALUATE,
        "best_model": {
            "name": best_model_name,
            "score": selection_report['best_model']['composite_score'],
            "performance": selection_report['best_model']['performance_score'],
            "bias": selection_report['best_model']['bias_score']
        },
        "accuracy": accuracy_metrics,  # ✅ NEW
        "outputs": {
            "local_directory": save_result['local_directory'],
            "files_saved": save_result['files_saved'],
            "gcs_uploaded": save_result.get('gcs_uploaded', 0)
        }
    }
    
    # Save summary
    summary_file = Path(OUTPUT_BASE_DIR) / "pipeline_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"""
✓ Pipeline Complete!

Best Model Selected: {best_model_name}
  • Composite Score: {summary['best_model']['score']:.2f}/100
  • Performance Score: {summary['best_model']['performance']:.2f}/100
  • Bias Score: {summary['best_model']['bias']:.2f}/100

Query Execution Accuracy:
  • Queries Executed: {accuracy_metrics['successfully_executed']}/{accuracy_metrics['total_queries']}
  • Valid Results: {accuracy_metrics['valid_results']}/{accuracy_metrics['successfully_executed']}
  • Overall Accuracy: {accuracy_metrics['overall_accuracy']:.1f}%

Models Evaluated: {len(MODELS_TO_EVALUATE)}
  {', '.join(MODELS_TO_EVALUATE)}

Outputs:
  • Local Directory: {save_result['local_directory']}
  • Files Saved: {save_result['files_saved']}
  • Summary File: {summary_file}

{'='*60}
""")
    
    return summary


summary_task = PythonOperator(
    task_id='generate_final_summary',
    python_callable=generate_final_summary,
    dag=dag
)


# ========================================
# TASK DEPENDENCIES
# ========================================

# Phase 1: Data Preparation
load_data_task >> generate_features_task >> store_metadata_task >> read_queries_task

# Phase 2: Multi-Model Processing
read_queries_task >> process_queries_task

# Phase 3 & 4: Evaluation and Bias Detection (parallel)
process_queries_task >> [evaluate_models_task, detect_bias_task]

# Phase 5: Model Selection (waits for both evaluation and bias)
[evaluate_models_task, detect_bias_task] >> select_model_task

# Phase 6: Execute & Validate Best Model Queries (NEW)
select_model_task >> execute_validate_task

# Phase 7: Save Best Model Responses
execute_validate_task >> save_responses_task

# Phase 8: Final Summary
save_responses_task >> summary_task