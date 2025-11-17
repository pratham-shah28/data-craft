"""
Complete Model Pipeline DAG - End-to-End
Reuses all existing functions from scripts/
✅ UPDATED for unified Docker setup
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import json
import os
import sys
from pathlib import Path

# ✅ UPDATED: Use unified Docker paths
sys.path.insert(0, '/opt/airflow/model-training/scripts')
sys.path.insert(0, '/opt/airflow/shared')

# Import ALL existing modules
from data_loader import ModelDataLoader
from feature_engineering import FeatureEngineer
from metadata_manager import MetadataManager
from prompts import build_prompt, FEW_SHOT_EXAMPLES
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# MLflow tracking
from mlflow_llm2_tracker import LLM2Tracker
import time

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "datacraft-data-pipeline")
REGION = Variable.get("REGION", default_var="us-central1")
BUCKET_NAME = Variable.get("BUCKET_NAME", default_var="isha-retail-data")
DATASET_ID = Variable.get("BQ_DATASET", default_var="datacraft_ml")
MODEL_NAME = Variable.get("GEMINI_MODEL", default_var="gemini-2.5-pro")

# ✅ UPDATED: Paths for unified Docker
QUERIES_FILE = "/opt/airflow/model-training/data/user_queries.txt"
OUTPUT_BASE_DIR = "/opt/airflow/outputs/model-training"
DATASET_NAME = "orders"

# ✅ UPDATED: Default args with new email
default_args = {
    'owner': 'datacraft-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['mlops0242@gmail.com'],  # UPDATED
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# DAG definition
dag = DAG(
    'model_pipeline_complete',
    default_args=default_args,
    description='Complete end-to-end pipeline: Load → Feature → Metadata → Model → Results',
    schedule_interval=None,
    catchup=False,
    tags=['model', 'gemini', 'end-to-end']
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
    
    # Use existing ModelDataLoader class
    loader = ModelDataLoader(
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID
    )
    
    table_name = f"{DATASET_NAME}_processed"
    
    # Check if exists
    if loader.table_exists(table_name):
        print(f"✓ Table {table_name} already exists")
        table_info = loader.get_table_info(table_name)
        
        ti.xcom_push(key='row_count', value=table_info.get('num_rows', 0))
        
        return {
            "status": "exists",
            "table": table_name,
            "rows": table_info.get('num_rows', 0)
        }
    
    # Load from GCS using existing method
    df = loader.load_processed_data_from_gcs(DATASET_NAME, stage='validated')
    
    # Load to BigQuery using existing method
    table_id = loader.load_to_bigquery(df, DATASET_NAME, table_suffix="_processed")
    
    ti.xcom_push(key='row_count', value=len(df))
    
    print(f"\n✓ Data loaded: {len(df):,} rows")
    
    return {
        "status": "loaded",
        "table": table_id,
        "rows": len(df)
    }


load_data_task = PythonOperator(
    task_id='load_data_to_bigquery',
    python_callable=load_data_to_bigquery,
    dag=dag
)


# ========================================
# PHASE 2: FEATURE ENGINEERING
# ========================================

def generate_features(**context):
    """STEP 2: Generate features using existing FeatureEngineer class"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 2: GENERATING FEATURES & METADATA")
    print("=" * 60)
    
    # Use existing ModelDataLoader
    loader = ModelDataLoader(
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID
    )
    
    # Load data sample
    df = loader.query_bigquery_table(f"{DATASET_NAME}_processed", limit=10000)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Load schema using existing method
    schema = loader.load_dataset_profile_from_gcs(DATASET_NAME)
    
    # Use existing FeatureEngineer class
    engineer = FeatureEngineer(df, schema, DATASET_NAME)
    
    # Generate metadata using existing method
    metadata = engineer.generate_metadata()
    print(f"✓ Generated metadata: {len(metadata['columns'])} columns")
    
    # Generate LLM context using existing method
    llm_context = engineer.create_llm_context()
    print(f"✓ Generated LLM context: {len(llm_context)} chars")
    
    # Get summary using existing method
    summary = engineer.get_feature_summary()
    
    # Store in XCom
    ti.xcom_push(key='metadata', value=metadata)
    ti.xcom_push(key='llm_context', value=llm_context)
    ti.xcom_push(key='feature_summary', value=summary)
    
    return summary


generate_features_task = PythonOperator(
    task_id='generate_features',
    python_callable=generate_features,
    dag=dag
)


# ========================================
# PHASE 3: METADATA STORAGE
# ========================================

def store_metadata_in_bigquery(**context):
    """STEP 3: Store metadata using existing MetadataManager class"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 3: STORING METADATA IN BIGQUERY")
    print("=" * 60)
    
    # Get from XCom
    metadata = ti.xcom_pull(task_ids='generate_features', key='metadata')
    llm_context = ti.xcom_pull(task_ids='generate_features', key='llm_context')
    
    # Use existing MetadataManager class
    manager = MetadataManager(PROJECT_ID, DATASET_ID)
    
    # Store using existing method
    manager.store_metadata(DATASET_NAME, metadata, llm_context)
    
    # Verify using existing method
    retrieved = manager.get_metadata(DATASET_NAME)
    
    if retrieved:
        print(f"✓ Metadata stored and verified")
        print(f"  Dataset: {retrieved['dataset_name']}")
        print(f"  Rows: {retrieved['row_count']:,}")
        
        return {
            "status": "success",
            "dataset": DATASET_NAME
        }
    else:
        raise ValueError("Failed to verify metadata!")


store_metadata_task = PythonOperator(
    task_id='store_metadata',
    python_callable=store_metadata_in_bigquery,
    dag=dag
)


# ========================================
# PHASE 4: QUERY PROCESSING
# ========================================

def read_user_queries(**context):
    """STEP 4: Read queries from file"""
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


read_queries_task = PythonOperator(
    task_id='read_user_queries',
    python_callable=read_user_queries,
    dag=dag
)


def process_all_queries(**context):
    """STEP 5: Process queries with MLflow tracking"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 5: PROCESSING QUERIES WITH GEMINI + MLFLOW")
    print("=" * 60)
    
    queries = ti.xcom_pull(task_ids='read_user_queries', key='user_queries')
    
    manager = MetadataManager(PROJECT_ID, DATASET_ID)
    metadata_result = manager.get_metadata(DATASET_NAME)
    
    if not metadata_result:
        raise ValueError(f"No metadata found for: {DATASET_NAME}")
    
    llm_context = metadata_result['llm_context']
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 2048
    }
    
    mlflow_tracker = LLM2Tracker(
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        generation_config=generation_config
    )
    
    mlflow_tracker.start_run()
    
    all_results = []
    
    print(f"\nProcessing {len(queries)} queries...")
    
    for idx, user_query in enumerate(queries, 1):
        print(f"\n[{idx}/{len(queries)}] {user_query}")
        
        query_start_time = time.time()
        
        try:
            prompt = build_prompt(user_query, llm_context, FEW_SHOT_EXAMPLES)
            
            response_text = _call_gemini(prompt)
            parsed = _parse_gemini_response(response_text)
            
            query_folder = _save_query_result(
                query_number=idx,
                user_query=user_query,
                parsed_response=parsed,
                raw_response=response_text
            )
            
            query_time = time.time() - query_start_time
            mlflow_tracker.log_query_processing(idx, success=True, response_time=query_time)
            
            print(f"  ✓ Saved to: {query_folder} ({query_time:.2f}s)")
            
            all_results.append({
                "query_number": idx,
                "user_query": user_query,
                "sql_query": parsed['sql_query'],
                "visualization": parsed['visualization'],
                "explanation": parsed['explanation'],
                "output_folder": query_folder,
                "response_time": query_time,
                "status": "success"
            })
            
        except Exception as e:
            query_time = time.time() - query_start_time
            mlflow_tracker.log_query_processing(idx, success=False, response_time=query_time)
            
            print(f"  ✗ Error: {str(e)}")
            all_results.append({
                "query_number": idx,
                "user_query": user_query,
                "status": "failed",
                "error": str(e),
                "response_time": query_time
            })
    
    summary_file = os.path.join(OUTPUT_BASE_DIR, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "total_queries": len(queries),
            "successful": sum(1 for r in all_results if r['status'] == 'success'),
            "failed": sum(1 for r in all_results if r['status'] == 'failed'),
            "timestamp": datetime.now().isoformat(),
            "results": all_results
        }, f, indent=2)
    
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    failed_count = len(queries) - success_count
    
    mlflow_tracker.log_final_metrics(
        total_queries=len(queries),
        successful_queries=success_count,
        failed_queries=failed_count
    )
    
    mlflow_tracker.log_artifacts(OUTPUT_BASE_DIR)
    
    mlflow_tracker.end_run(status="FINISHED")
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: {success_count}/{len(queries)}")
    print(f"MLflow tracking complete")
    print(f"{'='*60}")
    
    ti.xcom_push(key='all_results', value=all_results)
    ti.xcom_push(key='summary_file', value=summary_file)
    
    return {
        "total": len(queries),
        "successful": success_count,
        "failed": failed_count
    }


# ========================================
# HELPER FUNCTIONS
# ========================================

def _call_gemini(prompt: str) -> str:
    """Call Gemini API"""
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=REGION)
    
    # Create model
    model = GenerativeModel(
        MODEL_NAME,
        generation_config=GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
    )
    
    # Generate
    response = model.generate_content(prompt)
    return response.text


def _parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini response"""
    import re
    
    # Clean
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    response_text = response_text.strip()
    
    # Parse
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
                    "explanation": "Fallback"
                }
        else:
            result = {
                "sql_query": "SELECT * FROM dataset LIMIT 100;",
                "visualization": {"type": "table", "title": "Results"},
                "explanation": "Fallback"
            }
    
    # Ensure required fields
    if "sql_query" not in result:
        result["sql_query"] = "SELECT * FROM dataset LIMIT 100;"
    if "visualization" not in result:
        result["visualization"] = {"type": "table", "title": "Results"}
    if "explanation" not in result:
        result["explanation"] = "Generated query"
    
    return result


def _save_query_result(
    query_number: int,
    user_query: str,
    parsed_response: dict,
    raw_response: str
) -> str:
    """Save to folder"""
    query_folder = os.path.join(OUTPUT_BASE_DIR, f"query_{query_number:03d}")
    os.makedirs(query_folder, exist_ok=True)
    
    # Save files
    with open(os.path.join(query_folder, "user_query.txt"), 'w') as f:
        f.write(user_query)
    
    with open(os.path.join(query_folder, "sql_query.sql"), 'w') as f:
        f.write(parsed_response['sql_query'])
    
    with open(os.path.join(query_folder, "visualization.json"), 'w') as f:
        json.dump(parsed_response['visualization'], f, indent=2)
    
    with open(os.path.join(query_folder, "explanation.txt"), 'w') as f:
        f.write(parsed_response['explanation'])
    
    with open(os.path.join(query_folder, "raw_response.txt"), 'w') as f:
        f.write(raw_response)
    
    with open(os.path.join(query_folder, "result.json"), 'w') as f:
        json.dump({
            "query_number": query_number,
            "user_query": user_query,
            "sql_query": parsed_response['sql_query'],
            "visualization": parsed_response['visualization'],
            "explanation": parsed_response['explanation'],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    return query_folder


process_queries_task = PythonOperator(
    task_id='process_queries',
    python_callable=process_all_queries,
    dag=dag
)


# ========================================
# PHASE 5: UPLOAD TO GCS
# ========================================

def upload_results_to_gcs(**context):
    """STEP 6: Upload to GCS"""
    from google.cloud import storage
    
    print("\n" + "=" * 60)
    print("STEP 6: UPLOADING TO GCS")
    print("=" * 60)
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    uploaded = 0
    
    for root, dirs, files in os.walk(OUTPUT_BASE_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, OUTPUT_BASE_DIR)
            gcs_path = f"model_outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{relative_path}"
            
            try:
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                uploaded += 1
            except Exception as e:
                print(f"  ✗ {local_path}: {str(e)}")
    
    print(f"\n✓ Uploaded {uploaded} files")
    
    return {"uploaded": uploaded}


upload_gcs_task = PythonOperator(
    task_id='upload_to_gcs',
    python_callable=upload_results_to_gcs,
    dag=dag
)


# ========================================
# Task Dependencies
# ========================================

(
    load_data_task >> 
    generate_features_task >> 
    store_metadata_task >> 
    read_queries_task >> 
    process_queries_task >> 
    upload_gcs_task
)