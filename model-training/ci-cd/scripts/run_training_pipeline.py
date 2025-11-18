#!/usr/bin/env python3
"""
Run Model Training Pipeline Script
Triggers the Airflow DAG or runs training pipeline directly

Usage:
    python run_training_pipeline.py [--trigger-dag|--direct]
"""

import sys
import os
import argparse
from pathlib import Path
import time
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

def trigger_airflow_dag(dag_id: str = "model_pipeline_with_evaluation"):
    """
    Trigger Airflow DAG via API
    
    Args:
        dag_id: DAG ID to trigger
    """
    try:
        from airflow.api.client.local_client import Client
        from airflow.models import DagBag
        
        # Initialize Airflow client
        client = Client(None, None)
        
        # Trigger DAG
        print(f"Triggering Airflow DAG: {dag_id}")
        result = client.trigger_dag(dag_id)
        
        print(f"✓ DAG triggered successfully")
        print(f"  Run ID: {result.run_id}")
        
        return result.run_id
        
    except Exception as e:
        print(f"✗ Failed to trigger Airflow DAG: {e}")
        print("  Note: This requires Airflow to be running")
        raise

def run_training_directly():
    """
    Run training pipeline directly (without Airflow)
    This is a simplified version for CI/CD
    """
    print("\n" + "=" * 70)
    print("RUNNING MODEL TRAINING PIPELINE (DIRECT MODE)")
    print("=" * 70)
    
    # Import required modules
    try:
        from data_loader import ModelDataLoader
        from feature_engineering import FeatureEngineer
        from metadata_manager import MetadataManager
        from prompts import build_prompt, FEW_SHOT_EXAMPLES
        from model_evaluator import ModelEvaluator
        from bias_detector import BiasDetector
        from model_selector import ModelSelector
        from response_saver import ResponseSaver
        from query_executor import QueryExecutor
        
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
        
    except ImportError as e:
        print(f"✗ Failed to import required modules: {e}")
        print("  Make sure all dependencies are installed")
        raise
    
    # Load configuration
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "ci_cd_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gcp_config = config['gcp']
    training_config = config['training']
    
    PROJECT_ID = gcp_config['project_id']
    BUCKET_NAME = gcp_config['bucket_name']
    DATASET_ID = gcp_config['dataset_id']
    REGION = gcp_config['region']
    DATASET_NAME = training_config['dataset_name']
    MODELS_TO_EVALUATE = training_config['models_to_evaluate']
    
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=REGION)
    
    print("\nStep 1: Loading data to BigQuery...")
    loader = ModelDataLoader(BUCKET_NAME, PROJECT_ID, DATASET_ID)
    table_name = f"{DATASET_NAME}_processed"
    
    if not loader.table_exists(table_name):
        df = loader.load_processed_data_from_gcs(DATASET_NAME, stage='validated')
        loader.load_to_bigquery(df, DATASET_NAME, table_suffix="_processed")
        print(f"  ✓ Loaded {len(df):,} rows")
    else:
        print(f"  ✓ Table {table_name} already exists")
    
    print("\nStep 2: Generating features and metadata...")
    df = loader.query_bigquery_table(table_name, limit=10000)
    schema = loader.load_dataset_profile_from_gcs(DATASET_NAME)
    engineer = FeatureEngineer(df, schema, DATASET_NAME)
    metadata = engineer.generate_metadata()
    llm_context = engineer.create_llm_context()
    print("  ✓ Features and metadata generated")
    
    print("\nStep 3: Storing metadata in BigQuery...")
    manager = MetadataManager(PROJECT_ID, DATASET_ID)
    manager.store_metadata(DATASET_NAME, metadata, llm_context)
    print("  ✓ Metadata stored")
    
    print("\nStep 4: Reading user queries...")
    queries_file = Path(__file__).parent.parent.parent / training_config['queries_file']
    with open(queries_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    print(f"  ✓ Read {len(queries)} queries")
    
    print("\nStep 5: Processing queries with all models...")
    all_model_responses = {}
    
    for model_name in MODELS_TO_EVALUATE:
        print(f"\n  Processing with {model_name}...")
        model_responses = []
        
        model = GenerativeModel(
            model_name,
            generation_config=GenerationConfig(
                temperature=0.2,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
        )
        
        for idx, user_query in enumerate(queries, 1):
            print(f"    [{idx}/{len(queries)}] {user_query[:50]}...")
            
            try:
                prompt = build_prompt(user_query, llm_context, FEW_SHOT_EXAMPLES)
                response = model.generate_content(prompt)
                
                # Parse response (simplified)
                import json
                import re
                response_text = response.text
                response_text = re.sub(r'```json\s*', '', response_text)
                response_text = re.sub(r'```\s*', '', response_text)
                
                try:
                    parsed = json.loads(response_text)
                except:
                    parsed = {
                        "sql_query": "SELECT * FROM dataset LIMIT 100;",
                        "visualization": {"type": "table", "title": "Results"},
                        "explanation": "Generated query"
                    }
                
                model_responses.append({
                    "query_number": idx,
                    "user_query": user_query,
                    "sql_query": parsed.get('sql_query', ''),
                    "visualization": parsed.get('visualization', {}),
                    "explanation": parsed.get('explanation', ''),
                    "status": "success"
                })
                
            except Exception as e:
                print(f"      ✗ Error: {e}")
                model_responses.append({
                    "query_number": idx,
                    "user_query": user_query,
                    "status": "failed",
                    "error": str(e)
                })
        
        all_model_responses[model_name] = model_responses
        print(f"  ✓ Completed {model_name}")
    
    print("\nStep 6: Evaluating all models...")
    evaluator = ModelEvaluator(PROJECT_ID, DATASET_ID, config['paths']['evaluation_output'])
    test_df = loader.query_bigquery_table(table_name, limit=1000)
    
    all_evaluation_reports = {}
    for model_name, responses in all_model_responses.items():
        metrics = evaluator.evaluate_model_responses(model_name, responses, test_df)
        evaluator.save_evaluation_report(metrics, model_name)
        all_evaluation_reports[model_name] = metrics
        print(f"  ✓ Evaluated {model_name}")
    
    print("\nStep 7: Detecting bias in all models...")
    detector = BiasDetector(config['paths']['bias_output'])
    all_bias_reports = {}
    
    for model_name, responses in all_model_responses.items():
        bias_report = detector.detect_all_biases(model_name, responses, metadata)
        detector.save_bias_report(bias_report, model_name)
        all_bias_reports[model_name] = bias_report
        print(f"  ✓ Bias detection for {model_name}")
    
    print("\nStep 8: Selecting best model...")
    selector = ModelSelector(config['paths']['selection_output'])
    selection_report = selector.select_best_model(all_evaluation_reports, all_bias_reports)
    selector.save_selection_report(selection_report)
    best_model_name = selection_report['best_model']['name']
    print(f"  ✓ Best model selected: {best_model_name}")
    
    print("\nStep 9: Executing and validating best model queries...")
    executor = QueryExecutor(PROJECT_ID, DATASET_ID)
    best_model_responses = all_model_responses[best_model_name]
    
    execution_results = []
    for response in best_model_responses:
        if response.get('status') == 'success':
            result = executor.execute_and_validate(
                user_query=response['user_query'],
                sql_query=response['sql_query'],
                table_name=table_name,
                visualization=response['visualization']
            )
            result['query_number'] = response['query_number']
            execution_results.append(result)
    
    print(f"  ✓ Executed {len(execution_results)} queries")
    
    print("\nStep 10: Saving best model responses...")
    saver = ResponseSaver(PROJECT_ID, BUCKET_NAME, config['paths']['best_model_output'])
    save_result = saver.save_best_model_responses(
        model_name=best_model_name,
        responses=best_model_responses,
        selection_report=selection_report
    )
    print(f"  ✓ Saved {save_result['files_saved']} files")
    
    # Generate summary
    import json
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
        "outputs": {
            "local_directory": save_result['local_directory'],
            "files_saved": save_result['files_saved']
        }
    }
    
    summary_file = Path(config['paths']['model_training_output']) / "pipeline_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Pipeline complete! Summary saved to: {summary_file}")
    print("=" * 70 + "\n")
    
    return summary_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model training pipeline')
    parser.add_argument('--trigger-dag', action='store_true', help='Trigger Airflow DAG')
    parser.add_argument('--direct', action='store_true', help='Run pipeline directly')
    
    args = parser.parse_args()
    
    if args.trigger_dag:
        trigger_airflow_dag()
    elif args.direct:
        run_training_directly()
    else:
        # Default: try direct mode
        print("Running in direct mode (use --trigger-dag for Airflow)")
        run_training_directly()

