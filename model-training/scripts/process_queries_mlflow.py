import sys
sys.path.insert(0, '/opt/airflow/model-training/scripts')

from mlflow_llm2_tracker import LLM2Tracker
import time


def process_all_queries_with_mlflow(**context):
    """STEP 5: Process queries with MLflow tracking"""
    ti = context['ti']
    
    print("\n" + "=" * 60)
    print("STEP 5: PROCESSING QUERIES WITH GEMINI + MLFLOW")
    print("=" * 60)
    
    queries = ti.xcom_pull(task_ids='read_user_queries', key='user_queries')
    
    from metadata_manager import MetadataManager
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
            from prompts import build_prompt, FEW_SHOT_EXAMPLES
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
