from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.operators.email import EmailOperator
from airflow.operators.python import BranchPythonOperator

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from data_acquisition import acquire_data
from data_validation import DataValidator
from data_cleaning import DataCleaner
from bias_detection import BiasDetector
from upload_to_gcp import upload_to_gcs
from utils import load_config, setup_logging

# Load configuration
config = load_config()

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['ishas2505@gmail.com','sakseneshivi@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

def run_acquisition(**context):
    """Task 1: Data Acquisition"""
    logger = setup_logging("airflow_acquisition")
    
    # Get parameters from DAG run config
    source_file = context['dag_run'].conf.get('source_file')
    
    logger.info(f"Starting data acquisition from: {source_file or 'auto-detect'}")
    
    try:
        result = acquire_data(source_file=source_file)
        
        # Push dataset name to XCom for downstream tasks
        dataset_name = result['dataset_name']
        context['ti'].xcom_push(key='dataset_name', value=dataset_name)
        
        logger.info(f"✓ Acquisition complete. Dataset: {dataset_name}")
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "file_path": result['file_path']
        }
        
    except Exception as e:
        logger.error(f"Acquisition failed: {str(e)}")
        raise

def run_validation(**context):
    """Task 2: Data Validation"""
    logger = setup_logging("airflow_validation")
    
    # Get dataset name from previous task
    dataset_name = context['ti'].xcom_pull(key='dataset_name', task_ids='acquire_data')
    
    if not dataset_name:
        raise ValueError("Dataset name not found from acquisition task")
    
    logger.info(f"Starting validation for dataset: {dataset_name}")
    
    try:
        validator = DataValidator(dataset_name)
        report = validator.validate()
        
        # Check if validation passed
        if not report['overall_valid']:
            logger.warning("⚠ Validation failed — triggering anomaly alert email")
            # Trigger email task manually
            context['ti'].xcom_push(key='anomaly_detected', value=True)
        else:
            logger.info("✓ Validation passed")
        
        return {
            "status": "success",
            "overall_valid": report['overall_valid'],
            "dataset_name": dataset_name
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise

def run_cleaning(**context):
    """Task 3: Data Cleaning"""
    logger = setup_logging("airflow_cleaning")
    
    # Get dataset name from XCom
    dataset_name = context['ti'].xcom_pull(key='dataset_name', task_ids='acquire_data')
    
    if not dataset_name:
        raise ValueError("Dataset name not found from acquisition task")
    
    logger.info(f"Starting cleaning for dataset: {dataset_name}")

    try:
        cleaner = DataCleaner(dataset_name)
        cleaned_df = cleaner.clean_data()
        null_percentage = (cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns))) * 100
        if null_percentage > 0.15:  #  threshold
            logger.warning(f"⚠ High missing values detected: {null_percentage:.2f}% — triggering email alert")
            context['ti'].xcom_push(key='anomaly_detected', value=True)
        logger.info(f"✓ Cleaning complete. Final shape: {cleaned_df.shape}")
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "final_rows": len(cleaned_df),
            "final_columns": len(cleaned_df.columns)
        }
        
    except Exception as e:
        logger.error(f"Cleaning failed: {str(e)}")
        raise

def run_bias_detection(**context):
    """Task 4: Bias Detection"""
    logger = setup_logging("airflow_bias")
    
    # Get dataset name from XCom
    dataset_name = context['ti'].xcom_pull(key='dataset_name', task_ids='acquire_data')
    
    if not dataset_name:
        raise ValueError("Dataset name not found from acquisition task")
    
    logger.info(f"Starting bias detection for dataset: {dataset_name}")
    
    try:
        detector = BiasDetector(dataset_name)
        report = detector.detect_bias()
        
        # Log summary
        if 'summary' in report and 'bias_flags' in report['summary']:
            bias_flags = report['summary']['bias_flags']
            total_tests = report['summary']['total_tests']
            logger.info(f"✓ Bias detection complete: {bias_flags} flags in {total_tests} tests")
        else:
            logger.info("✓ Bias detection complete")
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "bias_flags": report.get('summary', {}).get('bias_flags', 0)
        }
        
    except Exception as e:
        logger.error(f"Bias detection failed: {str(e)}")
        raise

def send_anomaly_email_if_needed(**context):
    """Send email only if anomaly detected in validation or cleaning"""
    logger = setup_logging("conditional_email")
    
    # Check if any anomaly was detected
    anomaly_detected = False
    for task_id in ['validate_data', 'clean_data']:
        if context['ti'].xcom_pull(key='anomaly_detected', task_ids=task_id):
            anomaly_detected = True
            break
    
    if anomaly_detected:
        logger.warning("⚠ Anomaly detected - sending alert email")
        try:
            from airflow.utils.email import send_email
            send_email(
                to='ishas2505@gmail.com',
                subject='⚠ Data Anomaly Detected in ML Pipeline',
                html_content=f"""
                <h3>Data Anomaly Alert</h3>
                <p>An anomaly was detected during data validation or cleaning.</p>
                <p>Please check the Airflow logs for details.</p>
                <p><strong>DAG Run:</strong> {context['dag_run'].run_id}</p>
                <p><strong>Execution Date:</strong> {context['execution_date']}</p>
                """,
            )
            logger.info("✓ Alert email sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            # Don't fail the task if email fails - just log it
    else:
        logger.info("✓ No anomalies detected - skipping email")

def run_gcp_upload(**context):
    """Task 5: Upload to GCS"""
    logger = setup_logging("airflow_gcp")
    
    # Get dataset name from XCom
    dataset_name = context['ti'].xcom_pull(key='dataset_name', task_ids='acquire_data')
    
    if not dataset_name:
        raise ValueError("Dataset name not found from acquisition task")
    
    # Get upload preferences from DAG config
    include_raw = context['dag_run'].conf.get('include_raw', False)
    include_reports = context['dag_run'].conf.get('include_reports', True)
    
    logger.info(f"Starting GCP upload for dataset: {dataset_name}")
    
    try:
        result = upload_to_gcs(
            dataset_name=dataset_name,
            include_raw=include_raw,
            include_reports=include_reports
        )
        
        logger.info(f"✓ GCP upload complete: {result['files_uploaded']} files")
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "files_uploaded": result['files_uploaded'],
            "total_size": result['total_size_formatted']
        }
        
    except Exception as e:
        logger.error(f"GCP upload failed: {str(e)}")
        raise

def generate_summary_report(**context):
    """Task 6: Generate pipeline summary report"""
    logger = setup_logging("airflow_summary")
    
    # Gather results from all tasks
    dataset_name = context['ti'].xcom_pull(key='dataset_name', task_ids='acquire_data')
    acquisition_result = context['ti'].xcom_pull(task_ids='acquire_data')
    validation_result = context['ti'].xcom_pull(task_ids='validate_data')
    cleaning_result = context['ti'].xcom_pull(task_ids='clean_data')
    bias_result = context['ti'].xcom_pull(task_ids='detect_bias')
    gcp_result = context['ti'].xcom_pull(task_ids='upload_to_gcs')
    
    summary = {
        "pipeline_run": {
            "dataset_name": dataset_name,
            "execution_date": str(context['execution_date']),
            "dag_run_id": context['dag_run'].run_id,
        },
        "results": {
            "acquisition": acquisition_result,
            "validation": validation_result,
            "cleaning": cleaning_result,
            "bias_detection": bias_result,
            "gcp_upload": gcp_result
        }
    }
    
    # Save summary report
    import json
    summary_path = Path("reports") / f"{dataset_name}_pipeline_summary_{context['ds']}.json"
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Pipeline summary saved to {summary_path}")
    
    # Log final summary
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Validation: {'✓ Passed' if validation_result.get('overall_valid') else '⚠ Failed'}")
    logger.info(f"Cleaning: {cleaning_result.get('final_rows')} rows, {cleaning_result.get('final_columns')} columns")
    logger.info(f"Bias Detection: {bias_result.get('bias_flags', 0)} flags")
    logger.info(f"GCP Upload: {gcp_result.get('files_uploaded')} files ({gcp_result.get('total_size')})")
    logger.info("=" * 60)
    
    return summary

def send_success_email(**context):
    """Send success email after pipeline completion"""
    logger = setup_logging("success_email")
    
    dataset_name = context['ti'].xcom_pull(key='dataset_name', task_ids='acquire_data')
    
    try:
        from airflow.utils.email import send_email
        send_email(
            to='ishas2505@gmail.com',
            subject='✅ ML Data Pipeline Completed Successfully',
            html_content=f"""
            <h3>ML Data Pipeline Completed Successfully</h3>
            <p>All tasks in the DAG <b>ml_data_pipeline</b> ran successfully.</p>
            <p><strong>Dataset:</strong> {dataset_name}</p>
            <p><strong>DAG Run:</strong> {context['dag_run'].run_id}</p>
            <p><strong>Execution Date:</strong> {context['execution_date']}</p>
            <p>Summary report generated. Check Airflow logs or the 'reports/' folder for details.</p>
            """,
        )
        logger.info("✓ Success email sent")
    except Exception as e:
        logger.error(f"Failed to send success email: {str(e)}")
        # Don't fail the task - pipeline already succeeded

# Create the DAG
with DAG(
    'ml_data_pipeline',
    default_args=default_args,
    description='End-to-end ML data pipeline: acquisition → validation → cleaning → bias detection → GCP upload',
    schedule_interval=None,  # Triggered manually
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'data-pipeline', 'gcp', 'bias-detection'],
    max_active_runs=1,
) as dag:
    
    # Task 1: Data Acquisition
    acquire_task = PythonOperator(
        task_id='acquire_data',
        python_callable=run_acquisition,
        provide_context=True,
    )
    
    # Task 2: Data Validation
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=run_validation,
        provide_context=True,
    )
    
    # Task 3: Data Cleaning
    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=run_cleaning,
        provide_context=True,
    )
    
    # Task 4: Bias Detection
    bias_task = PythonOperator(
        task_id='detect_bias',
        python_callable=run_bias_detection,
        provide_context=True,
    )
    # Task 5: Check and send anomaly email if needed
    check_and_alert_task = PythonOperator(
        task_id='check_and_alert',
        python_callable=send_anomaly_email_if_needed,
        provide_context=True,
    )
    # Task 5: DVC Tracking
    dvc_add_task = BashOperator(
        task_id='dvc_add_data',
        bash_command='''
        cd {{ params.project_root }} && \
        dvc add data/processed/*.csv data/validated/*.csv && \
        echo "DVC files added successfully"
        ''',
        params={'project_root': str(Path(__file__).parent.parent)},
    )
    
    dvc_push_task = BashOperator(
        task_id='dvc_push',
        bash_command='''
        cd {{ params.project_root }} && \
        dvc push && \
        echo "DVC push completed successfully"
        ''',
        params={'project_root': str(Path(__file__).parent.parent)},
    )
    
    # Task 6: GCP Upload
    gcp_upload_task = PythonOperator(
        task_id='upload_to_gcs',
        python_callable=run_gcp_upload,
        provide_context=True,
    )
    
    success_email_task = PythonOperator(
    task_id='success_email',
    python_callable=send_success_email,
    provide_context=True,
    )
    
    # Task 7: Git Commit (DVC metadata)
    git_commit_task = BashOperator(
        task_id='git_commit_dvc',
        bash_command='''
        cd {{ params.project_root }} && \
        git add data/*.dvc .dvc/config config/dataset_profiles/*.json && \
        git commit -m "Pipeline run: {{ params.dataset_name }} - {{ ds }}" || \
        echo "No changes to commit"
        ''',
        params={
            'project_root': str(Path(__file__).parent.parent),
            'dataset_name': '{{ ti.xcom_pull(key="dataset_name", task_ids="acquire_data") }}'
        },
    )
    
    # Task 8: Generate Summary Report
    summary_task = PythonOperator(
        task_id='generate_summary',
        python_callable=generate_summary_report,
        provide_context=True,
    )
    
    # Define task dependencies
    # Linear flow with parallel branches
    # Define task dependencies
    # Define task dependencies - Clean linear flow
    acquire_task >> validate_task >> clean_task >> bias_task
    
    # After bias detection, check for anomalies and send email if needed
    bias_task >> check_and_alert_task
    
    # Then proceed with parallel DVC and GCP paths
    check_and_alert_task >> [dvc_add_task, gcp_upload_task]
    
    # DVC path
    dvc_add_task >> dvc_push_task >> git_commit_task
    
    # Both paths converge at summary
    [git_commit_task, gcp_upload_task] >> summary_task >> success_email_task