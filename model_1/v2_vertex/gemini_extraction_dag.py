# dags/gemini_invoice_extraction_dag.py

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# TODO: update these to match your actual paths inside the Airflow container/VM
INFERENCE_SCRIPT = "/Users/prathamshah/Desktop/projects/mlops project/mlops-project/model_1/v2_vertex/inference.py"
EVAL_SCRIPT      = "/Users/prathamshah/Desktop/projects/mlops project/mlops-project/model_1/v2_vertex/validate_llm1.py"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="gemini_invoice_extraction_eval",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # run manually
    catchup=False,
    description="Run Gemini invoice extraction and simple evaluation",
) as dag:

    run_inference = BashOperator(
        task_id="run_inference",
        bash_command=(
            'python "{{ params.inference_script }}" '
            # optionally override defaults here:
            # "--sample-path /path/to/invoices "
            # "--doc-type invoices "
            # "--target-path /path/to/invoices/invoice_6.pdf "
        ),
        params={"inference_script": INFERENCE_SCRIPT},
    )

    run_eval = BashOperator(
        task_id="run_eval",
        bash_command='python "{{ params.eval_script }}"',
        params={"eval_script": EVAL_SCRIPT},
    )

    # dependency: first run inference, then evaluate outputs
    run_inference >> run_eval
