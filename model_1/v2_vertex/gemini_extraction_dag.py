# dags/gemini_invoice_extraction_dag.py

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

HERE = Path(__file__).resolve()
DEFAULT_REPO_ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_REPO_ROOT))

DP_DIR = PROJECT_ROOT / "data_pipeline"
if not DP_DIR.exists():
    DP_DIR = PROJECT_ROOT / "data-pipeline"

INFERENCE_SCRIPT = Path(
    os.environ.get(
        "INFERENCE_SCRIPT",
        PROJECT_ROOT / "model_1" / "v2_vertex" / "inference.py",
    )
)
EVAL_SCRIPT = Path(
    os.environ.get(
        "EVAL_SCRIPT",
        PROJECT_ROOT / "model_1" / "v2_vertex" / "validate_llm1.py",
    )
)
DEFAULT_SAMPLE_PATH = Path(
    os.environ.get(
        "INFERENCE_SAMPLE_PATH",
        DP_DIR / "data" / "unstructured" / "invoices",
    )
)
DEFAULT_TARGET_PATH = Path(
    os.environ.get(
        "INFERENCE_TARGET_PATH",
        DEFAULT_SAMPLE_PATH / "invoice_6.pdf",
    )
)
GROUND_TRUTH_DIR = Path(
    os.environ.get(
        "GROUND_TRUTH_DIR",
        DP_DIR / "data" / "unstructured" / "invoices",
    )
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="gemini_invoice_extraction_eval",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    description="Run Gemini invoice extraction and simple evaluation",
) as dag:

    run_inference = BashOperator(
        task_id="run_inference",
        do_xcom_push=True,
        bash_command=(
            'python "{{ params.inference_script }}" '
            '--sample-path "{{ params.sample_path }}" '
            '--doc-type "{{ params.doc_type }}" '
            '--target-path "{{ params.target_path }}" '
            '--path-only'
        ),
        params={
            "inference_script": str(INFERENCE_SCRIPT),
            "sample_path": str(DEFAULT_SAMPLE_PATH),
            "doc_type": os.environ.get("INFERENCE_DOC_TYPE", "invoices"),
            "target_path": str(DEFAULT_TARGET_PATH),
        },
    )

    run_eval = BashOperator(
        task_id="run_eval",
        bash_command=(
            'python "{{ params.eval_script }}" '
            '--prediction-file "{{ ti.xcom_pull(task_ids=\'run_inference\') }}" '
            '--ground-truth-dir "{{ params.ground_truth_dir }}"'
        ),
        params={
            "eval_script": str(EVAL_SCRIPT),
            "ground_truth_dir": str(GROUND_TRUTH_DIR),
        },
    )

    run_inference >> run_eval
