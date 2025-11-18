# Data Pipeline

Here we implement a dual-pipeline architecture to handle both unstructured and structured data. The first pipeline focuses on transforming document-based data, such as PDFs, into a machine-readable format. The second pipeline follows a traditional ETL flow for structured datasets, where data is cleaned, validated, version-controlled, and prepared for downstream analytics.

In the unstructured pipeline, we leverage an LLM as part of the data processing workflow. We first fetch documents from external sources using a script, then convert each PDF page into a base64-encoded image representation using a custom preprocessing script. This prepares the data to be directly consumed by vision-enabled LLMs and allow them to produce structured data which can go into our traditional data-pipeline. We have also included dedicated unit tests to ensure the correctness of this conversion step and the robustness of the preprocessing stage.

To complement the LLM based data processing, the second phase follows a more traditional ETL workflow for structured datasets. After the data is acquired, the pipeline performs schema-based validation, cleaning, anomaly and bias detection, and then pushes each processed version into DVC for full reproducibility. The DAG above illustrates this end-to-end flow orchestrated in Apache Airflow, where each transformation step is modular, observable, and fault-tolerant. Once the data has been validated and version-controlled, it is pushed to cloud storage and a final report is generated summarizing the pipeline execution outcomes. Together, these two pipelines enable a unified MLOps system capable of handling real-world retail data in both structured and document-based formats, ensuring reliable, scalable, and automation-ready data delivery for downstream analytics and machine learning tasks.

---

##  Project Structure

```plaintext
data-pipeline/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt            # or pyproject.toml
├── .env.example                # put non-secret defaults here
├── .gitignore
├── dvc.yaml                    # if using DVC pipelines
├── .dvc/                       # DVC internal (auto-created)
│
├── config/                     # Configuration files & schema profiles
│   └── dataset_profiles/
│
├── dags/                       # Airflow DAGs
│   └── retail_data_pipeline.py
│
├── data/
│   ├── raw/                    # Source datasets (canonical CSV)
│   ├── processed/              # Validated pass-through CSV + reports
│   ├── validated/              # Cleaned outputs + metrics
│   └── unstructured/           # Document conversions (PDF → images/base64)
│       ├── invoices/
│       ├── us_tax/
│       ├── vehicle_insurance/
│       └── test_images.json
│
├── logs/                       # App/Pipeline logs
├── reports/                    # Generated output reports
├── airflow-logs/               # Airflow task logs (mounted by compose)
│
├── scripts/                    # Core pipeline components
│   ├── bias_detection.py
│   ├── data_acquisition.py
│   ├── data_cleaning.py
│   ├── data_validation.py
│   ├── fetch_data.py
│   ├── pdf_2_image.py
│   ├── schema_detector.py
│   ├── upload_to_gcp.py
│   └── utils.py
│
└── tests/                      #  unified tests folder (pytest discovers this)
    ├── __init__.py
    ├── conftest.py
    ├── README.md
    ├── assets/
    │   └── test.pdf
    ├── test_bias_detection.py
    ├── test_data_cleaning.py
    ├── test_data_validation.py
    ├── test_pdf_conversion.py
    ├── test_schema_detector.py
    └── test_utils.py

```

---

##  Environment Setup
Our entire environment is being handled by our docker image created in docker-compose.yml

If you want to manually or individually run any components, you can setup the environment as following: 
### Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install Poppler (required for PDF conversion)

macOS:
```bash
brew install poppler
```

Ubuntu/Debian:
```bash
sudo apt-get install poppler-utils
```

---

## Running the Pipeline

### Update Docker Configuration

Inside `docker-compose.yml`, ensure these environment variables exist or import them using your env file

```yaml
environment:
  - GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json
  - AIRFLOW__CORE__DAGS_FOLDER=/app/dags
  - AIRFLOW__CORE__LOAD_EXAMPLES=False
  - PYTHONPATH=/app
  - GCP_PROJECT_ID=<your_project_id>
  - GCS_BUCKET_NAME=<your_bucket_name>
```

### Build Docker Images

Run from the data pipeline root

```bash
docker compose build
```

### Start Airflow Services

Launch all containers:

```bash
docker compose up -d
```
This starts:

    airflow-webserver

    airflow-scheduler

    airflow-postgres

    data-pipeline (custom compute container)

If needed, you can spin up the webserver manually using this:
```bash
docker compose up airflow-webserver
```

### Access Airflow UI

Open your browser and navigate to:

http://localhost:8080

Login with:

| Field     | Value  |
|----------|--------|
| Username | admin  |
| Password | admin  |

### Verify & Trigger the DAG

Ensure your DAG file (e.g., `retail_data_pipeline.py`) is located under: dags/

Airflow automatically detects new DAGs.  
Enable the DAG in the UI and manually trigger a run or let the scheduler execute it automatically.

---

### Running for unstructured data
For unstrucutred data, only the script for fetching the data is required which is executed inside our docker container. Extraction of structured data would eventually be handled by LLM 

You can run it manually using the following: 

```python
python scripts/fetch_data.py
```

---

## Reproducibility + Data Versioning (DVC)

This pipeline is designed to guarantee end-to-end reproducibility of each step of the data pipelien across environments, machines, and executions using Docker & DAG. Several layers of reproducibility are enforced:

To reproduce the results just run the commands from the section above again

This project uses **DVC** for data version control.

### Initialize DVC (first time only)
```bash
dvc init
```

### Track raw data
```bash
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw data"
```

### Get latest versioned data
```bash
dvc pull
```

### Reproduce full pipeline
```bash
dvc repro
```

<!-- View pipeline dependencies:
```bash
dvc dag
``` -->

---

## Testing

This directory contains 42 unit tests covering the essential components of the data pipeline.

### Test Files

- **test_data_validation.py** (8 tests) - Tests data validation logic (structure, nulls, duplicates, anomalies)
- **test_data_cleaning.py** (6 tests) - Tests data cleaning operations (normalization, missing values, outliers, type standardization)
- **test_bias_detection.py** (6 tests) - Tests bias detection and fairness metrics
- **test_schema_detector.py** (7 tests) - Tests automatic schema detection and type inference
- **test_utils.py** (11 tests) - Tests utility helper functions
- **test_pdf_conversion.py** (1 test) - Testing conversion from PDF to Image which is not used at the moment but will be used for the next phase


### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/test_data_validation.py -v
python -m pytest tests/test_data_cleaning.py -v
python -m pytest tests/test_bias_detection.py -v
python -m pytest tests/test_schema_detector.py -v
python -m pytest tests/test_utils.py -v
```
### With Coverage Report
```bash
python -m pytest tests/ --cov=scripts --cov-report=html
```

### What's Tested

#### Data Validation
- Minimum dataset dimensions (rows/columns)
- Null value percentage checks
- Duplicate detection
- Statistical anomaly detection

#### Data Cleaning
- Column name normalization
- Missing value handling (by type)
- Duplicate removal
- Outlier capping using IQR
- Data type standardization

#### Bias Detection
- Protected attribute identification
- Representation bias (group balance)
- Disparate impact ratio (80% rule)
- Statistical parity tests

#### Schema Detection
- Automatic column type inference
- Protected attribute detection
- Schema profile generation

#### Utilities
- Encoding detection
- Formatting functions (size, duration)
- Configuration validation

---

####  Outputs Produced

Upon successful processing, the module generates the following artifacts:

- `data/validated/<dataset_name>_cleaned.csv`  
- `data/validated/<dataset_name>_cleaning_metrics.json`

The metrics JSON file stores:
- Cleaning summary (rows/columns before vs. after)
- Data quality score
- Missing data statistics
- Outlier handling details
- List of transformations applied

---

####  Integration Within Pipeline

This cleaning stage is executed as a PythonOperator task in the Airflow DAG. It uses schema profiles from the validation step and ensures that every run produces reproducible, high-quality data outputs. Metrics are logged for monitoring and future audit needs.

---

This module plays a crucial role in improving data quality, increasing confidence in downstream analytics, and maintaining strong data governance throughout the pipeline.

## Individual Components

### Data Acquisition Module (data_acquisition.py — Structured) 

This module discovers, reads, and standardizes **structured** datasets (CSV/Excel/JSON/Parquet) into a canonical CSV in `data/raw/`. It auto-detects file format and encoding, loads the data robustly, and generates a schema profile for downstream validation/cleaning.

---

#### Inputs & Dependencies
- **Search locations:**  
  - `data/raw/` (default)
- **Supported formats:** `csv`, `xlsx`, `xls`, `json`, `parquet`, `txt` *(txt treated as unknown → error)*
- **Config keys:**  
  - `data.raw_path`  
  - `data.supported_formats`
- **Utilities:** `setup_logging`, `load_config`, `ensure_dir`
- **Schema:** `SchemaDetector.generate_schema_profile(df, dataset_name)`
- **Libraries:** `pandas`, `chardet`, `pathlib`, `shutil` (optional utilities)

---

#### Core Functions

| Step | Function | What it does |
|-----|----------|---------------|
| Detect encoding | `detect_encoding(file_path)` | Reads the first 100KB with `chardet` and returns `(encoding, confidence)`. Used for CSVs. |
| Detect format | `detect_file_format(file_path)` | Infers format from extension; returns one of `csv/excel/json/parquet/text/unknown`. |
| Read file | `read_data_file(file_path)` | Loads the file by format. For CSV: tries detected encoding, then falls back to common encodings. |
| Discover file | `find_data_files(config)` | Searches configured locations (`data/raw/`) for the first file matching supported formats. |
| Acquire | `acquire_data(source_file=None)` | Orchestrates the flow: ensures raw dir, finds/reads file, logs summary, saves standardized CSV, generates schema profile, and prints next steps. |

---

#### Outputs Produced
- Canonical raw CSV:  
  `data/raw/<dataset_name>.csv`
- Schema profile JSON:  
  `config/dataset_profiles/<dataset_name>_profile.json`
- Logged acquisition summary (rows, columns, detected column types, protected attributes).

---

#### ▶ Example Usage
```bash
# Auto-detect a file under data/raw/
python scripts/data_acquisition.py
```

Or specify a concrete source file
```bash
python scripts/data_acquisition.py data/raw/retail_uk.csv
```

### **Data Cleaning Module (data_cleaning.py)**

This module is responsible for applying a comprehensive set of cleaning operations to validated structured data. The objective is to ensure data consistency, completeness, and usability before downstream analytics or ML processing. The cleaning workflow is fully automated and integrated within the Airflow pipeline.

The following operations are executed sequentially inside the `DataCleaner` class:

| Operation | Function Name | Description |
|----------|----------------|-------------|
| Normalize Column Names | `normalize_column_names()` | Standardizes column names by converting to lowercase and replacing spaces/special characters with underscores to improve schema consistency. |
| Standardize Data Types | `standardize_data_types()` | Converts each column to its appropriate type (datetime, numerical, categorical, text) based on the schema profile or intelligent auto-detection. |
| Handle Missing Values | `handle_missing_values()` | Ensures completeness by detecting missing entries and filling them using schema-aware strategies: identifiers dropped if missing, categorical → mode or “Unknown”, continuous → median, discrete numeric → mode. |
| Remove Duplicates | `remove_duplicates()` | Identifies duplicate records and removes them to maintain accuracy and data integrity. |
| Outlier Treatment | `handle_outliers()` | Detects outliers using configurable methods (IQR/Z-score) and caps values to reduce noise while preserving structure. |
| Remove Constant Columns | `remove_constant_columns()` | Removes columns with only one unique value since they provide no analytical value. |

---

### GitHub Data Fetch Module (fetch_data.py) for unstructured data

This module automates the acquisition of **unstructured data assets** directly from a public GitHub repository. It recursively fetches files and folders from a specified repo path using the GitHub API, ensuring that unstructured document samples are always up to date.

---

####  Inputs & Dependencies

- **Default Source**
  - GitHub Owner: `Azure-Samples`
  - Repository: `azure-ai-document-processing-samples`
  - Folder Path: `samples/assets`
  - These defaults can be overridden via CLI arguments.
- **Default Destination**
  - `data/unstructured/`
- Libraries:
  - `requests`, `argparse`, `pathlib`, `os`

---

#### Core Functionality

| Function | Purpose |
|---------|---------|
| `download_content(url, local_path)` | Calls GitHub API recursively. Downloads files and creates subdirectories to mirror the repo structure. |
| `main()` | Builds GitHub API URL, parses command-line args, and triggers the recursive fetch. |

GitHub API endpoint format:
https://api.github.com/repos/\<\owner>/<\repo>/contents path


### **Data Validation Module (data_validation.py)**

This module performs schema-aware validation of incoming **raw** datasets before they enter the cleaning stage. It loads (or generates) a schema profile, validates structural properties, checks data completeness and duplication, detects numerical anomalies, and writes an auditable validation report.

---

#### Inputs & Dependencies
- **Input file:** `data/raw/<dataset_name>.csv`
- **Config keys used:**
  - `validation.min_rows`
  - `validation.min_columns`
  - `validation.null_threshold`  *(fraction, e.g., 0.2 for 20%)*  
  - `validation.duplicate_threshold` *(fraction of duplicate rows allowed)*
- **Utilities:** `setup_logging`, `load_config`, `ensure_dir`, `detect_encoding`
- **Schema:** `SchemaDetector.load_schema_profile(dataset_name)`  
  Fallback: generates a new profile when none is found.

---

#### Validation Checks (executed by `DataValidator.validate()`)

| Check | Function | What it does | Output recorded in report |
|------|----------|--------------|---------------------------|
| Basic Structure | `validate_basic_structure(df)` | Confirms minimum rows/columns according to config. | `checks.structure` with actual vs. required rows/cols and boolean `valid`. |
| Nulls | `check_nulls(df)` | Computes per-column null percentages, flags columns exceeding `null_threshold`. | `checks.nulls` with per-column percentages, list of offenders, overall null % and boolean `valid`. |
| Duplicates | `check_duplicates(df)` | Counts duplicate rows and compares the share to `duplicate_threshold`. | `checks.duplicates` with count, percentage, and boolean `valid`. |
| Numerical Anomalies | `detect_anomalies(df)` | For numerical columns (from schema or inferred): computes IQR bounds (3×IQR) and reports outliers. | `checks.anomalies` with columns analyzed, columns with anomalies, and detailed bounds & counts per column. |

> The module also detects file encoding via `detect_encoding()` before loading to prevent decode errors and to improve robustness on diverse sources.

---

####  Report & Artifacts

- **Validation Report (JSON):**  
  `data/processed/<dataset_name>_validation_report.json`  
  Contains:
  - `timestamp`, `dataset_name`
  - `checks.structure`, `checks.nulls`, `checks.duplicates`, `checks.anomalies`
  - `overall_valid` (boolean)

- **Pass-through Validated Data (CSV):**  
  `data/processed/<dataset_name>_validated.csv`  
  The raw dataset persisted for downstream cleaning, regardless of pass/fail, with validation results logged for audit.

---

#### Behavior Notes

- **Schema awareness:** If a schema profile exists, column role/type information is used to drive which columns are treated as numerical; otherwise types are inferred.
- **Strictness knobs:** Update thresholds in `config` to tighten or relax acceptance criteria without code changes.
- **Logging:** All steps are logged under the `data_validation` logger for traceability in Airflow.

---

#### ▶Example Invocation

```bash
python scripts/data_validation.py <dataset_name>
```

---

### Storage & Upload (upload_to_gcp.py)**

This module handles authenticated interactions with **Google Cloud Storage (GCS)** for uploading versioned datasets, downloading staged artifacts, listing available datasets, and pushing a comprehensive **dataset metadata manifest**. It integrates tightly with the pipeline configuration, uses structured logging, and attaches useful file-level metadata at upload time.

---

####  Inputs & Dependencies

- **Config keys (config.yaml):**
  - `gcp.project_id`
  - `gcp.bucket_name`
  - `data.raw_path`
  - `data.processed_path`
  - `data.validated_path`
- **Environment:**
  - `GOOGLE_APPLICATION_CREDENTIALS` → path to a valid GCP service account JSON
- **Utilities used (`utils.py`):**
  - `setup_logging`, `load_config`, `validate_gcp_credentials`
  - `get_file_info` (size, path, etc.), `format_size`
  - `ensure_dir`, `print_success`, `print_error`, `print_info`
- **Third-party:**
  - `google-cloud-storage`, `google-auth`, `pandas`

---

####  Authentication

- `initialize_gcs_client()`  
  Validates credentials via `validate_gcp_credentials()`, loads the service account JSON from `GOOGLE_APPLICATION_CREDENTIALS`, and returns an authenticated `storage.Client(project_id)` instance.

---

####  Core Operations

| Operation | Function | What it does | Inputs | Outputs |
|---|---|---|---|---|
| Upload a single file | `upload_file_to_gcs(client, bucket, source_file, dest_blob)` | Uploads a local file to `gs://bucket/dest_blob`, sets metadata (`uploaded_at`, `original_filename`, `file_size`) | GCS client, bucket name, local path, destination key | Dict with `bucket`, `blob_name`, `size`, `public_url`, `uploaded_at` |
| Upload dataset bundle | `upload_to_gcs(dataset_name, include_raw=False, include_reports=True)` | Uploads `raw`, `processed`, `validated` CSVs and optional reports; then generates & uploads a dataset manifest | `dataset_name`, flags | Upload summary dict (file count, total size, files) |
| Upload dataset metadata | `upload_dataset_metadata(client, bucket, dataset_name, metadata)` | Writes combined JSON manifest to `gs://bucket/metadata/<dataset>_metadata.json` | GCS client, bucket, dataset, JSON metadata | None |
| Generate metadata | `generate_dataset_metadata(dataset_name, config)` | Scans local `raw/ processed/ validated/` files; reads CSVs for shape, columns, memory; attaches schema/validation/bias/cleaning artifacts if present | Dataset, config | Dict: `stages`, `schema_profile`, `validation_report`, `bias_report`, `cleaning_metrics` |
| Download staged file | `download_from_gcs(dataset_name, stage='processed', destination_path=None)` | Downloads `gs://data/<stage>/<dataset>_<stage>.csv` to local default path based on stage | Dataset, stage, optional destination | Local path string |
| List datasets | `list_gcs_datasets(bucket_name=None)` | Lists dataset names found under `data/processed/` in the bucket | Optional bucket | Sorted list of dataset names |

---

> Note: The uploader maps local filenames to standardized cloud keys. Locally, cleaned CSV is `<dataset>_cleaned.csv`, which is uploaded under `data/validated/<dataset>_validated.csv`.

---

#### Metadata Manifest Contents

`metadata/<dataset>_metadata.json` includes:
- `dataset_name`, `upload_timestamp`
- `stages`:
  - For each of `raw`, `processed`, `validated` (if present):  
    - `file_info` (size bytes, mtime, path)
    - `rows`, `columns`, `column_names`
    - `memory_usage_mb`
- Attached artifacts if found:
  - `schema_profile` (`config/dataset_profiles/<dataset>_profile.json`)
  - `validation_report` (`data/processed/<dataset>_validation_report.json`)
  - `bias_report` (`data/processed/<dataset>_bias_report.json`)
  - `cleaning_metrics` (`data/validated/<dataset>_cleaning_metrics.json`)
- `gcs_files`: list of uploaded file descriptors

This single file gives consumers a one-stop summary of the dataset’s current state and lineage.

---

#### CLI Usage

```bash
# Upload processed + validated (default), optionally include raw
python scripts/upload_to_gcp.py upload --dataset <dataset_name> [--include-raw] [--include-reports]

# Download a staged artifact back to local
python scripts/upload_to_gcp.py download --dataset <dataset_name> --stage processed
# stages: raw | processed | validated

# List discovered datasets in GCS (under data/processed/)
python scripts/upload_to_gcp.py list
```

