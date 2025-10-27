# Data Pipelines

Here we implement a dual-pipeline architecture to handle both unstructured and structured data. The first pipeline focuses on transforming document-based data, such as PDFs, into a machine-readable format. The second pipeline follows a traditional ETL flow for structured datasets, where data is cleaned, validated, version-controlled, and prepared for downstream analytics.

In the unstructured pipeline, we leverage an LLM as part of the data processing workflow. We first fetch documents from external sources using a script, then convert each PDF page into a base64-encoded image representation using a custom preprocessing script. This prepares the data to be directly consumed by vision-enabled LLMs and allow them to produce structured data which can go into our traditional data-pipeline. We have also included dedicated unit tests to ensure the correctness of this conversion step and the robustness of the preprocessing stage.

To complement the LLM based data processing, the second phase follows a more traditional ETL workflow for structured datasets. After the data is acquired, the pipeline performs schema-based validation, cleaning, anomaly and bias detection, and then pushes each processed version into DVC for full reproducibility. The DAG above illustrates this end-to-end flow orchestrated in Apache Airflow, where each transformation step is modular, observable, and fault-tolerant. Once the data has been validated and version-controlled, it is pushed to cloud storage and a final report is generated summarizing the pipeline execution outcomes. Together, these two pipelines enable a unified MLOps system capable of handling real-world retail data in both structured and document-based formats, ensuring reliable, scalable, and automation-ready data delivery for downstream analytics and machine learning tasks.

---

## 📁 Project Structure

```plaintext
data-pipeline/
│
├── config/                       # Configuration files
├── dags/                         # Orchestration scripts
│   └── retail_data_pipeline.py
│
├── data/
│   ├── raw/                      # Source datasets
│   ├── processed/                # Cleaned datasets
│   ├── validated/                # Data post-validation
│   └── unstructured/             # Document conversions (PDF → images)
│       ├── invoices/
│       ├── us_tax/
│       └── vehicle_insurance/
│       └── test_images.json
│
├── reports/                      # Generated output reports
│
├── scripts/                      # Core pipeline components
│   ├── data_acquisition.py       # Fetch external data
│   ├── pdf_2_image.py            # Convert PDFs to base64 images
│   ├── data_cleaning.py
│   ├── data_validation.py
│   ├── schema_detector.py
│   ├── upload_to_gcp.py
│   ├── bias_detection.py
│   ├── fetch_data.py
│   └── utils.py
│
└── test/
    ├── assets/                   # Sample test files
    │   └── test.pdf
    └── test_pdf_conversion.py
```

---

## ⚙️ Environment Setup

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

## 🚀 Running the Pipeline

### Full pipeline execution
```bash
python dags/retail_data_pipeline.py
```

### Run individual components

Example: Convert PDF to base64 images
```python
from scripts.pdf_2_image import pdf_to_base64_images
pdf_to_base64_images("test/assets/test.pdf")
```

---

## 🔁 Reproducibility + Data Versioning (DVC)

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

View pipeline dependencies:
```bash
dvc dag
```

---

## 🧪 Testing

Run unit tests:
```bash
pytest -q
```

Example test:
- `test/test_pdf_conversion.py` validates correct PDF → image output format

---

## Key Features

| Feature | Description |
|--------|-------------|
| Automated data ingestion | Fetches open-source datasets |
| PDF document parsing | Converts unstructured docs into AI-readable format |
| Schema enforcement | Ensures validated and consistent output |
| Reproducible pipeline | Full data lineage tracked with DVC |
| Modular components | Easy to expand and integrate |

---


