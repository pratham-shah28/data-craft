# DataCraft Platform - MLOps Project

**Team Members:** Isha Singh, Shivie Saksenaa, Pratham Sachinbhai Shah, Vishal Singh Rajpurohit, Sanskar Sharma, Tisha Patel

---

##  Project Vision

**DataCraft** is an AI-powered platform designed to democratize data insights within organizations by enabling both technical and non-technical users to extract actionable insights from structured and unstructured data through natural language queries.

### Core Problems We Solve

1. **Data Accessibility Gap** - Technical barriers prevent business users from accessing insights in enterprise data
2. **Unstructured Data Challenge** - Large volumes of unstructured data (invoices, reports, PDFs) are difficult to analyze
3. **Visualization Complexity** - Creating meaningful charts requires specialized tools and expertise
4. **Time-to-Insight Delay** - Manual data analysis creates bottlenecks in decision-making
5. **Cost of Business Intelligence** - Traditional BI solutions require significant investment

### Our Solution

A two-phase approach that combines automated data processing with AI-powered insights:

**Phase 1 (Current):** Production-grade data pipeline for structured data processing  
**Phase 2 (Upcoming):** LLM-based conversion of unstructured data into structured formats, enabling natural language querying and automated visualization

---

##  Architecture Overview

### System Flow

```
USER INPUT
    ↓
┌─────────────────────────────────────────┐
│  Phase 1: Data Pipeline (CURRENT)      │
│  ├─ Structured Data (CSV, Excel)       │
│  ├─ Unstructured Data Acquisition      │
│  └─ ETL + Validation + Bias Detection  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Phase 2: LLM Processing (PLANNED)     │
│  ├─ LLM1: Convert Unstructured → JSON  │
│  ├─ LLM2: Natural Language Queries     │
│  └─ Auto-generate Charts & Insights    │
└─────────────────────────────────────────┘
    ↓
OUTPUT: Clean Data + Visualizations + Insights
```

---

## 🔧 Current Implementation: Data Pipeline

This repository contains **Phase 1** - a production-grade MLOps data pipeline that serves as the foundation for the DataCraft platform.

### Pipeline Architecture

```
Acquire Data → Validate → Clean → Detect Bias
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            DVC Version Control                 Upload to GCS
            (Git + DVC Push)                    (Cloud Storage)
                    └─────────────────┬─────────────────┘
                                      ↓
                            Generate Summary Report
                                      ↓
                              Email Notifications
```

### What the Pipeline Does

The data pipeline automates the journey from raw data ingestion to clean, validated, version-controlled outputs ready for ML/analytics:

1. **Data Acquisition**
   - **Structured Data:** Ingests Walmart retail dataset (CSV/Excel)
   - **Unstructured Data:** Downloads documents from Azure samples (PDFs, invoices, forms)
   - Auto-detects file formats and encodings
   - Generates schema profiles

2. **Data Validation**
   - Checks schema consistency and data types
   - Validates null value thresholds (<15%)
   - Detects duplicate records (<5%)
   - Identifies statistical anomalies using IQR method

3. **Data Cleaning**
   - Normalizes column names (lowercase, underscores)
   - Handles missing values (mode for categorical, median for numeric)
   - Removes duplicates
   - Caps outliers using IQR or Z-score methods
   - Standardizes data types

4. **Bias Detection**
   - Auto-identifies protected attributes (gender, age, region, etc.)
   - Statistical parity testing (ANOVA)
   - Disparate impact ratio analysis (80% rule)
   - Representation bias detection
   - Generates actionable fairness recommendations

5. **Version Control (DVC)**
   - Tracks all data transformations
   - Git integration for metadata
   - Full reproducibility of pipeline runs

6. **Cloud Upload (GCP)**
   - Stores cleaned datasets in Google Cloud Storage
   - Uploads validation, cleaning, and bias reports
   - Maintains structured folder hierarchy

7. **Reporting & Alerts**
   - Generates JSON summary reports
   - Email alerts on validation failures or anomalies
   - Success notifications on completion

---

##  Project Structure

```
mlops-project/
│
├── data-pipeline/                     # Current Phase 1 Implementation
│   ├── dags/
│   │   └── retail_data_pipeline.py    # Airflow orchestration DAG
│   │
│   ├── scripts/
│   │   ├── data_acquisition.py        # Data ingestion + encoding detection
│   │   ├── schema_detector.py         # Schema profiling + type detection
│   │   ├── data_validation.py         # Quality checks
│   │   ├── data_cleaning.py           # Cleaning + standardization
│   │   ├── bias_detection.py          # Fairness analysis
│   │   ├── pdf_2_image.py             # PDF → Base64 conversion
│   │   ├── fetch_data.py              # Unstructured data downloader
│   │   ├── upload_to_gcp.py           # GCS uploader
│   │   └── utils.py                   # Shared utilities
│   │
│   ├── config/
│   │   ├── pipeline_config.yaml       # Pipeline configuration
│   │   └── dataset_profiles/          # Auto-generated schemas
│   │
│   ├── data/
│   │   ├── raw/                       # Source datasets
│   │   ├── processed/                 # Validated data
│   │   ├── validated/                 # Cleaned data
│   │   └── unstructured/              # PDF conversions
│   │
│   ├── tests/                         # Unit tests
│   ├── reports/                       # Pipeline execution reports
│   ├── logs/                          # Application logs
│   │
│   ├── Dockerfile
│   ├── docker-compose.yml             # Multi-container setup
│   ├── requirements.txt
│   └── dvc.yaml                       # DVC pipeline definition
│
├── backend/                           
└── frontend/                          
```

---

##  Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional)
- Google Cloud Platform account
- Poppler (for PDF processing)

### Installation

```bash
# Clone repository
git clone https://github.com/pratham-shah28/mlops-project.git
cd mlops-project/data-pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Poppler
# macOS: brew install poppler
# Ubuntu: sudo apt-get install poppler-utils
```

### Configuration

1. **Edit** `config/pipeline_config.yaml`:
   ```yaml
   gcp:
     bucket_name: "your-gcs-bucket-name"
     project_id: "your-gcp-project-id"
     credentials_path: "gcp/service-account.json"
   ```

2. **Place GCP service account key** at `gcp/service-account.json`

3. **Configure email alerts** in `docker-compose.yml` (optional)

### Running the Pipeline

**Option 1: Docker Compose (Recommended)**

```bash
docker-compose up -d

# Access Airflow UI: http://localhost:8080
# Username: admin | Password: admin
```

**Option 2: Standalone Scripts**

```bash
# Run individual stages
python scripts/data_acquisition.py data/raw/your-file.csv
python scripts/data_validation.py <dataset_name>
python scripts/data_cleaning.py <dataset_name>
python scripts/bias_detection.py <dataset_name>
```

**Option 3: DVC Pipeline**

```bash
dvc repro              # Reproduce full pipeline
dvc dag                # View dependencies
dvc push               # Push versioned data
```

---

##  Dataset Information

### Structured Data: Walmart Retail Dataset
- **Source:** [GitHub - Walmart eCommerce Analysis](https://github.com/virajbhutada/walmart-ecommerce-retail-analysis/tree/main/data)
- **Format:** CSV/Excel
- **Contains:** Product catalog, sales, customer ratings, reviews, transactions
- **Purpose:** Primary dataset for ETL and semantic querying

### Unstructured Data: Azure AI Document Samples
- **Source:** [GitHub - Azure AI Document Processing](https://github.com/Azure-Samples/azure-ai-document-processing-samples.git)
- **Format:** PNG/JPG (scanned docs), PDF, JSON
- **Contains:** Contracts, invoices, receipts, forms, ID cards
- **Purpose:** Demonstrates unstructured document workflows for Phase 2

---

##  Key Features

### Data Quality Assurance
 Automated schema detection and validation  
 Missing value detection with smart imputation  
 Duplicate record identification  
 Outlier detection (IQR/Z-score methods)  
 Data quality scoring  

### Bias Detection & Fairness
 Auto-detection of protected attributes  
 Statistical parity testing (ANOVA)  
 Disparate impact ratio (80% rule)  
 Representation bias detection  
 Actionable bias mitigation recommendations  

### MLOps Best Practices
 Full data lineage (DVC)  
 Reproducible pipelines  
 Apache Airflow orchestration  
 Cloud-native (GCP)  
 Email notifications  
 Comprehensive logging  
 Docker containerization  
 Unit testing  

---

##  Metrics & Business Goals

### Key Metrics
- **Dashboard Generation Time** - Time from raw data to insights
- **Query Accuracy** - Percentage of correctly processed queries
- **System Response Time** - API latency for data processing
- **Data Coverage** - Percentage of data successfully processed
- **User Satisfaction** - Usefulness of auto-generated outputs

### Business Goals
- **Simplify Data Access** - Make company data accessible to all employees
- **Save Time** - Reduce hours spent on manual report generation
- **Improve Decision Making** - Quick access to relevant metrics
- **Cost Reduction** - Decrease dependency on expensive BI tools

---

##  Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | Apache Airflow 2.9.2 | DAG execution, scheduling, monitoring |
| Language | Python 3.9 | Core pipeline implementation |
| Cloud | Google Cloud Platform | Scalable storage (GCS) |
| Version Control | DVC + Git | Data and metadata versioning |
| Data Processing | Pandas, NumPy | Data manipulation |
| Containerization | Docker + Compose | Reproducible environments |
| Testing | Pytest | Unit and integration tests |

---

##  Monitoring & Observability

### Email Notifications

**Anomaly Alerts** - Triggered when:
- Data validation fails
- Missing values exceed 15%

**Success Notifications** - Sent after:
- All pipeline stages complete successfully
- Summary report is generated

### Logging

All pipeline components generate structured logs:
- `logs/data_acquisition_<timestamp>.log`
- `logs/data_validation_<timestamp>.log`
- `logs/data_cleaning_<timestamp>.log`
- `logs/bias_detection_<timestamp>.log`
- `logs/gcp_upload_<timestamp>.log`

### Airflow UI

Access `http://localhost:8080` for:
- Real-time DAG status
- Task duration visualization (Gantt chart)
- Individual task logs
- Manual pipeline triggers
- Execution history

---

##  Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=scripts tests/

# Run specific test
pytest tests/test_bias_detection.py -v
```

Test files:
- `test_schema_detector.py` - Schema profiling
- `test_data_validation.py` - Quality checks
- `test_data_cleaning.py` - Cleaning operations
- `test_bias_detection.py` - Fairness analysis
- `test_utils.py` - Utility functions

---


##  Known Challenges & Mitigations

| Challenge | Mitigation Strategy |
|-----------|-------------------|
| **LLM1 Processing Speed** | Pre-process and cache common document types |
| **Data Coherence** | Schema harmonization and validation checks |
| **Context Management** | Implement conversation history tracking |
| **Vague User Queries** | Query suggestion system and clarification prompts |
| **Large Dataset Performance** | Sampling, caching, and query size limits |
| **Visualization Errors** | Auto-mapping chart types with fallback to tables |

---

##  License

This project is licensed under the MIT License.

---

##  Team

- **Isha Singh** 
- **Shivie Saksenaa** 
- **Pratham Sachinbhai Shah**
- **Vishal Singh Rajpurohit**
- **Sanskar Sharma**
- **Tisha Patel**

---

##  Resources

- **Repository:** [github.com/pratham-shah28/mlops-project](https://github.com/pratham-shah28/mlops-project)
- **Documentation:** See individual module READMEs in subdirectories
- **Issues:** [GitHub Issues](https://github.com/pratham-shah28/mlops-project/issues)

---

**Last Updated:** October 28, 2025  
**Status:** Phase 1 Complete | Phase 2 In Progress
