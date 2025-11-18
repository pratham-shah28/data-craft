# CI/CD Files Overview

This document provides an overview of all files in the CI/CD implementation.

## 📁 Directory Structure

```
model-training/ci-cd/
├── config/
│   └── ci_cd_config.yaml          # Configuration file
├── scripts/
│   ├── run_training_pipeline.py   # Main training pipeline runner
│   ├── validate_model_metrics.py   # Model validation script
│   ├── check_bias_thresholds.py    # Bias checking script
│   ├── compare_models.py           # Model comparison script
│   ├── push_to_registry.py         # Registry push script
│   ├── deploy_model.py             # Deployment script
│   └── send_notifications.py       # Email notification script
├── README.md                        # Complete setup guide
├── QUICK_START.md                   # Quick setup guide
└── FILES_OVERVIEW.md                # This file

.github/workflows/
└── model-training-ci.yml            # GitHub Actions workflow
```

---

## 📄 File Descriptions

### Configuration Files

#### `config/ci_cd_config.yaml`
**Purpose**: Central configuration file for all CI/CD settings

**Contains**:
- GCP configuration (project, region, bucket)
- Validation thresholds (min scores, max bias)
- Bias detection thresholds
- Rollback settings
- Email notification settings
- Deployment settings
- Paths configuration

**Usage**: Edit this file to adjust pipeline behavior

---

### Scripts

#### `scripts/run_training_pipeline.py`
**Purpose**: Main entry point for running the training pipeline

**Features**:
- Can trigger Airflow DAG or run directly
- Executes full model training workflow
- Generates pipeline summary JSON

**Usage**:
```bash
python run_training_pipeline.py --direct
python run_training_pipeline.py --trigger-dag
```

**Output**: `outputs/model-training/pipeline_summary.json`

---

#### `scripts/validate_model_metrics.py`
**Purpose**: Validates model metrics against quality thresholds

**Checks**:
- Composite score ≥ 70.0
- Performance score ≥ 75.0
- Bias score ≤ 40.0
- Success rate ≥ 80.0
- Overall accuracy ≥ 75.0

**Usage**:
```bash
python validate_model_metrics.py [metrics_file]
```

**Exit Code**: 0 if pass, 1 if fail

**Output**: Prints validation results to console

---

#### `scripts/check_bias_thresholds.py`
**Purpose**: Checks if bias scores exceed acceptable limits

**Checks**:
- Bias score ≤ 50.0
- Severity ≠ HIGH
- Number of biases ≤ 2

**Usage**:
```bash
python check_bias_thresholds.py [bias_report_file]
```

**Exit Code**: 0 if pass, 1 if fail

**Output**: Prints bias check results to console

---

#### `scripts/compare_models.py`
**Purpose**: Compares new model with previous model version

**Features**:
- Retrieves previous model from GCS
- Compares composite scores
- Blocks if performance degrades > 5%
- Implements rollback logic

**Usage**:
```bash
python compare_models.py [current_metrics_file]
```

**Exit Code**: 0 if new model is better, 1 if worse

**Output**: Prints comparison results to console

---

#### `scripts/push_to_registry.py`
**Purpose**: Pushes model to GCP Artifact Registry and Vertex AI Model Registry

**Features**:
- Uploads artifacts to GCS
- Pushes to Vertex AI Model Registry
- Pushes to Artifact Registry (optional)
- Creates versioned model entries

**Usage**:
```bash
python push_to_registry.py [metadata_file] [artifacts_dir]
```

**Output**: Model resource name for deployment

**Dependencies**: Requires GCP authentication and permissions

---

#### `scripts/deploy_model.py`
**Purpose**: Deploys model to Vertex AI endpoint

**Features**:
- Creates or updates Vertex AI endpoint
- Deploys model with traffic routing
- Supports manual approval (auto-deploy disabled by default)

**Usage**:
```bash
python deploy_model.py <model_resource_name>
```

**Output**: Endpoint resource name

**Dependencies**: Model must be in Vertex AI Model Registry

---

#### `scripts/send_notifications.py`
**Purpose**: Sends email notifications for pipeline events

**Event Types**:
- `success` - Training completed successfully
- `failed` - Training failed
- `validation_failed` - Validation failed
- `bias_failed` - Bias check failed
- `comparison_failed` - Model comparison failed

**Usage**:
```bash
python send_notifications.py success outputs/model-training/pipeline_summary.json
python send_notifications.py failed "Error message"
python send_notifications.py validation_failed
```

**Dependencies**: Requires email configuration in secrets

---

### Workflow Files

#### `.github/workflows/model-training-ci.yml`
**Purpose**: GitHub Actions workflow definition

**Triggers**:
- Push to `model-training/` directory
- Pull requests affecting `model-training/`
- Manual trigger (workflow_dispatch)

**Steps**:
1. Checkout code
2. Set up Python
3. Install dependencies
4. Authenticate to GCP
5. Run training pipeline
6. Validate model metrics
7. Check bias thresholds
8. Compare with previous model
9. Push to registry
10. Deploy model (optional)
11. Send notifications
12. Upload artifacts

**Dependencies**: Requires all GitHub Secrets configured

---

### Documentation Files

#### `README.md`
**Purpose**: Complete setup and configuration guide

**Contents**:
- Overview of CI/CD pipeline
- Prerequisites
- GCP setup instructions
- GitHub Secrets configuration
- Email configuration
- Local testing guide
- Pipeline workflow explanation
- Troubleshooting guide
- Maintenance instructions
- Security best practices
- Setup checklist

**Audience**: Anyone setting up or maintaining the CI/CD pipeline

---

#### `QUICK_START.md`
**Purpose**: Quick reference for fast setup

**Contents**:
- 5-minute setup guide
- Essential commands
- Verification steps
- Quick troubleshooting

**Audience**: Users who want to get started quickly

---

#### `FILES_OVERVIEW.md`
**Purpose**: This file - overview of all CI/CD files

**Contents**:
- Directory structure
- File descriptions
- Usage examples
- Dependencies

**Audience**: Developers understanding the codebase

---

## 🔗 Dependencies Between Files

```
GitHub Actions Workflow
    ↓
run_training_pipeline.py
    ↓
validate_model_metrics.py ──┐
check_bias_thresholds.py ────┼──→ compare_models.py
    ↓                         │
    ↓                         │
push_to_registry.py ←─────────┘
    ↓
deploy_model.py
    ↓
send_notifications.py
```

## 📊 Data Flow

1. **Training** → Generates metrics, evaluation reports, bias reports
2. **Validation** → Reads metrics, validates against thresholds
3. **Bias Check** → Reads bias reports, checks thresholds
4. **Comparison** → Reads current metrics, fetches previous from GCS
5. **Registry** → Reads metadata, uploads artifacts, pushes to registry
6. **Deployment** → Uses model resource name from registry
7. **Notifications** → Reads metrics/reports, sends emails

## 🔧 Configuration Flow

```
ci_cd_config.yaml
    ↓
All scripts read config
    ↓
Scripts use config values for:
  - Thresholds
  - GCP settings
  - Email settings
  - Paths
```

## 🎯 Key Files for Handoff

**Essential for Setup**:
1. `README.md` - Complete setup guide
2. `QUICK_START.md` - Quick reference
3. `config/ci_cd_config.yaml` - Configuration

**Essential for Operation**:
1. `.github/workflows/model-training-ci.yml` - Workflow definition
2. `scripts/run_training_pipeline.py` - Main runner
3. `scripts/validate_model_metrics.py` - Quality gate
4. `scripts/send_notifications.py` - Notifications

**Essential for Troubleshooting**:
1. `README.md` - Troubleshooting section
2. `scripts/*.py` - Individual script logs

---

**Last Updated**: 2025-01-27

