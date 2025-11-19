# CI/CD Pipeline for Model Training

Complete guide for setting up and running the automated model training CI/CD pipeline.

## 📋 Table of Contents

1. [Quick Setup](#quick-setup)
2. [Configuration](#configuration)
3. [Running the Pipeline](#running-the-pipeline)
4. [Pipeline Stages](#pipeline-stages)
5. [Troubleshooting](#troubleshooting)
6. [Implementation Details](#implementation-details)

---

## Quick Setup

### Prerequisites

- GitHub repository access
- GCP account with project access
- GCP service account with required permissions:
  - BigQuery Data Editor
  - Storage Admin
  - Vertex AI User (only if using Vertex AI Model Registry - optional)
- Email account for notifications

**Note:** Vertex AI is optional. The pipeline works with GCS only, which is sufficient for version control and reproducibility.

### Step 1: Configure Your Settings

Edit `ci-cd/config/ci_cd_config.yaml` and update these values to match your setup:

```yaml
gcp:
  project_id: "your-gcp-project-id"      # ← Update this
  region: "your-region"                    # ← Update this
  dataset_id: "your-bigquery-dataset"      # ← Update this
  bucket_name: "your-gcs-bucket"           # ← Update this
  vertex_ai:
    enabled: false  # Optional: Set to true only if you need Vertex AI features
```

### Step 2: Configure Email Settings

Edit `ci-cd/config/ci_cd_config.yaml` and update email settings:
```yaml
notifications:
  email:
    from_email: "your-email@gmail.com"  # ← Update this
    to_email: "your-email@gmail.com"     # ← Update this
```

### Step 3: Set GitHub Secrets

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**

Add these secrets:

- **`GCP_SA_KEY`**: Your GCP service account JSON (copy entire JSON content)
- **`EMAIL_SMTP_USER`**: (Optional) Your email address (defaults to `from_email` in config)
- **`EMAIL_SMTP_PASSWORD`**: **REQUIRED** - Your email app password (for Gmail, use app password)

### Step 4: Verify GCP Resources

Ensure these exist in your GCP project:
- **BigQuery dataset** (name from `dataset_id` in config)
- **GCS bucket** (name from `bucket_name` in config)
- **Service account** has these roles:
  - BigQuery Data Editor
  - Storage Admin
  - Vertex AI User (only if `vertex_ai.enabled: true` in config)

**Note:** Vertex AI is optional. If `vertex_ai.enabled: false`, you only need BigQuery Data Editor and Storage Admin roles.

### Step 5: Trigger Pipeline

Push code to `main` branch:
```bash
git push origin main
```

The pipeline triggers automatically when files in `model-training/**` or `ci-cd/**` change.

---

## Configuration

### Config File: `ci-cd/config/ci_cd_config.yaml`

All CI/CD settings are in this file. Update it to match your environment:

```yaml
gcp:
  project_id: "your-project-id"      # Your GCP project ID
  region: "us-east1"                 # Your GCP region
  dataset_id: "your_dataset"         # BigQuery dataset ID
  bucket_name: "your-bucket"         # GCS bucket for model artifacts
  vertex_ai:
    enabled: false                   # Optional: Enable Vertex AI Model Registry

notifications:
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    from_email: "your-email@gmail.com"
    to_email: "your-email@gmail.com"

model_registry:
  base_path: "models"                # Path in bucket for models
  version_format: "timestamp_commit"

rollback:
  enabled: true
  min_improvement_threshold: 0.0
```

### Validation Thresholds: `ci-cd/config/validation_thresholds.yaml`

Adjust these thresholds based on your requirements:

```yaml
validation_thresholds:
  performance:
    min_overall_score: 75.0
    min_success_rate: 80.0
    min_syntax_validity: 85.0
    min_intent_matching: 70.0
  
  bias:
    max_bias_score: 30.0
    max_severity: "MEDIUM"
  
  execution:
    min_execution_success_rate: 85.0
    min_result_validity_rate: 80.0
    min_overall_accuracy: 75.0
```

---

## Running the Pipeline

### Automatic Trigger

The pipeline **automatically triggers** when you push to `main` branch and files in `model-training/**` or `ci-cd/**` change.

### Manual Trigger

1. Go to GitHub → **Actions** tab
2. Select **Model Training CI/CD Pipeline**
3. Click **Run workflow** → Select branch → **Run workflow**

### Local Testing

For local testing:

1. **Update config file**: Edit `ci-cd/config/ci_cd_config.yaml` with your settings

2. **Set required environment variables**:
```bash
# GCP credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Email password (required for notifications)
export EMAIL_SMTP_PASSWORD="your-app-password"

# Optional: Override config values with env vars
export GCP_PROJECT_ID="your-project-id"  # Overrides config
export EMAIL_SMTP_USER="your-email@gmail.com"  # Overrides from_email in config
```

3. **Run scripts**:
```bash
python ci-cd/scripts/validate_model.py
python ci-cd/scripts/send_notification.py --status success
```

---

## Pipeline Stages

The pipeline runs these stages sequentially:

1. **Setup** - Install dependencies, authenticate with GCP
2. **Train** - Run model training pipeline
3. **Validate** - Check model performance against thresholds
4. **Detect Bias** - Run bias detection checks
5. **Compare & Select** - Compare with previous model, check rollback
6. **Push Registry** - Push model artifacts to GCS
7. **Notify** - Send email notifications

### Success Criteria

- ✅ All validation thresholds met
- ✅ Bias score within acceptable range
- ✅ New model performs better than previous (or rollback disabled)
- ✅ Model artifacts uploaded to GCS
- ✅ Email notification sent

---

## Troubleshooting

### Pipeline Not Triggering

- **Check branch**: Must push to `main` branch
- **Check file paths**: Only triggers when `model-training/**` or `ci-cd/**` files change
- **Check workflow file**: Verify `.github/workflows/model-training-ci-cd.yml` exists

### Authentication Failures

- **Verify `GCP_SA_KEY` secret**: Must be valid JSON
- **Check service account permissions**: Needs BigQuery Data Editor, Storage Admin
- **Verify project ID**: Match `project_id` in config with your GCP project

### Validation Failures

- **Review validation report**: Download artifact from GitHub Actions
- **Adjust thresholds**: Edit `validation_thresholds.yaml` if needed
- **Check evaluation reports**: Ensure model training completed successfully

### Email Not Sending

- **Verify `EMAIL_SMTP_PASSWORD` secret**: Must be set in GitHub Secrets (required)
- **Check email config**: Verify `from_email` and `to_email` in `ci_cd_config.yaml`
- **For Gmail**: Use app password (16 characters), not regular password
- **For local testing**: Set `export EMAIL_SMTP_PASSWORD="your-app-password"`
- **Optional `EMAIL_SMTP_USER`**: If not set, uses `from_email` from config

### Model Training Fails

- **Check logs**: Review error messages in GitHub Actions
- **Verify data pipeline**: Ensure data exists in BigQuery
- **Check GCP resources**: Verify BigQuery dataset and GCS bucket exist

### Model Registry Push Fails

- **Check GCS upload**: Verify artifacts uploaded to GCS successfully
  - Check GCS bucket: `gs://<bucket_name>/models/`
  - Verify files exist with timestamp and commit SHA
- **Check service account permissions**: Needs `Storage Admin` role for GCS
- **Review logs**: Check for specific error messages in push step

**If using Vertex AI (optional):**
- **Check Vertex AI API**: Ensure Vertex AI API is enabled
  ```bash
  gcloud services enable aiplatform.googleapis.com
  ```
- **Check service account permissions**: Needs `Vertex AI User` role
- **Note**: If Vertex AI fails, artifacts are still stored in GCS (which is sufficient)

---

## Implementation Details

### File Structure

```
ci-cd/
├── config/
│   ├── ci_cd_config.yaml          # Main configuration (UPDATE THIS)
│   └── validation_thresholds.yaml # Performance thresholds
├── scripts/
│   ├── run_training_pipeline.py   # Execute training
│   ├── validate_model.py          # Validate performance
│   ├── check_bias.py              # Check bias detection
│   ├── rollback_manager.py        # Compare models & rollback
│   ├── push_to_registry.py        # Push to GCS
│   └── send_notification.py       # Email notifications
└── README.md                       # This file
```

### How It Works

1. **Configuration**: All scripts read from `ci_cd_config.yaml` (env vars can override)
2. **Reuses Existing Code**: Scripts use modules from `model-training/scripts/`
3. **Outputs**: All results saved to `outputs/` directory
4. **Fail Fast**: Pipeline stops if any validation step fails

### Scripts Overview

- **`run_training_pipeline.py`**: Executes the training pipeline (placeholder - extend as needed)
- **`validate_model.py`**: Checks performance metrics against thresholds
- **`check_bias.py`**: Validates bias scores are acceptable
- **`rollback_manager.py`**: Compares new model with previous, blocks if worse
- **`push_to_registry.py`**: Uploads artifacts to GCS (and optionally Vertex AI Model Registry if enabled)
- **`send_notification.py`**: Sends email notifications with pipeline status

### Environment Variables

Scripts read from config file first, but environment variables can override:

**GCP Settings (optional overrides):**
- `GCP_PROJECT_ID` - Overrides `gcp.project_id`
- `REGION` - Overrides `gcp.region`
- `BUCKET_NAME` - Overrides `gcp.bucket_name`
- `BQ_DATASET` - Overrides `gcp.dataset_id`
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account JSON (required for local)

**Email Settings:**
- `EMAIL_SMTP_USER` - Optional, defaults to `from_email` in config
- `EMAIL_SMTP_PASSWORD` - **REQUIRED** (must be set as env var for security, not in config)

### GitHub Actions Workflow

Location: `.github/workflows/model-training-ci-cd.yml`

**Triggers:**
- Push to `main` branch (when `model-training/**` or `ci-cd/**` files change)
- Manual workflow dispatch

**Jobs:**
- Sequential execution with dependencies
- Artifacts passed between jobs
- Failures stop subsequent jobs

### Model Registry

The pipeline uses **GCS (Google Cloud Storage)** for model registry, which provides:
- ✅ Version control (timestamp + commit SHA in paths)
- ✅ Reproducibility (all artifacts stored with metadata)
- ✅ Simple setup (no additional APIs required)

**Model Storage:**
```
gs://<bucket_name>/<base_path>/<timestamp>_<commit-sha>/
├── model-selection/
│   └── model_selection_*.json
├── evaluation/
│   └── model_comparison_*.json
├── bias/
│   └── bias_comparison_*.json
├── best-model-responses/
│   └── **/*.json
└── validation/
    └── *.json
```

**Optional: Vertex AI Model Registry**

If you enable `vertex_ai.enabled: true` in config, the pipeline will also register models in Vertex AI Model Registry for:
- Advanced versioning features
- Integration with Vertex AI services
- Enhanced metadata management

**Note:** Vertex AI is **optional**. GCS alone fulfills all project requirements for version control and reproducibility.

### Rollback Mechanism

- Compares new model's composite score with previous model
- Blocks deployment if new model is worse (unless rollback disabled)
- Previous model metadata loaded from GCS
- Rollback threshold configurable in `ci_cd_config.yaml`

---

## Quick Reference

### Key Files
- **Config**: `ci-cd/config/ci_cd_config.yaml` ← **Update this for your setup**
- **Workflow**: `.github/workflows/model-training-ci-cd.yml`
- **Thresholds**: `ci-cd/config/validation_thresholds.yaml`

### Key Commands
```bash
# Push to trigger pipeline
git push origin main

# Test validation locally
python ci-cd/scripts/validate_model.py

# Test bias check locally
python ci-cd/scripts/check_bias.py
```

### Important URLs
- **GitHub Actions**: `https://github.com/<org>/<repo>/actions`
- **GCP Console**: `https://console.cloud.google.com/`
- **GCS Models**: `gs://<your-bucket>/models/`
- **Vertex AI Model Registry** (optional): `https://console.cloud.google.com/vertex-ai/models`

---

## Next Steps

1. ✅ Update `ci-cd/config/ci_cd_config.yaml` with your GCP settings
2. ✅ Set GitHub Secrets (`GCP_SA_KEY`, `EMAIL_SMTP_USER`, `EMAIL_SMTP_PASSWORD`)
3. ✅ Verify GCP resources exist (BigQuery dataset, GCS bucket)
4. ✅ Verify service account has required roles (Storage Admin, BigQuery Data Editor)
5. ✅ (Optional) Enable Vertex AI API if using `vertex_ai.enabled: true`
6. ✅ Push to `main` branch to trigger pipeline
7. ✅ Monitor pipeline in GitHub Actions tab
8. ✅ Check email for notifications
9. ✅ Verify model artifacts in GCS: `gs://<your-bucket>/models/`

---

**Need Help?** Check the troubleshooting section above or review the script logs in GitHub Actions.
