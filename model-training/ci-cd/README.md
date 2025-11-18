# CI/CD Pipeline for Model Training - Setup Guide

This directory contains the complete CI/CD automation for the model training pipeline. This guide provides step-by-step instructions to set up and configure the entire CI/CD system.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [GCP Setup](#gcp-setup)
4. [GitHub Secrets Configuration](#github-secrets-configuration)
5. [Email Configuration](#email-configuration)
6. [Local Testing](#local-testing)
7. [Pipeline Workflow](#pipeline-workflow)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

---

## 🎯 Overview

The CI/CD pipeline automates the following:

1. **Model Training**: Triggers model training when code changes are pushed
2. **Model Validation**: Validates model metrics against quality thresholds
3. **Bias Detection**: Checks for bias and blocks deployment if thresholds exceeded
4. **Model Comparison**: Compares new model with previous version
5. **Model Registry**: Pushes validated models to GCP Artifact Registry and Vertex AI Model Registry
6. **Deployment**: Deploys models to production (optional, manual approval recommended)
7. **Notifications**: Sends email notifications for pipeline events
8. **Rollback**: Implements rollback mechanism if new model performs worse

### Pipeline Flow

```
Code Push → Training → Validation → Bias Check → Comparison → Registry → Deployment → Notification
```

---

## 📦 Prerequisites

Before setting up the CI/CD pipeline, ensure you have:

1. **GitHub Repository**: Your code must be in a GitHub repository
2. **GCP Account**: Active Google Cloud Platform account with billing enabled
3. **GCP Project**: Existing GCP project (default: `datacraft-data-pipeline`)
4. **Python 3.10+**: For local testing and development
5. **gcloud CLI**: Installed and configured (for GCP setup)
6. **Email Account**: For sending notifications (Gmail recommended)

---

## ☁️ GCP Setup

### Step 1: Create Service Account

1. Go to [GCP Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** → **Service Accounts**
3. Click **Create Service Account**
4. Fill in details:
   - **Name**: `mlops-ci-cd-service-account`
   - **Description**: `Service account for CI/CD pipeline automation`
5. Click **Create and Continue**

### Step 2: Grant Required Permissions

Grant the following roles to the service account:

- **Storage Admin** (`roles/storage.admin`) - For GCS bucket access
- **BigQuery Data Editor** (`roles/bigquery.dataEditor`) - For BigQuery access
- **AI Platform Admin** (`roles/aiplatform.admin`) - For Model Registry
- **Artifact Registry Writer** (`roles/artifactregistry.writer`) - For Artifact Registry
- **Vertex AI User** (`roles/aiplatform.user`) - For Vertex AI endpoints

**Commands:**
```bash
PROJECT_ID="datacraft-data-pipeline"
SA_EMAIL="mlops-ci-cd-service-account@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user"
```

### Step 3: Create Service Account Key

1. In the service account details page, go to **Keys** tab
2. Click **Add Key** → **Create New Key**
3. Select **JSON** format
4. Download the key file (e.g., `mlops-ci-cd-key.json`)
5. **IMPORTANT**: Keep this file secure! You'll need it for GitHub Secrets

### Step 4: Create Artifact Registry Repository

```bash
PROJECT_ID="datacraft-data-pipeline"
REGION="us-central1"
REPO_NAME="model-registry"

gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=generic \
  --location=${REGION} \
  --project=${PROJECT_ID} \
  --description="Model registry for MLOps pipeline"
```

### Step 5: Verify GCS Bucket Exists

```bash
BUCKET_NAME="isha-retail-data"
PROJECT_ID="datacraft-data-pipeline"

# Check if bucket exists
gsutil ls -b gs://${BUCKET_NAME}

# If it doesn't exist, create it:
gsutil mb -p ${PROJECT_ID} -l us-central1 gs://${BUCKET_NAME}
```

### Step 6: Enable Required APIs

```bash
PROJECT_ID="datacraft-data-pipeline"

gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  storage-api.googleapis.com \
  bigquery.googleapis.com \
  --project=${PROJECT_ID}
```

---

## 🔐 GitHub Secrets Configuration

### Step 1: Access GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

### Step 2: Add Required Secrets

Add the following secrets one by one:

#### 1. `GCP_SA_KEY` (Required)
- **Value**: Contents of the service account JSON key file
- **How to get**: Open the downloaded JSON file, copy entire contents
- **Example**: `{"type": "service_account", "project_id": "...", ...}`

#### 2. `GCP_PROJECT_ID` (Optional, has default)
- **Value**: Your GCP project ID
- **Default**: `datacraft-data-pipeline`
- **Example**: `datacraft-data-pipeline`

#### 3. `GCP_REGION` (Optional, has default)
- **Value**: GCP region
- **Default**: `us-central1`
- **Example**: `us-central1`

#### 4. `EMAIL_SENDER` (Required for notifications)
- **Value**: Email address to send notifications from
- **Example**: `mlops0242@gmail.com`

#### 5. `EMAIL_SENDER_PASSWORD` (Required for notifications)
- **Value**: Email password or app password
- **For Gmail**: Use [App Password](https://support.google.com/accounts/answer/185833)
- **How to create Gmail App Password**:
  1. Go to Google Account settings
  2. Security → 2-Step Verification → App passwords
  3. Generate app password for "Mail"
  4. Use the 16-character password (no spaces)

#### 6. `EMAIL_RECIPIENTS` (Required for notifications)
- **Value**: Comma-separated list of recipient emails
- **Example**: `mlops0242@gmail.com,team@example.com`

#### 7. `SMTP_SERVER` (Optional, has default)
- **Value**: SMTP server address
- **Default**: `smtp.gmail.com`
- **Example**: `smtp.gmail.com`

#### 8. `SMTP_PORT` (Optional, has default)
- **Value**: SMTP port number
- **Default**: `587`
- **Example**: `587`

### Step 3: Verify Secrets

After adding all secrets, verify they appear in the secrets list. They should show as `●●●●●●●●` (hidden).

---

## 📧 Email Configuration

### Gmail Setup (Recommended)

1. **Enable 2-Step Verification**:
   - Go to [Google Account Security](https://myaccount.google.com/security)
   - Enable 2-Step Verification if not already enabled

2. **Generate App Password**:
   - Go to [App Passwords](https://myaccount.google.com/apppasswords)
   - Select "Mail" and your device
   - Generate and copy the 16-character password
   - Use this password in `EMAIL_SENDER_PASSWORD` secret

3. **Test Email Configuration**:
   ```bash
   export EMAIL_SENDER="your-email@gmail.com"
   export EMAIL_SENDER_PASSWORD="your-app-password"
   export EMAIL_RECIPIENTS="recipient@example.com"
   
   python model-training/ci-cd/scripts/send_notifications.py success outputs/model-training/pipeline_summary.json
   ```

### Other Email Providers

For other email providers, update `SMTP_SERVER` and `SMTP_PORT`:

- **Outlook**: `smtp-mail.outlook.com:587`
- **Yahoo**: `smtp.mail.yahoo.com:587`
- **Custom SMTP**: Use your organization's SMTP server

---

## 🧪 Local Testing

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
pip install pyyaml google-cloud-aiplatform
```

### Step 2: Set Environment Variables

```bash
export GCP_PROJECT_ID="datacraft-data-pipeline"
export GCP_REGION="us-central1"
export EMAIL_SENDER="your-email@gmail.com"
export EMAIL_SENDER_PASSWORD="your-app-password"
export EMAIL_RECIPIENTS="recipient@example.com"
```

### Step 3: Authenticate with GCP

```bash
# Option 1: Use service account key file
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Option 2: Use gcloud auth
gcloud auth application-default login
```

### Step 4: Test Individual Scripts

```bash
# Test validation script
python model-training/ci-cd/scripts/validate_model_metrics.py

# Test bias check script
python model-training/ci-cd/scripts/check_bias_thresholds.py

# Test comparison script
python model-training/ci-cd/scripts/compare_models.py

# Test notification script
python model-training/ci-cd/scripts/send_notifications.py success outputs/model-training/pipeline_summary.json
```

### Step 5: Test Full Pipeline (Optional)

```bash
# Run training pipeline directly
python model-training/ci-cd/scripts/run_training_pipeline.py --direct
```

---

## 🔄 Pipeline Workflow

### Automatic Triggers

The pipeline automatically triggers on:

1. **Changes to `model-training/data/user_queries.txt`** - The pipeline runs only when the user queries file is modified. This ensures:
   - Models are re-evaluated with updated queries
   - Training runs only when evaluation criteria change
   - Avoids unnecessary runs when code/config changes
2. **Pull requests affecting `user_queries.txt`** - Same as above
3. **Manual trigger** (via GitHub Actions UI) - Can be triggered manually regardless of file changes
4. **Workflow file changes** - Also triggers when the workflow file itself is modified (for testing)

**Note**: The pipeline is intentionally configured to run only when `user_queries.txt` changes. This is because:
- User queries define the evaluation dataset
- Changing queries requires re-evaluation of all models
- Code/config changes don't require retraining unless queries change
- This reduces unnecessary pipeline runs and costs

### Pipeline Steps

1. **Training** (`run_training_pipeline.py`)
   - Runs model training pipeline
   - Evaluates multiple models
   - Selects best model

2. **Validation** (`validate_model_metrics.py`)
   - Checks composite score ≥ 70.0
   - Checks performance score ≥ 75.0
   - Checks bias score ≤ 40.0
   - Checks success rate ≥ 80.0
   - Checks overall accuracy ≥ 75.0

3. **Bias Check** (`check_bias_thresholds.py`)
   - Checks bias score ≤ 50.0
   - Checks severity ≠ HIGH
   - Checks number of biases ≤ 2

4. **Comparison** (`compare_models.py`)
   - Compares with previous model
   - Blocks if performance degrades > 5%

5. **Registry Push** (`push_to_registry.py`)
   - Uploads artifacts to GCS
   - Pushes to Vertex AI Model Registry
   - Pushes to Artifact Registry (optional)

6. **Deployment** (`deploy_model.py`)
   - Deploys to Vertex AI endpoint
   - Requires manual approval (auto-deploy disabled by default)

7. **Notifications** (`send_notifications.py`)
   - Sends email on success/failure
   - Includes detailed metrics

### Configuration File

Edit `model-training/ci-cd/config/ci_cd_config.yaml` to adjust:

- **Validation thresholds**: Minimum scores required
- **Bias thresholds**: Maximum bias allowed
- **Rollback settings**: Performance degradation tolerance
- **Email settings**: SMTP configuration
- **Deployment settings**: Auto-deploy enabled/disabled

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Authentication failed" Error

**Problem**: GCP authentication fails in GitHub Actions

**Solution**:
- Verify `GCP_SA_KEY` secret contains valid JSON
- Check service account has required permissions
- Ensure service account key is not expired

#### 2. "Email sending failed" Error

**Problem**: Email notifications not sending

**Solution**:
- Verify `EMAIL_SENDER_PASSWORD` is an app password (not regular password)
- Check 2-Step Verification is enabled (for Gmail)
- Test email configuration locally first
- Verify `EMAIL_RECIPIENTS` is comma-separated

#### 3. "Model validation failed" Error

**Problem**: Model doesn't meet quality thresholds

**Solution**:
- Review validation results in GitHub Actions logs
- Adjust thresholds in `ci_cd_config.yaml` if needed
- Improve model training to meet thresholds
- Check if data quality issues exist

#### 4. "Bias check failed" Error

**Problem**: Bias exceeds acceptable limits

**Solution**:
- Review bias detection report
- Adjust model training to reduce bias
- Review data for potential bias sources
- Consider adjusting bias thresholds (not recommended)

#### 5. "Artifact Registry not found" Error

**Problem**: Artifact Registry repository doesn't exist

**Solution**:
```bash
# Create repository
gcloud artifacts repositories create model-registry \
  --repository-format=generic \
  --location=us-central1 \
  --project=datacraft-data-pipeline
```

#### 6. "Previous model not found" Warning

**Problem**: Comparison script can't find previous model

**Solution**:
- This is normal for first run
- Ensure GCS bucket has previous model metadata
- Check bucket permissions

### Debug Mode

Enable debug logging:

```bash
export PYTHONUNBUFFERED=1
export LOG_LEVEL=DEBUG
python model-training/ci-cd/scripts/validate_model_metrics.py
```

### View GitHub Actions Logs

1. Go to GitHub repository
2. Click **Actions** tab
3. Select the workflow run
4. Click on individual steps to view logs

---

## 🔧 Maintenance

### Updating Thresholds

Edit `model-training/ci-cd/config/ci_cd_config.yaml`:

```yaml
validation:
  thresholds:
    min_composite_score: 75.0  # Increase for stricter validation
    max_bias_score: 35.0        # Decrease for stricter bias control
```

### Adding New Models

Edit `model-training/ci-cd/config/ci_cd_config.yaml`:

```yaml
training:
  models_to_evaluate:
    - "gemini-2.5-flash"
    - "gemini-2.5-pro"
    - "gemini-2.0-flash"  # Add new model
```

### Monitoring Pipeline

1. **GitHub Actions**: View runs in Actions tab
2. **GCS**: Check `gs://isha-retail-data/best_model_responses/`
3. **Vertex AI**: View models in [Vertex AI Console](https://console.cloud.google.com/vertex-ai)
4. **Email**: Check notification emails for pipeline status

### Rollback Procedure

If a bad model is deployed:

1. **Automatic Rollback** (if enabled):
   - Pipeline will automatically rollback if comparison fails

2. **Manual Rollback**:
   ```bash
   # Get previous model resource name from GCS
   gsutil ls gs://isha-retail-data/best_model_responses/
   
   # Deploy previous model
   python model-training/ci-cd/scripts/deploy_model.py <previous_model_resource_name>
   ```

### Updating Dependencies

```bash
# Update requirements.txt
pip freeze > requirements.txt

# Test locally
python model-training/ci-cd/scripts/validate_model_metrics.py
```

---

## 📊 Pipeline Outputs

### Artifacts Generated

1. **Model Metrics**: `outputs/model-training/pipeline_summary.json`
2. **Best Model Responses**: `outputs/best-model-responses/`
3. **Evaluation Reports**: `outputs/evaluation/`
4. **Bias Reports**: `outputs/bias/`
5. **Selection Reports**: `outputs/model-selection/`

### GCS Storage

- **Path**: `gs://isha-retail-data/best_model_responses/{timestamp}_{model_name}/`
- **Contents**: All model artifacts, metadata, and reports

### Vertex AI Model Registry

- **Location**: [Vertex AI Console](https://console.cloud.google.com/vertex-ai/models)
- **Format**: `gemini-query-model-{timestamp}`

---

## 🔒 Security Best Practices

1. **Never commit service account keys** to repository
2. **Use GitHub Secrets** for all sensitive data
3. **Rotate service account keys** periodically
4. **Use least privilege** for service account permissions
5. **Enable audit logging** in GCP
6. **Review access logs** regularly

---

## 📞 Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review GitHub Actions logs
3. Check GCP Console for errors
4. Verify all secrets are configured correctly

---

## 📝 Checklist for Setup

Use this checklist to ensure complete setup:

- [ ] GCP project created and billing enabled
- [ ] Service account created with required roles
- [ ] Service account key downloaded
- [ ] Artifact Registry repository created
- [ ] GCS bucket exists and accessible
- [ ] Required APIs enabled
- [ ] GitHub Secrets configured (all 8 secrets)
- [ ] Email app password generated (for Gmail)
- [ ] Local testing completed
- [ ] First pipeline run successful
- [ ] Email notifications working
- [ ] Model registry push successful

---

## 🎉 Next Steps

After setup is complete:

1. **Trigger first pipeline run**:
   - Push a change to `model-training/` directory, or
   - Go to Actions → Workflows → Run workflow

2. **Monitor first run**:
   - Watch GitHub Actions logs
   - Check email notifications
   - Verify artifacts in GCS

3. **Adjust thresholds** (if needed):
   - Edit `ci_cd_config.yaml`
   - Commit and push changes

4. **Enable auto-deployment** (optional):
   - Set `deployment.vertex_ai.auto_deploy: true` in config
   - **Warning**: Only enable after thorough testing

---

**Last Updated**: 2025-01-27  
**Version**: 1.0.0

