# Complete Setup and Testing Guide for CI/CD Pipeline

This guide provides step-by-step instructions to set up GCP, Airflow, GitHub, and test the entire CI/CD pipeline locally to gather screenshots for your report.

---

## Table of Contents

1. [GCP Setup](#1-gcp-setup)
2. [Airflow Setup](#2-airflow-setup)
3. [GitHub Setup](#3-github-setup)
4. [Local Testing](#4-local-testing)
5. [Screenshot Guide](#5-screenshot-guide)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. GCP Setup

### Step 1.1: Create GCP Project (if not exists)

1. Go to [GCP Console](https://console.cloud.google.com/)
2. Click **"Select a project"** → **"New Project"**
3. Enter project name: `datacraft-478300` (or your preferred name)
4. Note your **Project ID** (e.g., `datacraft-478300`)
5. Click **"Create"**

### Step 1.2: Enable Required APIs

1. Go to **APIs & Services** → **Library**
2. Enable these APIs:
   - **BigQuery API**
   - **Cloud Storage API**
   - **Vertex AI API** (for Gemini models)
   - **Cloud Resource Manager API**

**Quick command (if using gcloud CLI):**
```bash
gcloud services enable \
  bigquery.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project=datacraft-478300
```

### Step 1.3: Create Service Account

1. Go to **IAM & Admin** → **Service Accounts**
2. Click **"Create Service Account"**
3. Enter details:
   - **Name:** `mlops-ci-cd-service-account`
   - **Description:** `Service account for CI/CD pipeline`
4. Click **"Create and Continue"**
5. Grant these roles:
   - **BigQuery Data Editor**
   - **BigQuery Job User**
   - **Storage Admin**
   - **Vertex AI User**
6. Click **"Continue"** → **"Done"**

### Step 1.4: Create Service Account Key

1. Click on the service account you just created
2. Go to **"Keys"** tab
3. Click **"Add Key"** → **"Create new key"**
4. Select **JSON** format
5. Click **"Create"** - Key will download automatically
6. **Save this file securely** - you'll need it for:
   - Airflow setup (place in `gcp/service-account.json`)
   - GitHub Secrets (copy entire JSON content)

### Step 1.5: Create BigQuery Dataset

1. Go to **BigQuery** → **SQL Workspace**
2. Click on your project name
3. Click **"⋮"** (three dots) → **"Create dataset"**
4. Enter:
   - **Dataset ID:** `datacraft_ml`
   - **Location type:** `Multi-region` or `US (multiple regions)`
5. Click **"Create dataset"**

**Verify:**
```bash
# Using gcloud CLI
gcloud alpha bq datasets create datacraft_ml \
  --location=US \
  --project=datacraft-478300
```

### Step 1.6: Create GCS Bucket

1. Go to **Cloud Storage** → **Buckets**
2. Click **"Create Bucket"**
3. Enter:
   - **Name:** `model-datacraft` (must be globally unique)
   - **Location type:** `Region` → `us-east1` (or your preferred region)
   - **Storage class:** `Standard`
   - **Access control:** `Uniform`
4. Click **"Create"**

**Verify:**
```bash
# Using gcloud CLI
gsutil mb -p datacraft-478300 -c STANDARD -l us-east1 gs://model-datacraft
```

### Step 1.7: Upload Data to GCS

The DAG expects data at: `gs://model-datacraft/data/validated/orders_validated.csv`

**Option A: Upload via Console**
1. Go to **Cloud Storage** → **Buckets** → `model-datacraft`
2. Create folder structure:
   - Click **"Create Folder"** → `data`
   - Inside `data`, create `validated`
3. Upload your CSV file:
   - Navigate to `data/validated/`
   - Click **"Upload Files"**
   - Select your CSV file
   - Rename it to: `orders_validated.csv`

**Option B: Upload via gsutil**
```bash
# Create folder structure
gsutil -m cp data-pipeline/data/validated/orders_validated.csv \
  gs://model-datacraft/data/validated/orders_validated.csv

# Verify upload
gsutil ls gs://model-datacraft/data/validated/
```

**Note:** If you don't have validated data, you can use the processed data:
```bash
# The DAG can also load from processed stage
gsutil -m cp data-pipeline/data/processed/orders_processed.csv \
  gs://model-datacraft/data/processed/orders_processed.csv
```

### Step 1.8: Verify GCP Setup

Check that everything is set up correctly:

```bash
# Verify service account exists
gcloud iam service-accounts list --project=datacraft-478300

# Verify BigQuery dataset exists
bq ls --project_id=datacraft-478300

# Verify GCS bucket exists
gsutil ls -p datacraft-478300

# Verify data file exists
gsutil ls gs://model-datacraft/data/validated/
```

---

## 2. Airflow Setup

### Step 2.1: Prepare Service Account Key

1. Copy your downloaded service account JSON key
2. Create directory in project root:
   ```bash
   mkdir -p gcp
   ```
3. Place the key file:
   ```bash
   cp /path/to/downloaded-key.json gcp/service-account.json
   ```
4. **Verify file exists:**
   ```bash
   ls -la gcp/service-account.json
   ```

### Step 2.2: Update Configuration Files

**Update `ci-cd/config/ci_cd_config.yaml`:**
```yaml
gcp:
  project_id: "datacraft-478300"  # Your actual project ID
  region: "us-east1"              # Your GCS bucket region
  dataset_id: "datacraft_ml"      # Your BigQuery dataset
  bucket_name: "model-datacraft"  # Your GCS bucket name

notifications:
  email:
    from_email: "your-email@gmail.com"  # Your email
    to_email: "your-email@gmail.com"    # Your email

model_registry:
  base_path: "models"

rollback:
  enabled: true
  min_improvement_threshold: 0.0
```

**Update `docker-compose.yml` environment variables:**
```yaml
environment:
  - GCP_PROJECT_ID=datacraft-478300  # Match your project ID
```

### Step 2.3: Build and Start Airflow

1. **Build Docker images:**
   ```bash
   cd /Users/sanskar/Personal/mlops-project
   docker-compose build
   ```

2. **Start Airflow services:**
   ```bash
   docker-compose up -d
   ```

3. **Wait for initialization (60-90 seconds):**
   ```bash
   # Check logs
   docker-compose logs -f airflow-webserver
   
   # Look for: "Airflow webserver is ready"
   ```

4. **Verify containers are running:**
   ```bash
   docker ps
   # Should see: airflow-webserver, airflow-scheduler, airflow-postgres
   ```

### Step 2.4: Access Airflow UI

1. Open browser: **http://localhost:8080**
2. Login:
   - **Username:** `admin`
   - **Password:** `admin`

### Step 2.5: Configure Airflow Variables

The DAG uses Airflow Variables. Set them in the UI:

1. Go to **Admin** → **Variables**
2. Click **"+"** to add each variable:

   **Variable 1:**
   - **Key:** `BUCKET_NAME`
   - **Val:** `model-datacraft` (your bucket name)
   - **Description:** `GCS bucket name for model artifacts`

   **Variable 2:**
   - **Key:** `REGION`
   - **Val:** `us-east1` (your region)
   - **Description:** `GCP region`

   **Variable 3:**
   - **Key:** `BQ_DATASET`
   - **Val:** `datacraft_ml` (your dataset ID)
   - **Description:** `BigQuery dataset ID`

**Alternative: Set via CLI (if Airflow CLI is accessible):**
```bash
docker exec -it airflow-webserver airflow variables set BUCKET_NAME model-datacraft
docker exec -it airflow-webserver airflow variables set REGION us-east1
docker exec -it airflow-webserver airflow variables set BQ_DATASET datacraft_ml
```

### Step 2.6: Verify DAG is Visible

1. In Airflow UI, go to **DAGs** page
2. Look for: **`model_pipeline_with_evaluation`**
3. DAG should show as **"No status"** (not yet run)
4. If DAG is not visible:
   - Check logs: `docker-compose logs airflow-scheduler`
   - Verify DAG file exists: `ls model-training/dags/model_pipeline_dag.py`
   - Check for import errors in DAG

### Step 2.7: Test DAG Execution (First Run)

1. **Enable the DAG:**
   - Toggle the switch next to `model_pipeline_with_evaluation`

2. **Trigger the DAG:**
   - Click on DAG name
   - Click **"Trigger DAG"** button (▶️ icon)
   - Confirm

3. **Monitor execution:**
   - Watch tasks turn green (success) or red (failed)
   - Click on individual tasks to see logs
   - Expected duration: **5-10 minutes**

4. **Verify outputs:**
   ```bash
   # Check local outputs
   ls -la outputs/best-model-responses/
   
   # Check GCS uploads
   gsutil ls -r gs://model-datacraft/best_model_responses/
   ```

**Screenshot Opportunity:** DAG execution graph showing all tasks completed

---

## 3. GitHub Setup

### Step 3.1: Repository Setup

1. **Ensure repository is on GitHub:**
   ```bash
   git remote -v
   # Should show your GitHub repository URL
   ```

2. **Verify branch:**
   ```bash
   git branch
   # Should be on main or your working branch
   ```

3. **Push code (if not already pushed):**
   ```bash
   git add .
   git commit -m "Setup CI/CD pipeline"
   git push origin main
   ```

### Step 3.2: Configure GitHub Secrets

Go to: **Repository → Settings → Secrets and variables → Actions**

**Add these secrets:**

1. **`GCP_SA_KEY`** (Required)
   - Click **"New repository secret"**
   - **Name:** `GCP_SA_KEY`
   - **Secret:** Paste entire JSON content from `gcp/service-account.json`
   - Click **"Add secret"**

2. **`EMAIL_SMTP_PASSWORD`** (Required for notifications)
   - **Name:** `EMAIL_SMTP_PASSWORD`
   - **Secret:** Your Gmail app password (16 characters)
   - How to get:
     1. Go to [Google Account → Security](https://myaccount.google.com/security)
     2. Enable **2-Step Verification** (if not enabled)
     3. Go to **App Passwords**
     4. Generate password for "Mail"
     5. Copy the 16-character password

3. **`EMAIL_SMTP_USER`** (Optional)
   - **Name:** `EMAIL_SMTP_USER`
   - **Secret:** Your email address
   - If not set, uses `from_email` from config

4. **`AIRFLOW_URL`** (Required)
   - **Name:** `AIRFLOW_URL`
   - **Secret:** `http://localhost:8080` (for local testing)
   - **Note:** For production, use your Airflow URL (Cloud Composer or exposed endpoint)

5. **`AIRFLOW_USERNAME`** (Required)
   - **Name:** `AIRFLOW_USERNAME`
   - **Secret:** `admin`

6. **`AIRFLOW_PASSWORD`** (Required)
   - **Name:** `AIRFLOW_PASSWORD`
   - **Secret:** `admin`

**Important Note:** For GitHub Actions to access local Airflow, you need to:
- Use a tunnel service (ngrok, localtunnel) to expose localhost:8080
- OR use Cloud Composer (managed Airflow)
- OR set up Airflow on a server accessible from GitHub Actions

**For local testing screenshots, you can:**
- Test CI/CD scripts locally (see Section 4)
- Mock the DAG trigger step
- Use GitHub Actions with a tunnel

### Step 3.3: Verify Workflow File

Check that `.github/workflows/model-training-ci-cd.yml` exists and is correct:

```bash
cat .github/workflows/model-training-ci-cd.yml
```

---

## 4. Local Testing

### Step 4.1: Test DAG Execution Locally

**Prerequisites:**
- Airflow is running (Section 2)
- DAG has been triggered at least once
- Outputs exist in GCS

**Test Steps:**

1. **Trigger DAG in Airflow UI:**
   - Go to http://localhost:8080
   - Trigger `model_pipeline_with_evaluation`
   - Wait for completion

2. **Verify outputs in GCS:**
   ```bash
   gsutil ls -r gs://model-datacraft/best_model_responses/
   # Should see: best_model_responses/YYYYMMDD_HHMMSS_gemini-2.5-flash/
   ```

3. **Verify outputs locally:**
   ```bash
   ls -la outputs/best-model-responses/
   # Should see model run directories
   ```

**Screenshot Opportunities:**
- Airflow DAG execution graph
- Task logs showing successful completion
- GCS bucket showing uploaded files

### Step 4.2: Test CI/CD Scripts Locally

**Set environment variables:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="gcp/service-account.json"
export GCP_PROJECT_ID="datacraft-478300"
export EMAIL_SMTP_PASSWORD="your-app-password"
```

**Test 1: Download Outputs**
```bash
python ci-cd/scripts/download_outputs.py
```
**Expected:** Downloads latest model outputs from GCS to `outputs/best-model-responses/`

**Screenshot:** Terminal showing download progress

**Test 2: Validate Model**
```bash
python ci-cd/scripts/validate_model.py
```
**Expected:** Validates performance thresholds, shows PASSED or FAILED

**Screenshot:** Terminal showing validation results

**Test 3: Check Bias**
```bash
python ci-cd/scripts/check_bias.py
```
**Expected:** Validates bias thresholds, shows PASSED or FAILED

**Screenshot:** Terminal showing bias check results

**Test 4: Rollback Check**
```bash
python ci-cd/scripts/rollback_manager.py
```
**Expected:** Compares with previous model (or shows "first deployment")

**Screenshot:** Terminal showing comparison results

**Test 5: Push to Registry**
```bash
# Get commit SHA
COMMIT_SHA=$(git rev-parse --short HEAD)

python ci-cd/scripts/push_to_registry.py --commit-sha $COMMIT_SHA
```
**Expected:** Uploads artifacts to `gs://model-datacraft/models/{timestamp}_{commit}/`

**Screenshot:** Terminal showing upload progress

**Test 6: Send Notification**
```bash
python ci-cd/scripts/send_notification.py --status success
```
**Expected:** Sends email notification

**Screenshot:** Email received in inbox

### Step 4.3: Test Full CI/CD Flow Locally

**Simulate the complete pipeline:**

```bash
# 1. Set environment
export GOOGLE_APPLICATION_CREDENTIALS="gcp/service-account.json"
export GCP_PROJECT_ID="datacraft-478300"
export EMAIL_SMTP_PASSWORD="your-app-password"

# 2. Download outputs (simulates CI/CD download step)
python ci-cd/scripts/download_outputs.py

# 3. Validate (simulates CI/CD validate job)
python ci-cd/scripts/validate_model.py || echo "Validation failed"

# 4. Check bias (simulates CI/CD bias check job)
python ci-cd/scripts/check_bias.py || echo "Bias check failed"

# 5. Compare and deploy (simulates CI/CD deploy job)
COMMIT_SHA=$(git rev-parse --short HEAD)
python ci-cd/scripts/rollback_manager.py && \
python ci-cd/scripts/push_to_registry.py --commit-sha $COMMIT_SHA

# 6. Send notification (simulates CI/CD notify job)
python ci-cd/scripts/send_notification.py --status success
```

**Screenshot:** Terminal showing complete pipeline execution

### Step 4.4: Test DAG Trigger Script (Optional)

If you want to test the DAG trigger script locally:

**Note:** This requires Airflow to be accessible. For local testing, you can mock this.

```bash
export AIRFLOW_URL="http://localhost:8080"
export AIRFLOW_DAG_ID="model_pipeline_with_evaluation"
export AIRFLOW_USERNAME="admin"
export AIRFLOW_PASSWORD="admin"

python ci-cd/scripts/trigger_dag.py
```

**Screenshot:** Terminal showing DAG trigger and wait status

---

## 5. Screenshot Guide

### Screenshot Checklist for Report

Capture these screenshots in order:

#### Phase 1: Setup Verification

1. **✅ GCP Console - Service Account**
   - Screenshot: Service account with required roles
   - Caption: "Service account configured with BigQuery and Storage permissions"

2. **✅ GCP Console - BigQuery Dataset**
   - Screenshot: `datacraft_ml` dataset visible
   - Caption: "BigQuery dataset created for model training"

3. **✅ GCP Console - GCS Bucket**
   - Screenshot: Bucket `model-datacraft` with data files
   - Caption: "GCS bucket with validated data ready for training"

4. **✅ Airflow UI - DAG List**
   - Screenshot: `model_pipeline_with_evaluation` DAG visible
   - Caption: "Model training DAG loaded in Airflow"

#### Phase 2: DAG Execution

5. **✅ Airflow UI - DAG Execution Graph**
   - Screenshot: All tasks showing green (success)
   - Caption: "Complete DAG execution with all tasks successful"

6. **✅ Airflow UI - Task Logs**
   - Screenshot: Task logs showing "Best model selected: gemini-2.5-flash"
   - Caption: "DAG successfully selected best model"

7. **✅ GCS Console - Model Outputs**
   - Screenshot: `best_model_responses/{timestamp}_{model}/` folder
   - Caption: "Model outputs saved to GCS by DAG"

#### Phase 3: CI/CD Execution (Local Testing)

8. **✅ Terminal - Download Outputs**
   - Screenshot: `download_outputs.py` execution
   - Caption: "CI/CD downloading model outputs from GCS"

9. **✅ Terminal - Validation**
   - Screenshot: `validate_model.py` showing "VALIDATION PASSED"
   - Caption: "Model validation against quality thresholds"

10. **✅ Terminal - Bias Check**
    - Screenshot: `check_bias.py` showing "BIAS CHECK PASSED"
    - Caption: "Bias validation confirming acceptable scores"

11. **✅ Terminal - Rollback Check**
    - Screenshot: `rollback_manager.py` showing comparison results
    - Caption: "Model comparison with previous deployment"

12. **✅ Terminal - Push to Registry**
    - Screenshot: `push_to_registry.py` showing upload progress
    - Caption: "Model artifacts pushed to production registry"

13. **✅ GCS Console - Production Registry**
    - Screenshot: `models/{timestamp}_{commit}/` folder
    - Caption: "Production model registry with versioned artifacts"

14. **✅ Email Notification**
    - Screenshot: Email received with pipeline status
    - Caption: "Email notification sent with pipeline results"

#### Phase 4: GitHub Actions (If Testing)

15. **✅ GitHub Actions - Workflow Run**
    - Screenshot: Workflow run list showing triggered pipeline
    - Caption: "CI/CD pipeline triggered on code push"

16. **✅ GitHub Actions - Job Status**
    - Screenshot: All jobs showing green checkmarks
    - Caption: "All CI/CD jobs completed successfully"

17. **✅ GitHub Actions - Artifacts**
    - Screenshot: Artifacts section showing downloadable files
    - Caption: "Training outputs available as artifacts"

### Screenshot Tips

- **Use clear, descriptive filenames:** `01_gcp_service_account.png`
- **Capture full context:** Include UI elements, URLs, timestamps
- **Highlight key information:** Use arrows or annotations
- **Show before/after:** Capture state before and after operations
- **Include error messages:** If something fails, screenshot the error

---

## 6. Troubleshooting

### Issue: DAG Not Visible in Airflow

**Symptoms:** DAG doesn't appear in Airflow UI

**Solutions:**
1. Check DAG file exists: `ls model-training/dags/model_pipeline_dag.py`
2. Check for syntax errors:
   ```bash
   docker exec -it airflow-scheduler python -m py_compile /opt/airflow/dags/model-training/model_pipeline_dag.py
   ```
3. Check scheduler logs:
   ```bash
   docker-compose logs airflow-scheduler | grep -i error
   ```
4. Verify DAG folder is mounted correctly in `docker-compose.yml`

### Issue: DAG Fails with "Permission Denied"

**Symptoms:** Tasks fail with GCP permission errors

**Solutions:**
1. Verify service account key exists:
   ```bash
   ls -la gcp/service-account.json
   ```
2. Verify service account has required roles
3. Check credentials in Airflow:
   ```bash
   docker exec -it airflow-webserver cat /opt/airflow/gcp/service-account.json | head -5
   ```
4. Regenerate service account key if needed

### Issue: "Dataset not found" Error

**Symptoms:** DAG fails when loading data from GCS

**Solutions:**
1. Verify data file exists:
   ```bash
   gsutil ls gs://model-datacraft/data/validated/orders_validated.csv
   ```
2. Check file name matches exactly: `orders_validated.csv`
3. Verify bucket name in Airflow Variable matches actual bucket
4. Check service account has Storage Object Viewer role

### Issue: BigQuery Table Creation Fails

**Symptoms:** Error creating table in BigQuery

**Solutions:**
1. Verify dataset exists:
   ```bash
   bq ls --project_id=datacraft-478300
   ```
2. Check service account has BigQuery Data Editor role
3. Verify dataset location matches bucket region

### Issue: CI/CD Scripts Can't Find Outputs

**Symptoms:** `download_outputs.py` says "No model outputs found"

**Solutions:**
1. Verify DAG has run and completed successfully
2. Check GCS path:
   ```bash
   gsutil ls -r gs://model-datacraft/best_model_responses/
   ```
3. Verify bucket name in `ci_cd_config.yaml` matches actual bucket
4. Check service account has Storage Object Viewer role

### Issue: Validation Fails

**Symptoms:** `validate_model.py` fails with threshold errors

**Solutions:**
1. Check validation report:
   ```bash
   cat outputs/validation/validation_report.json
   ```
2. Review which thresholds failed
3. Adjust thresholds in `validation_thresholds.yaml` if needed (for testing)
4. Or improve model performance

### Issue: Email Not Sending

**Symptoms:** No email notification received

**Solutions:**
1. Verify `EMAIL_SMTP_PASSWORD` is set:
   ```bash
   echo $EMAIL_SMTP_PASSWORD
   ```
2. Check email config in `ci_cd_config.yaml`
3. For Gmail: Use app password, not regular password
4. Check spam folder

### Issue: Docker Containers Won't Start

**Symptoms:** `docker-compose up` fails

**Solutions:**
1. Check Docker is running:
   ```bash
   docker ps
   ```
2. Check for port conflicts:
   ```bash
   lsof -i :8080  # Check if port 8080 is in use
   ```
3. Clean up and restart:
   ```bash
   docker-compose down
   docker-compose up -d
   ```
4. Check logs:
   ```bash
   docker-compose logs
   ```

### Issue: Airflow Variables Not Working

**Symptoms:** DAG uses default values instead of variables

**Solutions:**
1. Verify variables are set in Airflow UI
2. Check variable names match exactly (case-sensitive)
3. Restart Airflow scheduler:
   ```bash
   docker-compose restart airflow-scheduler
   ```

---

## Quick Reference Commands

### GCP Commands
```bash
# List service accounts
gcloud iam service-accounts list --project=datacraft-478300

# List BigQuery datasets
bq ls --project_id=datacraft-478300

# List GCS buckets
gsutil ls -p datacraft-478300

# Upload file to GCS
gsutil cp local-file.csv gs://model-datacraft/data/validated/orders_validated.csv

# List files in GCS
gsutil ls -r gs://model-datacraft/
```

### Airflow Commands
```bash
# Start Airflow
docker-compose up -d

# Stop Airflow
docker-compose down

# View logs
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler

# Access Airflow CLI
docker exec -it airflow-webserver airflow variables list
docker exec -it airflow-webserver airflow dags list
```

### Local Testing Commands
```bash
# Set environment
export GOOGLE_APPLICATION_CREDENTIALS="gcp/service-account.json"
export GCP_PROJECT_ID="datacraft-478300"
export EMAIL_SMTP_PASSWORD="your-app-password"

# Test scripts
python ci-cd/scripts/download_outputs.py
python ci-cd/scripts/validate_model.py
python ci-cd/scripts/check_bias.py
python ci-cd/scripts/rollback_manager.py
python ci-cd/scripts/push_to_registry.py --commit-sha $(git rev-parse --short HEAD)
python ci-cd/scripts/send_notification.py --status success
```

---

## Next Steps

1. **Complete GCP setup** (Section 1)
2. **Complete Airflow setup** (Section 2)
3. **Complete GitHub setup** (Section 3)
4. **Test locally** (Section 4)
5. **Capture screenshots** (Section 5)
6. **Test GitHub Actions** (optional, requires Airflow access)

---

## Support

If you encounter issues not covered in troubleshooting:
1. Check logs: `docker-compose logs`
2. Check GCP console for errors
3. Verify all configuration files match your setup
4. Ensure all secrets and variables are set correctly

**Good luck with your setup and testing!** 🚀

