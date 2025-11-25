# CI/CD Pipeline for Model Training

Automated CI/CD pipeline that triggers model training, validates quality, and deploys to production.

## Overview

This CI/CD pipeline automates the model training workflow:

1. **Triggers Airflow DAG** when code changes (user queries, training code, configs)
2. **Waits for DAG completion** (DAG trains models, evaluates, selects best)
3. **Downloads outputs** from GCS (where DAG saves results)
4. **Enforces quality thresholds** (performance, bias, execution accuracy)
5. **Compares with production** (rollback protection)
6. **Deploys to production** (if all checks pass, tagged with commit SHA)
7. **Sends notifications** (email alerts to stakeholders)

## Architecture

```
Developer Changes Code
    ↓
Push to GitHub
    ↓
CI/CD Detects Change
    ↓
Trigger Airflow DAG
    ↓
Wait for DAG Completion
    ↓
Download Best Model Responses from GCS
    (DAG saves to: best_model_responses/{timestamp}_{model}/)
    ↓
Validate Quality Thresholds
    (Uses: model_selection_report.json + accuracy_metrics.json)
    ↓
Compare with Production
    (Uses: model_selection_report.json)
    ↓
Deploy to Production (if valid)
    (Uploads to: models/{timestamp}_{commit}/)
    ↓
Send Notifications
```

**Note:** CI/CD only uses best model responses saved by the DAG. No separate evaluation/bias reports needed.

## Prerequisites

### 1. GitHub Secrets

Go to: **Repository → Settings → Secrets and variables → Actions**

Add these secrets:

- **`GCP_SA_KEY`** (Required)
  - Full JSON content of your GCP service account key
  - Permissions: BigQuery Data Editor, Storage Admin

- **`EMAIL_SMTP_PASSWORD`** (Required for notifications)
  - Email app password (for Gmail, generate app password)
  - How to get: Google Account → Security → App Passwords

- **`EMAIL_SMTP_USER`** (Optional)
  - Your email address
  - Defaults to `from_email` in config

- **`AIRFLOW_URL`** (Required)
  - Airflow base URL (e.g., `http://localhost:8080` or Cloud Composer URL)

- **`AIRFLOW_USERNAME`** (Required)
  - Airflow username (default: `admin`)

- **`AIRFLOW_PASSWORD`** (Required)
  - Airflow password (default: `admin`)

### 2. Configuration Files

Update `ci-cd/config/ci_cd_config.yaml`:

```yaml
gcp:
  project_id: "your-gcp-project-id"
  region: "us-east1"
  dataset_id: "datacraft_ml"
  bucket_name: "model-datacraft"

notifications:
  email:
    from_email: "your-email@gmail.com"
    to_email: "your-email@gmail.com"

model_registry:
  base_path: "models"

rollback:
  enabled: true
  min_improvement_threshold: 0.0
```

Update `ci-cd/config/validation_thresholds.yaml`:

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

### 3. GCP Resources

Ensure these exist:
- BigQuery dataset: `datacraft_ml` (or as configured)
- GCS bucket: `model-datacraft` (or as configured)
- Service account with required permissions

## Pipeline Jobs

### 1. Trigger DAG
- Triggers Airflow DAG `model_pipeline_with_evaluation`
- Waits for DAG completion (max 30 minutes)
- Fails if DAG fails

### 2. Download Outputs
- Downloads latest best model responses from GCS
- DAG saves to: `best_model_responses/{timestamp}_{model}/`
- Extracts to `outputs/best-model-responses/` directory
- Uploads as GitHub artifact

### 3. Validate
- Reads `model_selection_report.json` for performance metrics
- Reads `accuracy_metrics.json` for execution metrics
- Checks performance thresholds (composite_score, success_rate)
- Validates execution accuracy
- Fails if thresholds not met

### 4. Check Bias
- Reads `model_selection_report.json` for bias_score
- Validates bias thresholds
- Fails if bias too high

### 5. Compare & Deploy
- Reads `model_selection_report.json` for current model metrics
- Compares with previous model from production registry
- Blocks deployment if worse (rollback)
- Promotes to production if better
- Uploads to `models/{timestamp}_{commit}/` with commit SHA

### 6. Notify
- Sends email notification
- Includes pipeline status and metrics
- Runs even on failure

## When Pipeline Triggers

Pipeline triggers on push to `main` branch when these files change:
- `model-training/data/user_queries.txt` - User queries
- `model-training/scripts/**` - Training code
- `ci-cd/**` - CI/CD configuration
- `.github/workflows/model-training-ci-cd.yml` - Workflow file

You can also trigger manually via GitHub Actions UI.

## Testing the Pipeline

### Step 1: Local Testing (Optional)

Test scripts locally before pushing:

```bash
# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GCP_PROJECT_ID="datacraft-478300"
export EMAIL_SMTP_PASSWORD="your-app-password"

# Test downloading outputs (requires DAG to have run first)
python ci-cd/scripts/download_outputs.py

# Test validation
python ci-cd/scripts/validate_model.py

# Test bias check
python ci-cd/scripts/check_bias.py

# Test rollback check
python ci-cd/scripts/rollback_manager.py

# Test notification
python ci-cd/scripts/send_notification.py --status success
```

### Step 2: Trigger Pipeline

1. **Make a change** to trigger the pipeline:
   ```bash
   # Edit user queries
   echo "What are the top 5 products by sales?" >> model-training/data/user_queries.txt
   
   # Commit and push
   git add model-training/data/user_queries.txt
   git commit -m "Add new query"
   git push origin main
   ```

2. **Monitor pipeline** in GitHub Actions:
   - Go to: **Repository → Actions**
   - Click on the latest workflow run
   - Watch each job execute

### Step 3: Verify Results

1. **Check GitHub Actions logs**:
   - Each job should show detailed logs
   - Look for "✓" success markers
   - Check for any "✗" errors

2. **Check GCS bucket**:
   ```bash
   gsutil ls -r gs://model-datacraft/models/
   ```
   - Should see new model version with timestamp and commit SHA
   - Example: `models/20250115_120000_abc1234/`

3. **Check email notification**:
   - Should receive email with pipeline status
   - Includes best model, scores, and metrics

### Step 4: Test Rollback Scenario

To test rollback protection:

1. **First deployment** (should succeed):
   - Push changes that improve model
   - Pipeline should deploy

2. **Second deployment** (should block):
   - Push changes that make model worse
   - Pipeline should block deployment
   - Check logs for "DEPLOYMENT BLOCKED"

## Troubleshooting

### DAG Not Triggering

**Issue:** `trigger_dag` job fails

**Solutions:**
- Check `AIRFLOW_URL` secret is correct
- Check `AIRFLOW_USERNAME` and `AIRFLOW_PASSWORD` are correct
- Verify Airflow is accessible from GitHub Actions
- Check DAG ID matches: `model_pipeline_with_evaluation`

### DAG Timeout

**Issue:** DAG doesn't complete within 30 minutes

**Solutions:**
- Check Airflow logs for DAG execution
- Verify DAG is running (not stuck)
- Increase timeout in `trigger_dag.py` if needed

### No Outputs Found

**Issue:** `download_outputs` job fails

**Solutions:**
- Verify DAG completed successfully
- Check GCS bucket has best model responses at `best_model_responses/`
- Verify DAG uploaded outputs (check Airflow logs)

### Validation Fails

**Issue:** `validate` job fails

**Solutions:**
- Check validation report in `outputs/validation/validation_report.json`
- Review which thresholds failed
- Adjust thresholds in `validation_thresholds.yaml` if needed
- Or improve model performance

### Rollback Triggered

**Issue:** Deployment blocked due to rollback

**Solutions:**
- This is expected if new model is worse
- Check comparison report in `outputs/validation/model_comparison_report.json`
- Improve model or adjust `min_improvement_threshold` in config

### Email Not Sending

**Issue:** No email notification received

**Solutions:**
- Check `EMAIL_SMTP_PASSWORD` secret is set
- Verify email config in `ci_cd_config.yaml`
- Check notification job logs for errors
- For Gmail: Use app password, not regular password

## Files Structure

```
ci-cd/
├── config/
│   ├── ci_cd_config.yaml          # CI/CD configuration
│   └── validation_thresholds.yaml  # Quality thresholds
├── scripts/
│   ├── trigger_dag.py             # Trigger Airflow DAG
│   ├── download_outputs.py        # Download from GCS
│   ├── validate_model.py          # Enforce quality thresholds
│   ├── check_bias.py              # Check bias thresholds
│   ├── rollback_manager.py        # Compare with production
│   ├── push_to_registry.py        # Deploy to production
│   └── send_notification.py       # Send email notifications
└── README.md                       # This file
```

## Key Concepts

### Quality Gates

CI/CD enforces minimum thresholds:
- **Performance**: Overall score, success rate, syntax validity, intent matching
- **Bias**: Bias score, severity level
- **Execution**: Execution success rate, result validity, overall accuracy

If any threshold fails, deployment is blocked.

### Rollback Protection

CI/CD compares new model with production:
- **Better**: Deploys if new model score > production score
- **Worse**: Blocks deployment if new model score < production score
- **First deployment**: Always deploys (no previous model)

### Version Control

Each deployment is tagged with:
- **Timestamp**: `YYYYMMDD_HHMMSS`
- **Commit SHA**: First 7 characters of git commit

Example: `models/20250115_120000_abc1234/`

## Next Steps

1. **Configure secrets** in GitHub
2. **Update config files** with your settings
3. **Test locally** (optional)
4. **Push changes** to trigger pipeline
5. **Monitor execution** in GitHub Actions
6. **Verify deployment** in GCS

For questions or issues, check the troubleshooting section above.
