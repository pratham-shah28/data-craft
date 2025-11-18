# CI/CD Pipeline - Quick Start Guide

## 🚀 Quick Setup (5 Minutes)

### 1. GCP Setup (One-time)

```bash
# Set variables
export PROJECT_ID="datacraft-data-pipeline"
export REGION="us-central1"
export SA_NAME="mlops-ci-cd-service-account"

# Create service account
gcloud iam service-accounts create ${SA_NAME} \
  --display-name="MLOps CI/CD Service Account" \
  --project=${PROJECT_ID}

# Grant permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Create key
gcloud iam service-accounts keys create mlops-ci-cd-key.json \
  --iam-account=${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com

# Create Artifact Registry
gcloud artifacts repositories create model-registry \
  --repository-format=generic \
  --location=${REGION} \
  --project=${PROJECT_ID}

# Enable APIs
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com \
  --project=${PROJECT_ID}
```

### 2. GitHub Secrets (5 minutes)

Go to: **Repository Settings → Secrets and variables → Actions**

Add these secrets:

| Secret Name | Value | How to Get |
|------------|-------|------------|
| `GCP_SA_KEY` | Contents of `mlops-ci-cd-key.json` | Copy entire JSON file content |
| `EMAIL_SENDER` | Your email | `your-email@gmail.com` |
| `EMAIL_SENDER_PASSWORD` | Gmail app password | [Generate App Password](https://myaccount.google.com/apppasswords) |
| `EMAIL_RECIPIENTS` | Comma-separated emails | `email1@gmail.com,email2@gmail.com` |

**Optional** (have defaults):
- `GCP_PROJECT_ID` → `datacraft-data-pipeline`
- `GCP_REGION` → `us-central1`
- `SMTP_SERVER` → `smtp.gmail.com`
- `SMTP_PORT` → `587`

### 3. Test Locally (Optional)

```bash
# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="./mlops-ci-cd-key.json"
export EMAIL_SENDER="your-email@gmail.com"
export EMAIL_SENDER_PASSWORD="your-app-password"
export EMAIL_RECIPIENTS="recipient@gmail.com"

# Test validation script
python model-training/ci-cd/scripts/validate_model_metrics.py

# Test notification
python model-training/ci-cd/scripts/send_notifications.py success outputs/model-training/pipeline_summary.json
```

### 4. Trigger Pipeline

**Option A: Update user queries**
```bash
# Edit model-training/data/user_queries.txt
git add model-training/data/user_queries.txt
git commit -m "Update user queries"
git push
```

**Option B: Manual trigger**
1. Go to GitHub → Actions
2. Select "Model Training CI/CD Pipeline"
3. Click "Run workflow"

**Note**: The pipeline only runs automatically when `user_queries.txt` changes. For other changes, use manual trigger.

## ✅ Verification

1. **Check GitHub Actions**: Should see pipeline running
2. **Check Email**: Should receive notification
3. **Check GCS**: `gs://isha-retail-data/best_model_responses/`
4. **Check Vertex AI**: [Console](https://console.cloud.google.com/vertex-ai/models)

## 🔧 Configuration

Edit `model-training/ci-cd/config/ci_cd_config.yaml` to adjust:
- Validation thresholds
- Bias thresholds
- Email settings
- Deployment settings

## 📚 Full Documentation

See [README.md](README.md) for complete setup guide.

## 🆘 Troubleshooting

**"Authentication failed"**
- Check `GCP_SA_KEY` secret is valid JSON
- Verify service account has permissions

**"Email failed"**
- Use Gmail App Password (not regular password)
- Enable 2-Step Verification first

**"Validation failed"**
- Check thresholds in `ci_cd_config.yaml`
- Review model metrics in logs

---

**That's it!** Your CI/CD pipeline is now set up. 🎉

