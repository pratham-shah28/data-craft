# LLM1 Model Integration Summary

**Date:** November 17, 2025  
**Integrated From:** `pratham-model-deployment` branch  
**Integrated By:** Viraj (for Task 7 - Experiment Tracking)

---

## 🎯 Purpose

This integration brings Pratham's LLM1 (Document Extraction) training code into the `vishal` branch to enable:
1. **Task 1 completion** - LLM1 model training for document extraction
2. **Task 7 implementation** - MLflow experiment tracking for LLM1 pipeline
3. **End-to-end MLOps workflow** - from unstructured data → structured JSON

---

## 📦 What Was Integrated

### 1. Model Code (`model_1/`)
```
model_1/
├── README.md                          # Documentation on dynamic few-shot prompting
├── v1_agnostic/                       # Cloud-agnostic version
│   ├── Dockerfile
│   ├── app/
│   │   └── server.py                  # FastAPI server for inference
│   └── requirements.txt
└── v2_vertex/                         # Vertex AI optimized version
    ├── example_provider.py            # Builds few-shot examples from invoices
    ├── inference.py                   # Main inference script using Gemini
    └── main.py
```

### 2. Training Data (`data-pipeline/data/unstructured/invoices/`)
```
invoices/
├── examples.jsonl                     # Pre-built few-shot manifest (base64 images + JSON)
├── invoice_1.pdf + invoice_1.json     # Example 1
├── invoice_2.pdf + invoice_2.json     # Example 2
...
└── invoice_6.pdf + invoice_6.json     # Example 6
```

---

## 🔧 Key Components

### A. Dynamic Few-Shot Prompting System
**Location:** `model_1/v2_vertex/inference.py`

**How it works:**
1. Detects document type (invoice, contract, tax form, etc.)
2. Loads relevant few-shot examples from `examples.jsonl`
3. Constructs optimized prompt with schema + examples
4. Calls Gemini 2.5 Flash with `response_mime_type="application/json"`
5. Parses structured JSON output using `JsonOutputParser`

**Supported Document Types:**
- Invoices → vendor, totals, tax lines
- Financial statements → balance sheet fields
- Tax forms → form IDs, taxpayer metadata
- Certificates → issuer, validity dates
- Contracts → parties, obligations, signatures

### B. Example Provider
**Location:** `model_1/v2_vertex/example_provider.py`

**Functionality:**
- Scans `data-pipeline/data/unstructured/invoices/` for PDF-JSON pairs
- Uses `pdf_to_base64_images()` to convert PDFs → base64 data URLs
- Builds structured examples list for few-shot prompting
- Writes `examples.jsonl` manifest (already pre-built in this integration)

**Key Design:**
- No model retraining needed when adding new document types
- Just add new PDF-JSON pairs and rebuild examples
- Extensible to any document family

---

## 🚀 How to Use LLM1 Now

### Option 1: Run Inference Directly
```bash
cd /Users/viraj/Desktop/mlops-project/model_1/v2_vertex

# Set environment variables
export GOOGLE_CLOUD_PROJECT="mlops-472423"
export VERTEX_LOCATION="us-central1"
export VERTEX_MODEL="gemini-2.5-flash"
export TARGET_PDF="../../data-pipeline/data/unstructured/vehicle_insurance/policy_1.pdf"

# Run inference
python inference.py
```

### Option 2: Integrate with MLflow (Your Task 7)
```python
import mlflow
from model_1.v2_vertex import inference

# Start MLflow tracking
mlflow.set_experiment("LLM1_Document_Extraction")

with mlflow.start_run(run_name="invoice_extraction_test"):
    # Log parameters
    mlflow.log_param("model", "gemini-2.5-flash")
    mlflow.log_param("temperature", 0.0)
    mlflow.log_param("max_tokens", 8192)
    mlflow.log_param("document_type", "invoice")
    
    # Run inference (you'll need to adapt this)
    result = run_inference(...)
    
    # Log metrics
    mlflow.log_metric("num_fields_extracted", len(result.keys()))
    mlflow.log_metric("confidence_score", calculate_confidence(result))
    
    # Log artifacts
    mlflow.log_artifact("output.json")
    mlflow.log_artifact("examples.jsonl")
```

---

## 🔗 Dependencies

### Already in Data Pipeline:
- ✅ `pdf_2_image.py` - Converts PDFs to base64 images
- ✅ Invoice examples (invoice_1 through invoice_6)
- ✅ `data/unstructured/` structure

### LLM1 Specific:
- Google Cloud Vertex AI SDK
- `vertexai.generative_models` 
- `langchain_core.output_parsers`
- Gemini 2.5 Flash model access

---

## 📝 Next Steps for Task 7 (Experiment Tracking)


Now that you have LLM1 code integrated, you need to:

### 1. Set Up MLflow Tracking (Due Nov 15)
```bash
# Install MLflow
pip install mlflow

# Start MLflow UI
mlflow ui --port 5000
```

### 2. Create MLflow Integration Script
Create `model_1/mlflow_tracking.py` to:
- Track LLM1 experiments (document type, model params, examples used)
- Log metrics (extraction accuracy, field completeness, processing time)
- Version model artifacts (prompts, examples.jsonl, inference configs)
- Compare different few-shot example combinations

### 3. Track Key Experiments
Log experiments for:
- Different Gemini model versions (flash vs pro)
- Temperature variations (0.0, 0.3, 0.7)
- Number of few-shot examples (2, 3, 5)
- Document types (invoices, contracts, certificates)

### 4. Example MLflow Script Structure
```python
# model_1/mlflow_tracking.py
import mlflow
import json
from pathlib import Path
from v2_vertex.inference import run_inference  # you'll need to refactor this

def track_experiment(
    doc_type: str,
    temperature: float,
    num_examples: int,
    target_pdf: str
):
    mlflow.set_experiment(f"LLM1_{doc_type}_extraction")
    
    with mlflow.start_run():
        # Log params
        mlflow.log_param("document_type", doc_type)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("num_examples", num_examples)
        mlflow.log_param("model", "gemini-2.5-flash")
        
        # Run inference
        start_time = time.time()
        result = run_inference(target_pdf, temperature, num_examples)
        duration = time.time() - start_time
        
        # Log metrics
        mlflow.log_metric("inference_time_sec", duration)
        mlflow.log_metric("num_fields_extracted", len(result))
        mlflow.log_metric("json_valid", 1 if is_valid_json(result) else 0)
        
        # Log artifacts
        output_path = Path("output.json")
        output_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(output_path))
        
        return result
```

---

## 🎓 Key Learnings from Pratham's Implementation

1. **Dynamic Few-Shot Prompting** - More scalable than fixed prompts
2. **Document Type Detection** - Enables automatic example selection
3. **Base64 Image Encoding** - Simplifies multi-modal LLM inference
4. **JSON Output Parser** - Ensures structured, schema-compliant output
5. **Modular Design** - Easy to extend to new document types

---

## ✅ Integration Checklist

- [x] Merged `model_1/` directory with both v1 and v2 implementations
- [x] Integrated `examples.jsonl` manifest with base64 invoice images
- [x] Verified invoice examples (PDFs + JSONs) are present
- [x] Documented integration process
- [ ] Test LLM1 inference locally
- [ ] Integrate with MLflow for Task 7
- [ ] Create experiment tracking dashboard
- [ ] Document results for final report (due Nov 18)

---

## 📧 Questions?

**For LLM1 Model Questions:** Contact Pratham  
**For MLflow Integration:** This is your task (Viraj/Vishal)  
**For LLM2 Integration:** Contact Isha (already completed)

---

**Status:** ✅ LLM1 code successfully integrated into `vishal` branch  
**Commits Made:**
1. `e63dc99` - feat: Merge LLM1 training model code from pratham-model-deployment branch
2. `55df840` - feat: Add examples.jsonl manifest for LLM1 few-shot learning

**Ready for:** Task 7 (Experiment Tracking with MLflow) implementation 🚀
