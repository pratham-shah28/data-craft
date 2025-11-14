# Model Pipeline DAG - Quick Start Guide

## 🎯 **What This Pipeline Does**

Evaluates multiple Gemini models, detects biases, selects the best model, executes queries, and validates answers.

**Input:** 30 user queries (natural language questions)  
**Output:** Best model selected + validated SQL queries + actual answers

---

## 📁 **Required Setup**

### **1. File Structure**
```
mlops-project/
├── model-training/
│   ├── dags/
│   │   └── model_pipeline_complete_with_eval.py  
│   ├── scripts/
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── metadata_manager.py
│   │   ├── prompts.py
│   │   ├── model_evaluator.py        
│   │   ├── bias_detector.py          
│   │   ├── model_selector.py         
│   │   ├── response_saver.py          
│   │   └── query_executor.py          
│   └── data/
│       └── user_queries.txt           (test queries)
└── docker-compose.yml                 
```

### **2. Create DAG Folders**
```bash
mkdir -p model-training/dags
mkdir -p data-pipeline/dags
```

### **3. Move DAG File**
Place `model_pipeline_complete_with_eval.py` in `model-training/dags/`

---

## 🚀 **How to Run**

### **Step 1: Start Airflow**
```bash
# From project root
docker-compose up -d

# Wait 60 seconds for Airflow to initialize
```

### **Step 2: Access Airflow UI**
```
URL: http://localhost:8080
Username: admin
Password: admin
```

### **Step 3: Find Your DAG**
Look for: **`model_pipeline_with_evaluation`**

Tags: `model`, `gemini`, `evaluation`, `bias-detection`, `mlops`

### **Step 4: Trigger DAG**
1. Click on the DAG name
2. Click "Trigger DAG" button (▶️ icon)
3. Confirm (no configuration needed)

### **Step 5: Monitor Progress**
Watch the task progress:
```
load_data_to_bigquery → generate_features → store_metadata → 
read_user_queries → process_queries_with_all_models → 
[evaluate_all_models + detect_bias_in_all_models] → 
select_best_model → execute_validate_queries → 
save_best_model_responses → generate_final_summary
```

**Expected Duration:** 5-10 minutes (depends on query count and model speed)

---

## 📊 **What Happens**

### **Phase 1: Data Setup** (1-2 min)
- Loads orders data to BigQuery
- Generates metadata (column info, statistics)
- Reads 30 queries from `user_queries.txt`

### **Phase 2: Multi-Model Processing** (3-5 min)
- **gemini-2.5-flash** processes all 30 queries
- **gemini-2.5-pro** processes all 30 queries
- Total: 60 SQL queries generated

### **Phase 3: Evaluation** (1 min)
- Tests SQL syntax validity
- Tests SQL executability
- Checks format compliance
- **Checks intent matching** (does SQL answer the question?)
- Measures response time

### **Phase 4: Bias Detection** (30 sec)
- Checks visualization type distribution
- Checks query pattern distribution
- Checks column usage fairness
- Checks sentiment in explanations

### **Phase 5: Model Selection** (10 sec)
- Calculates composite scores
- Ranks models
- Selects best model

### **Phase 6: Query Execution & Validation** ✅ NEW (1-2 min)
- Executes best model's SQL on BigQuery
- Gets actual results
- Validates results are correct
- Generates natural language answers

### **Phase 7: Save Results** (30 sec)
- Saves only best model's responses
- Includes execution results and answers
- Uploads to GCS

### **Phase 8: Summary** (10 sec)
- Generates pipeline summary
- Reports accuracy metrics

---

## 📁 **Where to Find Results**

### **Best Model Selection:**
```bash
/opt/airflow/outputs/model-selection/model_selection_*.json
```

```json
{
  "best_model": {
    "name": "gemini-2.5-flash",
    "composite_score": 89.5,
    "performance_score": 92.0,
    "bias_score": 25.0
  }
}
```

### **Accuracy Metrics:**
```bash
/opt/airflow/outputs/best-model-responses/*/accuracy_metrics.json
```

```json
{
  "total_queries": 30,
  "successfully_executed": 28,
  "valid_results": 25,
  "overall_accuracy": 83.3%
}
```

### **Individual Query Results:**
```bash
/opt/airflow/outputs/best-model-responses/*/query_001/execution_results.json
```

```json
{
  "user_query": "What are total sales in 2024?",
  "sql_query": "SELECT SUM(sales)...",
  "execution_status": "success",
  "results_valid": true,
  "result_data": [{"total_sales": 1234567.89}],
  "natural_language_answer": "The total is $1,234,567.89"
}
```

---

## 🐛 **Troubleshooting**

### **DAG not showing?**
```bash
# Check DAG folder
ls -la model-training/dags/

# Check inside container
docker exec -it airflow-scheduler bash
ls -la /opt/airflow/dags/model-training/

# Check scheduler logs
docker-compose logs airflow-scheduler | grep model_pipeline
```

### **DAG failing?**
```bash
# View task logs in Airflow UI
# Or via CLI:
docker exec -it airflow-scheduler airflow tasks test \
  model_pipeline_with_evaluation \
  load_data_to_bigquery \
  2025-01-01
```

### **Import errors?**
```bash
# Check PYTHONPATH inside container
docker exec -it airflow-scheduler bash
echo $PYTHONPATH

# Should include:
# /opt/airflow/model-training/scripts
```

---

## 📝 **Quick Test (Without Full Run)**

Test individual scripts locally:

```bash
cd model-training/scripts

# Test evaluation logic
python model_evaluator.py

# Test bias detection
python bias_detector.py

# Test model selection
python model_selector.py

# Test query execution (requires GCP credentials)
python query_executor.py
```

---

## ✅ **Success Criteria**

Pipeline succeeds when:
- ✅ Both models process all 30 queries
- ✅ Evaluation reports generated
- ✅ Bias reports generated
- ✅ Best model selected
- ✅ Queries executed with >80% accuracy
- ✅ Results saved with natural language answers

**Final Output:** Production-ready model with validated, correct answers! 🎉

---

## 📞 **Need Help?**

Check logs:
```bash
# Scheduler logs
docker-compose logs -f airflow-scheduler

# Webserver logs
docker-compose logs -f airflow-webserver

# Task-specific logs
# (Available in Airflow UI → DAG → Task → Logs)
```