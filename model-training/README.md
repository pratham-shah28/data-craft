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

### **Phase 6: Query Execution & Validation** 
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
  "natural_language_answer": "The total is $98,244,567.67"
}
```