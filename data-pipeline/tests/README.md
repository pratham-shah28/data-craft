# Test Suite for MLOps Data Pipeline

## Overview

This directory contains 42 unit tests covering the essential components of the data pipeline.

## Test Files

- **test_data_validation.py** (8 tests) - Tests data validation logic (structure, nulls, duplicates, anomalies)
- **test_data_cleaning.py** (6 tests) - Tests data cleaning operations (normalization, missing values, outliers, type standardization)
- **test_bias_detection.py** (6 tests) - Tests bias detection and fairness metrics
- **test_schema_detector.py** (7 tests) - Tests automatic schema detection and type inference
- **test_utils.py** (11 tests) - Tests utility helper functions

## Running the Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/test_data_validation.py -v
python -m pytest tests/test_data_cleaning.py -v
python -m pytest tests/test_bias_detection.py -v
python -m pytest tests/test_schema_detector.py -v
python -m pytest tests/test_utils.py -v
```

### With Coverage Report
```bash
python -m pytest tests/ --cov=scripts --cov-report=html
```

## What's Tested

### Data Validation
- Minimum dataset dimensions (rows/columns)
- Null value percentage checks
- Duplicate detection
- Statistical anomaly detection

### Data Cleaning
- Column name normalization
- Missing value handling (by type)
- Duplicate removal
- Outlier capping using IQR
- Data type standardization

### Bias Detection
- Protected attribute identification
- Representation bias (group balance)
- Disparate impact ratio (80% rule)
- Statistical parity tests

### Schema Detection
- Automatic column type inference
- Protected attribute detection
- Schema profile generation

### Utilities
- Encoding detection
- Formatting functions (size, duration)
- Configuration validation
