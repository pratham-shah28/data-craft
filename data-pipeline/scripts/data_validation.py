import pandas as pd
import numpy as np  # Add this import
import json
from pathlib import Path
from utils import setup_logging, load_config, ensure_dir, detect_encoding
from schema_detector import SchemaDetector

class DataValidator:
    def __init__(self, dataset_name):
        self.logger = setup_logging("data_validation")
        self.config = load_config()
        self.dataset_name = dataset_name
        self.validation_report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_name": dataset_name,
            "checks": {}
        }
        
        # Load schema profile
        detector = SchemaDetector()
        self.schema_profile = detector.load_schema_profile(dataset_name)
        
        if self.schema_profile is None:
            self.logger.warning("No schema profile found. Generating new profile...")
    
    def validate_basic_structure(self, df):
        """Validate basic data structure"""
        self.logger.info("Validating basic structure...")
        
        min_rows = self.config['validation']['min_rows']
        min_cols = self.config['validation']['min_columns']
        
        structure_valid = len(df) >= min_rows and len(df.columns) >= min_cols
        
        self.validation_report['checks']['structure'] = {
            "valid": bool(structure_valid),  # Convert to Python bool
            "actual_rows": int(len(df)),
            "actual_columns": int(len(df.columns)),
            "min_rows_required": int(min_rows),
            "min_columns_required": int(min_cols)
        }
        
        if not structure_valid:
            self.logger.warning(f"Structure validation failed. Dataset too small.")
        else:
            self.logger.info("Structure validation passed")
        
        return structure_valid
    
    def check_nulls(self, df):
        """Check for null values across all columns"""
        self.logger.info("Checking for null values...")
        
        null_threshold = self.config['validation']['null_threshold']
        null_percentages = df.isnull().sum() / len(df)
        
        columns_exceeding_threshold = null_percentages[null_percentages > null_threshold].to_dict()
        
        nulls_valid = len(columns_exceeding_threshold) == 0
        
        self.validation_report['checks']['nulls'] = {
            "valid": bool(nulls_valid),  # Convert to Python bool
            "threshold": float(null_threshold),
            "null_percentages": {k: float(v) for k, v in null_percentages.items()},
            "columns_exceeding_threshold": {k: float(v) for k, v in columns_exceeding_threshold.items()},
            "total_null_cells": int(df.isnull().sum().sum()),
            "null_percentage_overall": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        }
        
        if not nulls_valid:
            self.logger.warning(f"Null check failed. Columns exceeding threshold: {list(columns_exceeding_threshold.keys())}")
        else:
            self.logger.info("Null check passed")
        
        return nulls_valid
    
    def check_duplicates(self, df):
        """Check for duplicate records"""
        self.logger.info("Checking for duplicates...")
        
        duplicate_threshold = self.config['validation']['duplicate_threshold']
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df)
        
        duplicates_valid = duplicate_percentage <= duplicate_threshold
        
        self.validation_report['checks']['duplicates'] = {
            "valid": bool(duplicates_valid),  # Convert to Python bool
            "threshold": float(duplicate_threshold),
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": float(duplicate_percentage)
        }
        
        if not duplicates_valid:
            self.logger.warning(f"Duplicate check failed. Found {duplicate_count} duplicates ({duplicate_percentage:.2%})")
        else:
            self.logger.info(f"Duplicate check passed. Found {duplicate_count} duplicates ({duplicate_percentage:.2%})")
        
        return duplicates_valid
    
    def detect_anomalies(self, df):
        """Detect anomalies in numerical columns (dynamic)"""
        self.logger.info("Detecting anomalies...")
        
        # Get numerical columns from schema profile if available
        if self.schema_profile:
            numerical_cols = self.schema_profile['column_types'].get('continuous_numeric', []) + \
                           self.schema_profile['column_types'].get('discrete_numeric', [])
        else:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        anomalies = {}
        
        for col in numerical_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                if len(outliers) > 0:
                    anomalies[col] = {
                        "outlier_count": int(len(outliers)),  # Convert to Python int
                        "outlier_percentage": float(len(outliers) / len(df) * 100),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "min_value": float(df[col].min()),
                        "max_value": float(df[col].max())
                    }
        
        self.validation_report['checks']['anomalies'] = {
            "columns_analyzed": numerical_cols,
            "columns_with_anomalies": list(anomalies.keys()),
            "details": anomalies
        }
        
        self.logger.info(f"Found anomalies in {len(anomalies)} columns")
        
        return anomalies
    
    def validate(self):
        """Run all validation checks"""
        self.logger.info(f"Starting data validation for: {self.dataset_name}")
        
        try:
            # Load data
            data_path = f'/opt/airflow/data-pipeline/data/raw/{self.dataset_name}.csv'
            encoding, confidence = detect_encoding(data_path)
            df = pd.read_csv(data_path, encoding=encoding)
            self.logger.info(f"Loaded data: {df.shape}")
            
            # Run validation checks
            structure_valid = self.validate_basic_structure(df)
            nulls_valid = self.check_nulls(df)
            duplicates_valid = self.check_duplicates(df)
            anomalies = self.detect_anomalies(df)
            
            # Overall validation status - CONVERT TO PYTHON BOOL
            self.validation_report['overall_valid'] = bool(structure_valid and nulls_valid and duplicates_valid)
            
            # Save validation report
            ensure_dir(self.config['data']['processed_path'])
            report_path = Path(self.config['data']['processed_path']) / f"{self.dataset_name}_validation_report.json"
            
            with open(report_path, 'w') as f:
                json.dump(self.validation_report, f, indent=2)
            
            self.logger.info(f"Validation report saved to {report_path}")
            
            # Save validated data for next stage
            output_path = Path(self.config['data']['processed_path']) / f"{self.dataset_name}_validated.csv"
            df.to_csv(output_path, index=False)
            self.logger.info(f"Validated data saved to {output_path}")
            
            if self.validation_report['overall_valid']:
                self.logger.info("✓ All validation checks passed")
            else:
                self.logger.warning("⚠ Some validation checks failed. Review validation report.")
            
            return self.validation_report
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_validation.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    validator = DataValidator(dataset_name)
    validator.validate()