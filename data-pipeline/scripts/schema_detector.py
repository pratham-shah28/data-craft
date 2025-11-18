import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from utils import setup_logging, load_config, ensure_dir

class SchemaDetector:
    def __init__(self):
        self.logger = setup_logging("schema_detector")
        self.config = load_config()
        
    def detect_column_type(self, series):
        """Intelligently detect column type"""
        # Skip if all null
        if series.isnull().all():
            return "unknown"
        
        # Remove nulls for type detection
        series_clean = series.dropna()
        
        # Check for datetime
        if series_clean.dtype == 'object':
            try:
                # Suppress the datetime parsing warning
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Could not infer format')
                    
                    # Try common date formats first for efficiency
                    common_formats = [
                        '%Y-%m-%d',
                        '%m/%d/%Y',
                        '%d/%m/%Y',
                        '%Y-%m-%d %H:%M:%S',
                        '%m/%d/%Y %H:%M:%S',
                        '%d-%m-%Y',
                        '%Y/%m/%d'
                    ]
                    
                    # Try specific formats first (faster and no warnings)
                    for fmt in common_formats:
                        try:
                            parsed = pd.to_datetime(series_clean.head(100), format=fmt, errors='coerce')
                            if parsed.notna().sum() / len(series_clean.head(100)) > 0.8:
                                return "datetime"
                        except:
                            continue
                    
                    # Fall back to infer_datetime_format if specific formats don't work
                    parsed = pd.to_datetime(series_clean.head(100), errors='coerce')
                    if parsed.notna().sum() / len(series_clean.head(100)) > 0.8:
                        return "datetime"
            except:
                pass
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series_clean):
            # Check if it's an ID column (high uniqueness)
            if series_clean.nunique() / len(series_clean) > 0.95:
                return "identifier"
            # Check if integer with low cardinality (might be categorical)
            elif series_clean.dtype in ['int64', 'int32'] and series_clean.nunique() < 20:
                return "categorical"
            # Check if it's continuous numeric
            elif series_clean.dtype in ['float64', 'float32']:
                return "continuous_numeric"
            else:
                return "discrete_numeric"
        
        # Check for boolean
        if series_clean.dtype == 'bool':
            return "boolean"
        
        # Check for categorical (text with limited unique values)
        unique_ratio = series_clean.nunique() / len(series_clean)
        if unique_ratio < 0.5:
            return "categorical"
        
        # High cardinality text
        return "text"
    
    def detect_protected_attributes(self, df):
        """Auto-detect potential protected/sensitive attributes"""
        self.logger.info("Detecting potential protected attributes...")
        
        protected_attrs = []
        sensitive_keywords = self.config['bias']['sensitive_keywords']
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if column name contains sensitive keywords
            for keyword in sensitive_keywords:
                if keyword in col_lower:
                    protected_attrs.append(col)
                    self.logger.info(f"Detected protected attribute: {col}")
                    break
            
            # Check if categorical with reasonable cardinality (2-20 unique values)
            if col not in protected_attrs:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    unique_count = df[col].nunique()
                    if 2 <= unique_count <= 20:
                        # Additional heuristic: might be demographic
                        protected_attrs.append(col)
                        self.logger.info(f"Detected potential demographic attribute: {col}")
        
        return protected_attrs
    
    def generate_schema_profile(self, df, dataset_name):
        """Generate comprehensive schema profile"""
        self.logger.info(f"Generating schema profile for: {dataset_name}")
        
        schema_profile = {
            "dataset_name": dataset_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            "columns": {},
            "column_types": {},
            "protected_attributes": [],
            "recommended_metrics": [],
            "data_quality": {}
        }
        
        # Analyze each column
        for col in df.columns:
            detected_type = self.detect_column_type(df[col])
            
            schema_profile["columns"][col] = {
                "detected_type": detected_type,
                "pandas_dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float(df[col].isnull().sum() / len(df) * 100),
                "unique_count": int(df[col].nunique()),
                "unique_percentage": float(df[col].nunique() / len(df) * 100),
            }
            
            # Add type-specific metadata
            if detected_type in ["continuous_numeric", "discrete_numeric"]:
                schema_profile["columns"][col].update({
                    "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                    "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                    "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
                })
                schema_profile["recommended_metrics"].append(col)
            
            elif detected_type == "categorical":
                top_values = df[col].value_counts().head(5).to_dict()
                schema_profile["columns"][col]["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            
            elif detected_type == "datetime":
                schema_profile["columns"][col].update({
                    "min_date": str(pd.to_datetime(df[col]).min()),
                    "max_date": str(pd.to_datetime(df[col]).max()),
                })
            
            # Track column types
            if detected_type not in schema_profile["column_types"]:
                schema_profile["column_types"][detected_type] = []
            schema_profile["column_types"][detected_type].append(col)
        
        # Auto-detect protected attributes
        if self.config['bias']['auto_detect_protected_attrs']:
            schema_profile["protected_attributes"] = self.detect_protected_attributes(df)
        
        # Data quality assessment
        schema_profile["data_quality"] = {
            "total_nulls": int(df.isnull().sum().sum()),
            "null_percentage": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100),
        }
        
        # Save schema profile
        ensure_dir("config/dataset_profiles")
        profile_path = Path("config/dataset_profiles") / f"{dataset_name}_profile.json"
        
        with open(profile_path, 'w') as f:
            json.dump(schema_profile, f, indent=2)
        
        self.logger.info(f"Schema profile saved to {profile_path}")
        
        return schema_profile
    
    def load_schema_profile(self, dataset_name):
        """Load existing schema profile"""
        profile_path = Path("config/dataset_profiles") / f"{dataset_name}_profile.json"
        
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                return json.load(f)
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python schema_detector.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    dataset_name = Path(dataset_path).stem
    
    detector = SchemaDetector()
    df = pd.read_csv(dataset_path)
    profile = detector.generate_schema_profile(df, dataset_name)
    
    print(f"\n✓ Schema profile generated for {dataset_name}")
    print(f"Columns: {len(profile['columns'])}")
    print(f"Protected attributes detected: {profile['protected_attributes']}")