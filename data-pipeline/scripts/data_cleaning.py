import pandas as pd
import numpy as np
import json
from pathlib import Path
from utils import setup_logging, load_config, ensure_dir
from schema_detector import SchemaDetector

class DataCleaner:
    def __init__(self, dataset_name):
        self.logger = setup_logging("data_cleaning")
        self.config = load_config()
        self.dataset_name = dataset_name
        self.cleaning_metrics = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_name": dataset_name,
            "operations": []
        }
        
        # Load schema profile
        detector = SchemaDetector()
        self.schema_profile = detector.load_schema_profile(dataset_name)
    
    def get_column_types(self, df):
        """Get column types from schema profile or detect dynamically"""
        if self.schema_profile and 'column_types' in self.schema_profile:
            return self.schema_profile['column_types']
        else:
            # Fallback: basic type detection
            self.logger.warning("No schema profile found. Using basic type detection...")
            detector = SchemaDetector()
            profile = detector.generate_schema_profile(df, self.dataset_name)
            return profile['column_types']
    
    def handle_missing_values(self, df):  # FIXED INDENTATION - this must be inside the class
        """Handle missing values based on column type"""
        self.logger.info("Handling missing values...")
        
        # Get identifier columns - USE NORMALIZED NAMES
        if self.schema_profile:
            identifier_cols = self.schema_profile['column_types'].get('identifier', [])
            # Normalize the identifier column names to match the dataframe
            identifier_cols = [col.lower().replace(' ', '_').replace('-', '_') for col in identifier_cols]
        else:
            identifier_cols = []
        
        # Drop rows where identifiers are missing (these are critical)
        if identifier_cols:
            # Filter to only include columns that actually exist in the dataframe
            existing_identifier_cols = [col for col in identifier_cols if col in df.columns]
            if existing_identifier_cols:
                before_count = len(df)
                df = df.dropna(subset=existing_identifier_cols)
                after_count = len(df)
                if before_count != after_count:
                    self.logger.info(f"Dropped {before_count - after_count} rows with missing identifiers")
        
        # Get column types from schema
        if self.schema_profile:
            categorical_cols = [col.lower().replace(' ', '_').replace('-', '_') 
                              for col in self.schema_profile['column_types'].get('categorical', [])]
            continuous_cols = [col.lower().replace(' ', '_').replace('-', '_') 
                             for col in self.schema_profile['column_types'].get('continuous_numeric', [])]
            discrete_cols = [col.lower().replace(' ', '_').replace('-', '_') 
                           for col in self.schema_profile['column_types'].get('discrete_numeric', [])]
        else:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            continuous_cols = df.select_dtypes(include=['float64']).columns.tolist()
            discrete_cols = df.select_dtypes(include=['int64']).columns.tolist()
        
        # Handle categorical columns - fill with mode or 'Unknown'
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                if df[col].mode().empty:
                    df[col].fillna('Unknown', inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                self.logger.info(f"Filled missing values in categorical column: {col}")
        
        # Handle continuous numeric columns - fill with median
        for col in continuous_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                self.logger.info(f"Filled missing values in continuous column: {col}")
        
        # Handle discrete numeric columns - fill with mode
        for col in discrete_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                self.logger.info(f"Filled missing values in discrete column: {col}")
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate records"""
        self.logger.info("Removing duplicates...")
        
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        self.cleaning_metrics['operations'].append({
            "operation": "remove_duplicates",
            "duplicates_removed": int(duplicates_removed),
            "rows_remaining": len(df)
        })
        
        return df
    
    def handle_outliers(self, df):
        """Handle outliers in numerical columns with configurable method"""
        self.logger.info("Handling outliers...")
        
        column_types = self.get_column_types(df)
        numerical_cols = column_types.get('continuous_numeric', []) + \
                        column_types.get('discrete_numeric', [])
        
        # Normalize column names
        numerical_cols = [col.lower().replace(' ', '_').replace('-', '_') for col in numerical_cols]
        
        outlier_method = self.config['cleaning']['outlier_method']
        threshold = self.config['cleaning']['outlier_threshold']
        outliers_info = {}
        
        for col in numerical_cols:
            if col not in df.columns or df[col].notna().sum() == 0:
                continue
            
            if outlier_method == 'iqr':
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
            elif outlier_method == 'zscore':
                # Z-score method
                mean = df[col].mean()
                std = df[col].std()
                
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                self.logger.warning(f"Unknown outlier method: {outlier_method}. Skipping.")
                continue
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers = outliers_mask.sum()
            
            if outliers > 0:
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                outliers_info[col] = {
                    "count": int(outliers),
                    "percentage": float(outliers / len(df) * 100),
                    "method": outlier_method,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
                self.logger.info(f"Capped {outliers} outliers in {col} using {outlier_method} method")
        
        self.cleaning_metrics['operations'].append({
            "operation": "handle_outliers",
            "method": outlier_method,
            "threshold": threshold,
            "outliers_capped": outliers_info
        })
        
        return df
    
    def standardize_data_types(self, df):
        """Standardize data types based on detected schema"""
        self.logger.info("Standardizing data types...")
        
        column_types = self.get_column_types(df)
        conversions = {
            "datetime": [],
            "numeric": [],
            "categorical": [],
            "text": []
        }
        
        # Convert datetime columns (normalize names)
        datetime_cols = [col.lower().replace(' ', '_').replace('-', '_') 
                        for col in column_types.get('datetime', [])]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                conversions["datetime"].append(col)
        
        # Convert numerical columns (normalize names)
        numerical_cols = [col.lower().replace(' ', '_').replace('-', '_') 
                         for col in column_types.get('continuous_numeric', []) + column_types.get('discrete_numeric', [])]
        
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                conversions["numeric"].append(col)
        
        # Convert categorical columns to category dtype (normalize names)
        categorical_cols = [col.lower().replace(' ', '_').replace('-', '_') 
                           for col in column_types.get('categorical', [])]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                conversions["categorical"].append(col)
        
        # Strip whitespace from text columns (normalize names)
        text_cols = [col.lower().replace(' ', '_').replace('-', '_') 
                    for col in column_types.get('text', [])]
        for col in text_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                conversions["text"].append(col)
        
        # Strip whitespace from all string columns not yet processed
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col not in conversions["text"] and col not in conversions["categorical"]:
                df[col] = df[col].str.strip()
        
        self.cleaning_metrics['operations'].append({
            "operation": "standardize_data_types",
            "conversions": conversions
        })
        
        self.logger.info(f"Standardized data types: {sum(len(v) for v in conversions.values())} columns")
        
        return df
    
    def normalize_column_names(self, df):
        """Normalize column names (remove special chars, standardize case)"""
        self.logger.info("Normalizing column names...")
        
        original_names = df.columns.tolist()
        
        # Create clean column names
        new_names = []
        for col in df.columns:
            # Convert to lowercase, replace spaces/special chars with underscore
            clean_name = col.lower().strip()
            clean_name = ''.join(c if c.isalnum() else '_' for c in clean_name)
            clean_name = '_'.join(filter(None, clean_name.split('_')))  # Remove multiple underscores
            new_names.append(clean_name)
        
        # Handle duplicate names
        seen = {}
        final_names = []
        for name in new_names:
            if name in seen:
                seen[name] += 1
                final_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                final_names.append(name)
        
        df.columns = final_names
        
        name_mapping = dict(zip(original_names, final_names))
        changed_names = {k: v for k, v in name_mapping.items() if k != v}
        
        if changed_names:
            self.logger.info(f"Normalized {len(changed_names)} column names")
            self.cleaning_metrics['operations'].append({
                "operation": "normalize_column_names",
                "changed_names": changed_names
            })
        
        return df
    
    def remove_constant_columns(self, df):
        """Remove columns with only one unique value"""
        self.logger.info("Checking for constant columns...")
        
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.logger.warning(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
            
            self.cleaning_metrics['operations'].append({
                "operation": "remove_constant_columns",
                "columns_removed": constant_cols
            })
        
        return df
    
    def clean_data(self):
        """Run comprehensive data cleaning pipeline"""
        self.logger.info(f"Starting data cleaning for: {self.dataset_name}")
        
        try:
            # Load validated data
            input_path = Path(self.config['data']['processed_path']) / f"{self.dataset_name}_validated.csv"
            df = pd.read_csv(input_path)
            
            initial_shape = df.shape
            self.logger.info(f"Initial data shape: {initial_shape}")
            
            # Run cleaning operations in sequence
            df = self.normalize_column_names(df)
            df = self.standardize_data_types(df)
            df = self.handle_missing_values(df)
            df = self.remove_duplicates(df)
            df = self.handle_outliers(df)
            df = self.remove_constant_columns(df)
            
            final_shape = df.shape
            self.logger.info(f"Final data shape: {final_shape}")
            
            # Calculate overall metrics
            self.cleaning_metrics['summary'] = {
                "initial_rows": int(initial_shape[0]),
                "initial_columns": int(initial_shape[1]),
                "final_rows": int(final_shape[0]),
                "final_columns": int(final_shape[1]),
                "rows_removed": int(initial_shape[0] - final_shape[0]),
                "columns_removed": int(initial_shape[1] - final_shape[1]),
                "rows_retained_percentage": float((final_shape[0] / initial_shape[0]) * 100),
                "data_quality_score": float((final_shape[0] / initial_shape[0]) * 100)
            }
            
            # Data quality assessment
            null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
            
            self.cleaning_metrics['quality_metrics'] = {
                "null_percentage": float(null_percentage),
                "duplicate_percentage": float(duplicate_percentage),
                "completeness_score": float(100 - null_percentage)
            }
            
            # Save cleaned data
            ensure_dir(self.config['data']['validated_path'])
            output_path = Path(self.config['data']['validated_path']) / f"{self.dataset_name}_cleaned.csv"
            df.to_csv(output_path, index=False)
            self.logger.info(f"Cleaned data saved to {output_path}")
            
            # Save cleaning metrics
            metrics_path = Path(self.config['data']['validated_path']) / f"{self.dataset_name}_cleaning_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.cleaning_metrics, f, indent=2)
            
            self.logger.info(f"Cleaning metrics saved to {metrics_path}")
            
            # Log summary
            self.logger.info("=" * 60)
            self.logger.info("DATA CLEANING SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"Rows: {initial_shape[0]} → {final_shape[0]} ({self.cleaning_metrics['summary']['rows_retained_percentage']:.1f}% retained)")
            self.logger.info(f"Columns: {initial_shape[1]} → {final_shape[1]}")
            self.logger.info(f"Data Quality Score: {self.cleaning_metrics['summary']['data_quality_score']:.1f}%")
            self.logger.info(f"Completeness: {self.cleaning_metrics['quality_metrics']['completeness_score']:.1f}%")
            self.logger.info("=" * 60)
            self.logger.info("✓ Data cleaning completed successfully")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_cleaning.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    cleaner = DataCleaner(dataset_name)
    cleaner.clean_data()