# scripts/feature_engineering.py
"""
Feature Engineering for Query & Visualization Model
Generates metadata and context for the selected dataset
NO vector search - metadata stored directly in BigQuery
"""

import pandas as pd
from typing import Dict, List, Optional
import numpy as np
import json
import logging
from pathlib import Path
import sys
import warnings

# Suppress dateutil warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging, get_temp_dir


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class FeatureEngineer:
    """
    Generate metadata and context for LLM-based query generation
    
    Purpose:
    - Extract schema information from dataset
    - Generate rich context for LLM prompts
    - Identify relationships for query generation
    - No vector embeddings needed (user selects dataset)
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        schema_profile: Optional[dict] = None,
        dataset_name: str = "dataset"
    ):
        """
        Initialize feature engineer
        
        Args:
            df: Source DataFrame from BigQuery
            schema_profile: Optional pre-loaded schema from GCS
            dataset_name: Name of the dataset (selected by user)
        """
        self.df = df
        self.schema = schema_profile or {}
        self.dataset_name = dataset_name
        self.logger = setup_logging(self.__class__.__name__)
        
        # Parse schema if it exists
        self._schema_dict = self._parse_schema_profile()
        
        # Cache for expensive operations
        self._metadata_cache = None
        self._context_cache = None
    
    def _parse_schema_profile(self) -> Dict[str, Dict]:
        """Parse schema profile into a dictionary keyed by column name"""
        schema_dict = {}
        
        if not self.schema or "columns" not in self.schema:
            return schema_dict
        
        columns = self.schema.get("columns", [])
        
        if isinstance(columns, list):
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name")
                    if col_name:
                        schema_dict[col_name] = col
                elif isinstance(col, str):
                    schema_dict[col] = {"name": col}
        elif isinstance(columns, dict):
            schema_dict = columns
        
        return schema_dict
    
    def generate_metadata(self) -> Dict:
        """
        Generate comprehensive metadata for LLM context
        
        This metadata will be stored in BigQuery and retrieved when
        the user asks questions about this dataset.
        
        Returns:
            Dictionary with table metadata, column info, statistics, samples
        """
        if self._metadata_cache:
            return self._metadata_cache
        
        self.logger.info(f"Generating metadata for {self.dataset_name}")
        
        metadata = {
            "table_name": self.dataset_name,
            "row_count": int(len(self.df)),
            "column_count": int(len(self.df.columns)),
            "memory_mb": float(self.df.memory_usage(deep=True).sum() / 1024**2),
            "columns": [],
            "sample_values": {},
            "statistics": {},
            "relationships": self._infer_relationships(),
            "date_columns": self._identify_date_columns(),
            "numeric_columns": self._identify_numeric_columns(),
            "categorical_columns": self._identify_categorical_columns()
        }
        
        # Process each column
        for col in self.df.columns:
            col_info = self._generate_column_metadata(col)
            metadata["columns"].append(col_info)
            
            # Add sample values for categorical/low-cardinality columns
            if self._is_categorical_or_low_cardinality(col):
                metadata["sample_values"][col] = self._get_sample_values(col)
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                metadata["statistics"][col] = self._get_numeric_statistics(col)
        
        self._metadata_cache = metadata
        
        self.logger.info(
            f"✓ Generated metadata: {len(metadata['columns'])} columns, "
            f"{len(metadata['sample_values'])} categorical, "
            f"{len(metadata['statistics'])} numeric"
        )
        
        return metadata
    
    def _generate_column_metadata(self, col: str) -> Dict:
        """Generate metadata for a single column"""
        dtype = self.df[col].dtype
        
        col_info = {
            "name": col,
            "type": str(dtype),
            "description": self._infer_column_description(col),
            "unique_values": int(self.df[col].nunique()),
            "null_count": int(self.df[col].isnull().sum()),
            "null_percentage": float(self.df[col].isnull().sum() / len(self.df) * 100),
            "is_primary_key": bool(self._is_likely_primary_key(col)),
            "is_foreign_key": bool(self._is_likely_foreign_key(col)),
            "cardinality": self._categorize_cardinality(col)
        }
        
        # Add type-specific metadata
        if pd.api.types.is_numeric_dtype(dtype):
            col_info["data_class"] = "numeric"
            col_info["range"] = {
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max())
            }
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_info["data_class"] = "datetime"
            col_info["range"] = {
                "min": str(self.df[col].min()),
                "max": str(self.df[col].max())
            }
        else:
            col_info["data_class"] = "categorical"
        
        return col_info
    
    def _infer_column_description(self, col_name: str) -> str:
        """Infer human-readable description from column name"""
        # Check schema profile first
        if col_name in self._schema_dict:
            col_schema = self._schema_dict[col_name]
            if isinstance(col_schema, dict):
                desc = col_schema.get("description")
                if desc and desc != col_name:
                    return desc
        
        # Heuristic-based inference
        col_lower = col_name.lower()
        
        # Common ID patterns
        if 'id' in col_lower and (col_lower.endswith('id') or col_lower.endswith('_id')):
            base = col_name[:-2] if col_name.endswith('ID') else col_name[:-3]
            return f"{base.replace('_', ' ').title()} Identifier"
        
        # Date/time patterns
        if 'date' in col_lower or 'time' in col_lower:
            return f"{col_name.replace('_', ' ').title()} Timestamp"
        
        # Financial patterns
        if any(x in col_lower for x in ['amount', 'price', 'cost', 'profit', 'revenue', 'sales']):
            return f"{col_name.replace('_', ' ').title()} Amount"
        
        # Quantity patterns
        if any(x in col_lower for x in ['quantity', 'count', 'qty']):
            return f"{col_name.replace('_', ' ').title()} Count"
        
        # Geographic patterns
        if any(x in col_lower for x in ['city', 'state', 'country', 'region']):
            return f"{col_name.replace('_', ' ').title()} Location"
        
        # Default: title case with underscores replaced
        return col_name.replace('_', ' ').title()
    
    def _is_categorical_or_low_cardinality(self, col: str, threshold: int = 50) -> bool:
        """Check if column is categorical or has low cardinality"""
        return (
            self.df[col].dtype == 'object' or 
            self.df[col].nunique() < threshold
        )
    
    def _get_sample_values(self, col: str, top_n: int = 10) -> Dict:
        """Get top N most common values with counts"""
        value_counts = self.df[col].value_counts().head(top_n)
        return {str(k): int(v) for k, v in value_counts.items()}
    
    def _get_numeric_statistics(self, col: str) -> Dict:
        """Get comprehensive statistics for numeric column"""
        series = self.df[col].dropna()
        
        if len(series) == 0:
            return {}
        
        return {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75))
        }
    
    def _is_likely_primary_key(self, col: str) -> bool:
        """Heuristic to identify potential primary keys"""
        col_lower = col.lower()
        is_id_column = (
            col_lower.endswith('_id') or 
            col_lower == 'id' or
            col_lower == 'row_id'
        )
        is_unique = self.df[col].nunique() == len(self.df)
        has_no_nulls = self.df[col].isnull().sum() == 0
        
        return is_id_column and is_unique and has_no_nulls
    
    def _is_likely_foreign_key(self, col: str) -> bool:
        """Heuristic to identify potential foreign keys"""
        col_lower = col.lower()
        is_fk_pattern = (
            col_lower.endswith('_id') and 
            col_lower != 'id' and
            col_lower != 'row_id' and
            not self._is_likely_primary_key(col)
        )
        cardinality_ratio = self.df[col].nunique() / len(self.df)
        has_duplicates = cardinality_ratio < 0.9
        
        return is_fk_pattern and has_duplicates
    
    def _categorize_cardinality(self, col: str) -> str:
        """Categorize column by cardinality"""
        unique_count = self.df[col].nunique()
        row_count = len(self.df)
        ratio = unique_count / row_count if row_count > 0 else 0
        
        if ratio > 0.95:
            return "high"
        elif ratio > 0.5:
            return "medium-high"
        elif ratio > 0.1:
            return "medium"
        elif ratio > 0.01:
            return "low"
        else:
            return "very-low"
    
    def _identify_date_columns(self) -> List[str]:
        """Identify all datetime columns"""
        date_cols = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                date_cols.append(col)
        return date_cols
    
    def _identify_numeric_columns(self) -> List[str]:
        """Identify all numeric columns"""
        return [
            col for col in self.df.columns 
            if pd.api.types.is_numeric_dtype(self.df[col])
        ]
    
    def _identify_categorical_columns(self) -> List[str]:
        """Identify categorical columns"""
        return [
            col for col in self.df.columns 
            if self._is_categorical_or_low_cardinality(col)
        ]
    
    def _infer_relationships(self) -> List[Dict]:
        """
        Infer dimension-measure pairs for aggregations
        These help the LLM generate better queries
        """
        relationships = []
        
        categorical_cols = self._identify_categorical_columns()
        numeric_cols = self._identify_numeric_columns()
        
        # Find dimension-measure pairs
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                # Skip high cardinality and ID columns
                if self.df[cat_col].nunique() > 100:
                    continue
                if cat_col.lower().endswith('_id') or num_col.lower().endswith('_id'):
                    continue
                
                relationships.append({
                    "type": "dimension_measure",
                    "dimension": cat_col,
                    "measure": num_col,
                    "suggested_aggregations": ["SUM", "AVG", "COUNT", "MIN", "MAX"]
                })
        
        return relationships[:20]  # Limit to top 20
    
    def create_llm_context(self) -> str:
        """
        Create rich text context for LLM prompts
        
        This context will be included in the prompt when the user
        asks questions, helping the LLM understand the data structure.
        
        Returns:
            Formatted text describing the dataset
        """
        if self._context_cache:
            return self._context_cache
        
        metadata = self.generate_metadata()
        
        context_parts = [
            f"Dataset: {self.dataset_name}",
            f"Description: Business data table with {metadata['row_count']:,} records",
            f"Total Records: {metadata['row_count']:,}",
            f"Total Columns: {metadata['column_count']}",
            f"\nColumn Definitions:"
        ]
        
        # Add column descriptions
        for col_info in metadata["columns"]:
            col_desc = (
                f"\n- {col_info['name']}: {col_info['description']} "
                f"({col_info['data_class']}, {col_info['unique_values']} unique values)"
            )
            context_parts.append(col_desc)
        
        # Add sample values for key categorical columns
        if metadata["sample_values"]:
            context_parts.append("\n\nKey Categories and Sample Values:")
            for col, samples in list(metadata["sample_values"].items())[:5]:
                sample_str = ", ".join([f"{k} ({v})" for k, v in list(samples.items())[:3]])
                context_parts.append(f"\n- {col}: {sample_str}")
        
        # Add common analysis patterns
        if metadata["relationships"]:
            context_parts.append("\n\nCommon Analysis Patterns:")
            for rel in metadata["relationships"][:5]:
                context_parts.append(
                    f"\n- Analyze {rel['measure']} by {rel['dimension']}"
                )
        
        self._context_cache = "\n".join(context_parts)
        
        self.logger.info(
            f"✓ Generated LLM context: {len(self._context_cache)} characters"
        )
        
        return self._context_cache
    
    def save_metadata(self, output_path: str):
        """Save generated metadata to JSON file (for debugging/inspection)"""
        metadata = self.generate_metadata()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"✓ Saved metadata to {output_path}")
    
    def get_feature_summary(self) -> Dict:
        """Get summary of feature engineering results"""
        metadata = self.generate_metadata()
        
        return {
            "dataset": self.dataset_name,
            "total_rows": metadata["row_count"],
            "total_columns": metadata["column_count"],
            "numeric_columns": len(metadata["numeric_columns"]),
            "categorical_columns": len(metadata["categorical_columns"]),
            "date_columns": len(metadata["date_columns"]),
            "relationships_found": len(metadata["relationships"]),
            "primary_keys": sum(
                1 for col in metadata["columns"] if col.get("is_primary_key")
            ),
            "foreign_keys": sum(
                1 for col in metadata["columns"] if col.get("is_foreign_key")
            )
        }


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Local testing script
    Usage: python feature_engineering.py
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import ModelDataLoader
    
    # Configuration
    PROJECT_ID = "datacraft-data-pipeline"
    BUCKET_NAME = "isha-retail-data"
    DATASET_ID = "datacraft_ml"
    DATASET_NAME = "orders"
    
    try:
        print("\n" + "=" * 60)
        print("TESTING FEATURE ENGINEERING")
        print("=" * 60 + "\n")
        
        # Load data
        print("1. Loading data from BigQuery...")
        loader = ModelDataLoader(BUCKET_NAME, PROJECT_ID, DATASET_ID)
        
        try:
            df = loader.query_bigquery_table(f"{DATASET_NAME}_processed", limit=10000)
        except:
            print("   BigQuery table not found, loading from GCS...")
            df = loader.load_processed_data_from_gcs(DATASET_NAME, 'validated')
        
        print(f"   Loaded {len(df):,} rows")
        
        # Load schema
        schema = loader.load_dataset_profile_from_gcs(DATASET_NAME)
        
        # Initialize feature engineer
        print("\n2. Generating features...")
        engineer = FeatureEngineer(df, schema, DATASET_NAME)
        
        # Generate metadata
        metadata = engineer.generate_metadata()
        print(f"   ✓ Generated metadata for {len(metadata['columns'])} columns")
        
        # Generate LLM context
        context = engineer.create_llm_context()
        print(f"   ✓ Generated LLM context ({len(context)} chars)")
        
        # Get summary
        print("\n3. Feature Summary:")
        summary = engineer.get_feature_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Save outputs
        temp_dir = get_temp_dir()
        metadata_path = temp_dir / f"{DATASET_NAME}_metadata.json"
        engineer.save_metadata(str(metadata_path))
        
        context_path = temp_dir / f"{DATASET_NAME}_context.txt"
        with open(context_path, 'w') as f:
            f.write(context)
        print(f"\n   ✓ Saved outputs to {temp_dir}")
        
        # Display samples
        print("\n4. Sample Metadata (first column):")
        print(json.dumps(metadata["columns"][0], indent=2))
        
        print("\n5. Sample LLM Context (first 500 chars):")
        print(context[:500] + "...")
        
        print("\n6. Sample Relationships (first 3):")
        for rel in metadata["relationships"][:3]:
            print(f"   - {rel['dimension']} × {rel['measure']}")
        
        print("\n" + "=" * 60)
        print("✓ FEATURE ENGINEERING TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)