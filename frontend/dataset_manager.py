# frontend/dataset_manager.py

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# ✅ Add model-training scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'model-training' / 'scripts'))

from data_loader import ModelDataLoader
from feature_engineering import FeatureEngineer
from metadata_manager import MetadataManager


class DatasetManager:
    """
    Manage multiple datasets dynamically
    Discovers datasets from BigQuery metadata table
    Also supports dataset upload via UI using SAME logic as add_new_dataset.py
    """

    def __init__(self, project_id: str, dataset_id: str, bucket_name: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bucket_name = bucket_name

        self.metadata_manager = MetadataManager(project_id, dataset_id)
        self.loader = ModelDataLoader(bucket_name, project_id, dataset_id)

    # ---------------------------------
    # ✅ CLEANING (MATCHES LOCAL LOGIC)
    # ---------------------------------
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )

        df = df.drop_duplicates()

        for col in df.select_dtypes(include=["float", "int"]).columns:
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("unknown")

        return df

    # ---------------------------------
    # ✅ MAIN UI UPLOAD PIPELINE
    # ---------------------------------
    def upload_dataset_from_ui(self, uploaded_file, dataset_name: str) -> dict:
        """
        Uses SAME steps as add_new_dataset.py:

        1. Read CSV
        2. Clean
        3. Load to BigQuery
        4. Feature engineer metadata
        5. Store metadata
        """

        try:
            # ✅ Step 1: Read CSV
            df = pd.read_csv(uploaded_file)

            # ✅ Step 2: Clean
            df = self.clean_dataset(df)

            # ✅ Step 3: Load to BigQuery (EXACT SAME METHOD)
            table_id = self.loader.load_to_bigquery(
                df, dataset_name, table_suffix="_processed"
            )

            # ✅ Step 4: Feature Engineering + Metadata
            engineer = FeatureEngineer(df, {}, dataset_name)
            metadata = engineer.generate_metadata()
            llm_context = engineer.create_llm_context()

            # ✅ Step 5: Store Metadata (EXACT SAME METHOD)
            self.metadata_manager.store_metadata(
                dataset_name,
                metadata,
                llm_context
            )

            return {
                "status": "success",
                "table_id": table_id,
                "rows": len(df),
                "columns": len(df.columns)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    # ---------------------------------
    # ✅ EXISTING READ METHODS (UNCHANGED)
    # ---------------------------------
    def get_available_datasets(self) -> list:
        try:
            return self.metadata_manager.list_datasets()
        except Exception:
            return [{
                'name': 'orders',
                'rows': 51290,
                'columns': 23,
                'updated': None
            }]

    def get_dataset_info(self, dataset_name: str) -> dict:
        try:
            metadata = self.metadata_manager.get_metadata(dataset_name)

            if metadata:
                return {
                    'name': dataset_name,
                    'row_count': metadata.get('row_count', 0),
                    'column_count': metadata.get('column_count', 0),
                    'columns': metadata.get('metadata', {}).get('columns', []),
                    'updated': metadata.get('updated_at'),
                    'llm_context': metadata.get('llm_context', ''),
                    'table_name': f"{dataset_name}_processed"
                }
            return None
        except Exception:
            return None

    def get_dataset_columns(self, dataset_name: str) -> list:
        info = self.get_dataset_info(dataset_name)
        if info and 'columns' in info:
            return [col.get('name') for col in info['columns']]
        return []

    def get_sample_queries(self, dataset_name: str) -> list:
        return [
            f"Show me insights from {dataset_name}",
            f"What trends exist in {dataset_name}?",
            f"Summarize key metrics from {dataset_name}"
        ]
