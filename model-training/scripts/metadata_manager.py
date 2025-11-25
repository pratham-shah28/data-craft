# scripts/metadata_manager.py
"""
Metadata Storage Manager - BigQuery Only
Stores and retrieves dataset metadata for LLM query generation
NO vector search needed - user selects dataset upfront
"""

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import json
from typing import Dict, Optional, List
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging, setup_gcp_credentials


class BigQueryJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that rounds floats to prevent BigQuery PARSE_JSON errors.
    BigQuery cannot round-trip some floating-point numbers through string representation.
    """
    def __init__(self, float_precision=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.float_precision = float_precision
    
    def encode(self, obj):
        # Recursively round floats in the object
        rounded_obj = self._round_floats(obj)
        return super().encode(rounded_obj)
    
    def _round_floats(self, obj):
        """Recursively round floats in nested structures"""
        if isinstance(obj, float):
            # Round to reasonable precision for BigQuery
            return round(obj, self.float_precision)
        elif isinstance(obj, dict):
            return {k: self._round_floats(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._round_floats(item) for item in obj]
        else:
            return obj


class MetadataManager:
    """
    Manage dataset metadata in BigQuery
    
    Purpose:
    - Store metadata and context for each dataset after processing
    - Quick retrieval when user selects a dataset
    - No semantic search needed (user explicitly chooses dataset)
    
    Flow:
    1. After data processing: Store metadata in BigQuery
    2. User selects dataset from dropdown
    3. Load metadata for that dataset
    4. Use metadata in LLM prompts
    """
    
    def __init__(
        self, 
        project_id: str, 
        dataset_id: str = "datacraft_ml",
        service_account_path: Optional[str] = None
    ):
        """
        Initialize metadata manager
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID (default: 'datacraft_ml')
            service_account_path: Optional path to service account JSON
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.logger = setup_logging(self.__class__.__name__)
        
        # Setup GCP credentials
        setup_gcp_credentials(service_account_path, self.logger)
        
        # Initialize BigQuery client
        self.client = bigquery.Client(project=project_id)
        
        # Ensure metadata table exists
        self._ensure_metadata_table()
    
    def _ensure_metadata_table(self):
        """Create metadata table if it doesn't exist"""
        table_id = f"{self.project_id}.{self.dataset_id}.dataset_metadata"
        
        schema = [
            bigquery.SchemaField("dataset_name", "STRING", mode="REQUIRED", 
                description="Name of the dataset (e.g., 'orders')"),
            bigquery.SchemaField("metadata", "JSON", mode="REQUIRED",
                description="Complete metadata including columns, statistics, relationships"),
            bigquery.SchemaField("llm_context", "STRING", mode="REQUIRED",
                description="Rich text context for LLM prompts"),
            bigquery.SchemaField("row_count", "INTEGER", mode="REQUIRED",
                description="Number of rows in the dataset"),
            bigquery.SchemaField("column_count", "INTEGER", mode="REQUIRED",
                description="Number of columns in the dataset"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED",
                description="When metadata was first created"),
            bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED",
                description="When metadata was last updated"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        try:
            self.client.create_table(table, exists_ok=True)
            self.logger.info(f"✓ Metadata table ready: {table_id}")
        except Exception as e:
            self.logger.error(f"Error creating metadata table: {str(e)}")
            raise
    
    def store_metadata(
        self, 
        dataset_name: str, 
        metadata: dict, 
        llm_context: str
    ):
        """
        Store or update dataset metadata in BigQuery
        
        Args:
            dataset_name: Name of the dataset (e.g., 'orders')
            metadata: Complete metadata dictionary from FeatureEngineer
            llm_context: Rich text context for LLM prompts
            
        Example:
            >>> manager = MetadataManager('my-project')
            >>> manager.store_metadata('orders', metadata, context)
        """
        table_id = f"{self.project_id}.{self.dataset_id}.dataset_metadata"
        
        self.logger.info(f"Storing metadata for: {dataset_name}")
        
        # Use MERGE for upsert (insert or update)
        query = f"""
        MERGE `{table_id}` T
        USING (
          SELECT 
            @dataset_name AS dataset_name,
            PARSE_JSON(@metadata) AS metadata,
            @llm_context AS llm_context,
            @row_count AS row_count,
            @column_count AS column_count,
            CURRENT_TIMESTAMP() AS created_at,
            CURRENT_TIMESTAMP() AS updated_at
        ) S
        ON T.dataset_name = S.dataset_name
        WHEN MATCHED THEN
          UPDATE SET 
            metadata = S.metadata,
            llm_context = S.llm_context,
            row_count = S.row_count,
            column_count = S.column_count,
            updated_at = S.updated_at
        WHEN NOT MATCHED THEN
          INSERT (dataset_name, metadata, llm_context, row_count, column_count, created_at, updated_at)
          VALUES (dataset_name, metadata, llm_context, row_count, column_count, created_at, updated_at)
        """
        
        # Use custom encoder to round floats for BigQuery compatibility
        metadata_json = json.dumps(metadata, cls=BigQueryJSONEncoder, float_precision=6)
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name),
                bigquery.ScalarQueryParameter("metadata", "STRING", metadata_json),
                bigquery.ScalarQueryParameter("llm_context", "STRING", llm_context),
                bigquery.ScalarQueryParameter("row_count", "INT64", metadata.get("row_count", 0)),
                bigquery.ScalarQueryParameter("column_count", "INT64", metadata.get("column_count", 0)),
            ]
        )
        
        self.client.query(query, job_config=job_config).result()
        
        self.logger.info(
            f"✓ Stored metadata for: {dataset_name} "
            f"({metadata.get('row_count', 0):,} rows, "
            f"{metadata.get('column_count', 0)} columns)"
        )
    
    def get_metadata(self, dataset_name: str) -> Optional[Dict]:
        """
        Retrieve metadata for a specific dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with metadata and llm_context, or None if not found
            
        Example:
            >>> manager = MetadataManager('my-project')
            >>> data = manager.get_metadata('orders')
            >>> print(data['llm_context'])
        """
        table_id = f"{self.project_id}.{self.dataset_id}.dataset_metadata"
        
        self.logger.info(f"Retrieving metadata for: {dataset_name}")
        
        query = f"""
        SELECT 
            dataset_name,
            TO_JSON_STRING(metadata) as metadata_json,
            llm_context,
            row_count,
            column_count,
            updated_at
        FROM `{table_id}`
        WHERE dataset_name = @dataset_name
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name)
            ]
        )
        
        results = self.client.query(query, job_config=job_config).result()
        
        for row in results:
            self.logger.info(f"✓ Found metadata for: {dataset_name}")
            
            # Parse the JSON string back to dict
            metadata_dict = json.loads(row.metadata_json)
            
            return {
                "dataset_name": row.dataset_name,
                "metadata": metadata_dict,
                "llm_context": row.llm_context,
                "row_count": row.row_count,
                "column_count": row.column_count,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None
            }
        
        self.logger.warning(f"No metadata found for: {dataset_name}")
        return None
    
    def list_datasets(self) -> List[Dict]:
        """
        List all datasets with metadata available
        
        This can be used to populate a dropdown in the UI
        
        Returns:
            List of dictionaries with dataset info
            
        Example:
            >>> manager = MetadataManager('my-project')
            >>> datasets = manager.list_datasets()
            >>> for ds in datasets:
            >>>     print(f"{ds['name']}: {ds['rows']:,} rows")
        """
        table_id = f"{self.project_id}.{self.dataset_id}.dataset_metadata"
        
        query = f"""
        SELECT 
            dataset_name,
            row_count,
            column_count,
            updated_at
        FROM `{table_id}`
        ORDER BY updated_at DESC
        """
        
        results = self.client.query(query).result()
        
        datasets = [
            {
                "name": row.dataset_name,
                "rows": row.row_count,
                "columns": row.column_count,
                "updated": row.updated_at.isoformat() if row.updated_at else None
            }
            for row in results
        ]
        
        self.logger.info(f"✓ Found {len(datasets)} datasets with metadata")
        
        return datasets
    
    def delete_metadata(self, dataset_name: str):
        """
        Delete metadata for a dataset
        
        Args:
            dataset_name: Name of the dataset to delete
        """
        table_id = f"{self.project_id}.{self.dataset_id}.dataset_metadata"
        
        query = f"""
        DELETE FROM `{table_id}`
        WHERE dataset_name = @dataset_name
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name)
            ]
        )
        
        self.client.query(query, job_config=job_config).result()
        
        self.logger.info(f"✓ Deleted metadata for: {dataset_name}")
    
    def metadata_exists(self, dataset_name: str) -> bool:
        """
        Check if metadata exists for a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if metadata exists, False otherwise
        """
        return self.get_metadata(dataset_name) is not None
    
    def get_table_info(self) -> Dict:
        """Get information about the metadata table itself"""
        table_id = f"{self.project_id}.{self.dataset_id}.dataset_metadata"
        
        try:
            table = self.client.get_table(table_id)
            return {
                "table_id": table_id,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None
            }
        except NotFound:
            return {}


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Local testing script
    Usage: python metadata_manager.py
    """
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import ModelDataLoader
    from feature_engineering import FeatureEngineer
    
    # Configuration
    PROJECT_ID = "datacraft-data-pipeline"
    BUCKET_NAME = "isha-retail-data"
    DATASET_ID = "datacraft_ml"
    DATASET_NAME = "orders"
    
    try:
        print("\n" + "=" * 60)
        print("TESTING METADATA MANAGER")
        print("=" * 60 + "\n")
        
        # Initialize manager
        print("1. Initializing metadata manager...")
        manager = MetadataManager(PROJECT_ID, DATASET_ID)
        
        # Load data and generate features
        print("\n2. Loading data and generating features...")
        loader = ModelDataLoader(BUCKET_NAME, PROJECT_ID, DATASET_ID)
        
        try:
            df = loader.query_bigquery_table(f"{DATASET_NAME}_processed", limit=10000)
        except:
            df = loader.load_processed_data_from_gcs(DATASET_NAME, 'validated')
        
        schema = loader.load_dataset_profile_from_gcs(DATASET_NAME)
        engineer = FeatureEngineer(df, schema, DATASET_NAME)
        
        metadata = engineer.generate_metadata()
        llm_context = engineer.create_llm_context()
        
        # Store metadata
        print("\n3. Storing metadata in BigQuery...")
        manager.store_metadata(DATASET_NAME, metadata, llm_context)
        
        # Retrieve metadata
        print("\n4. Retrieving metadata...")
        retrieved = manager.get_metadata(DATASET_NAME)
        print(f"   Dataset: {retrieved['dataset_name']}")
        print(f"   Rows: {retrieved['row_count']:,}")
        print(f"   Columns: {retrieved['column_count']}")
        print(f"   Updated: {retrieved['updated_at']}")
        
        # Verify metadata structure
        print(f"\n   Metadata keys: {list(retrieved['metadata'].keys())}")
        print(f"   Number of column definitions: {len(retrieved['metadata'].get('columns', []))}")
        
        # List all datasets
        print("\n5. Listing all datasets with metadata:")
        datasets = manager.list_datasets()
        for ds in datasets:
            print(f"   - {ds['name']}: {ds['rows']:,} rows, {ds['columns']} columns")
        
        # Display sample context
        print("\n6. Sample LLM Context (first 300 chars):")
        print(retrieved['llm_context'][:300] + "...")
        
        # Display sample column info
        print("\n7. Sample column metadata (first column):")
        if retrieved['metadata'].get('columns'):
            first_col = retrieved['metadata']['columns'][0]
            print(json.dumps(first_col, indent=2))
        
        # Table info
        print("\n8. Metadata table info:")
        table_info = manager.get_table_info()
        for key, value in table_info.items():
            print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        print("✓ METADATA MANAGER TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)