"""
Model Data Loader for DataCraft ML Pipeline
Loads data from GCS, stores in BigQuery, works locally and in Airflow
"""

from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, Optional, List
from datetime import datetime
import os
import sys
# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils import setup_gcp_credentials, get_temp_dir

class ModelDataLoader:
    """
    Comprehensive data loader for model training pipeline
    
    Features:
    - Load CSV from GCS bucket
    - Store data in BigQuery
    - Load metadata (schema, validation, bias reports) from GCS
    - Works in both local environment and Airflow
    - Automatic credential handling
    """
    
    def __init__(
        self, 
        bucket_name: str, 
        project_id: str, 
        dataset_id: str = "datacraft_ml",
        location: str = "US",
        service_account_path: Optional[str] = None
    ):
        """
        Initialize data loader with GCS and BigQuery clients
        
        Args:
            bucket_name: GCS bucket name (e.g., 'isha-retail-data')
            project_id: GCP project ID (e.g., 'datacraft-data-pipeline')
            dataset_id: BigQuery dataset ID for storing tables (default: 'datacraft_ml')
            location: BigQuery dataset location (default: 'US')
            service_account_path: Optional explicit path to service account JSON
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Setup GCP credentials automatically
        setup_gcp_credentials(service_account_path, self.logger)
        
        # Initialize GCP clients
        self._initialize_clients()
        
        # Create BigQuery dataset if not exists
        self._create_bq_dataset()
    
    def _initialize_clients(self):
        """Initialize Google Cloud Storage and BigQuery clients"""
        try:
            # Initialize clients
            self.storage_client = storage.Client(project=self.project_id)
            self.bq_client = bigquery.Client(project=self.project_id)
            self.bucket = self.storage_client.bucket(self.bucket_name)
            
            # Test connection
            if not self.bucket.exists():
                raise ValueError(f"Bucket does not exist: gs://{self.bucket_name}")
            
            self.logger.info("✓ GCP clients initialized successfully")
            self.logger.info(f"  Project: {self.project_id}")
            self.logger.info(f"  Bucket: gs://{self.bucket_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GCP clients: {str(e)}")
            self.logger.error("\nTroubleshooting:")
            self.logger.error("1. Ensure service-account.json is in the gcp/ folder")
            self.logger.error("2. Or run: gcloud auth application-default login")
            self.logger.error("3. Verify bucket exists and you have access")
            raise
    
    def _create_bq_dataset(self):
        """Create BigQuery dataset if it doesn't exist"""
        dataset_ref = f"{self.project_id}.{self.dataset_id}"
        
        try:
            self.bq_client.get_dataset(dataset_ref)
            self.logger.info(f"✓ BigQuery dataset exists: {dataset_ref}")
        except NotFound:
            # Create dataset
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.location
            dataset.description = "DataCraft ML pipeline data and embeddings"
            
            dataset = self.bq_client.create_dataset(dataset, timeout=30)
            self.logger.info(f"✓ Created BigQuery dataset: {dataset_ref}")
        except Exception as e:
            self.logger.error(f"Error with BigQuery dataset: {str(e)}")
            raise
    
    # ========================================
    # GCS DATA LOADING
    # ========================================
    
    def load_processed_data_from_gcs(
        self, 
        dataset_name: str, 
        stage: str = 'validated'
    ) -> pd.DataFrame:
        """
        Load processed/validated data directly from GCS
        
        Args:
            dataset_name: Name of the dataset (e.g., 'orders')
            stage: Data stage - 'processed' (cleaned) or 'validated'
            
        Returns:
            DataFrame with the loaded data
            
        Example:
            >>> loader = ModelDataLoader('isha-retail-data', 'my-project')
            >>> df = loader.load_processed_data_from_gcs('orders', stage='processed')
            >>> print(f"Loaded {len(df)} rows")
        """
        # Construct GCS path based on actual bucket structure
        if stage == 'processed':
            blob_path = f"data/processed/{dataset_name}_processed.csv"
        elif stage == 'validated':
            blob_path = f"data/validated/{dataset_name}_validated.csv"
        else:
            raise ValueError(
                f"Invalid stage: {stage}. Must be 'processed' or 'validated'"
            )
        
        self.logger.info(f"Loading from gs://{self.bucket_name}/{blob_path}")
        
        blob = self.bucket.blob(blob_path)
        
        if not blob.exists():
            raise FileNotFoundError(
                f"Dataset not found at gs://{self.bucket_name}/{blob_path}\n"
                f"Available datasets: {self.list_available_datasets(stage)}"
            )
        
        # Download to temporary location
        temp_dir = get_temp_dir()
        temp_path = temp_dir / f"{dataset_name}_{stage}.csv"
        
        blob.download_to_filename(str(temp_path))
        
        # Load into DataFrame
        df = pd.read_csv(temp_path)
        
        self.logger.info(
            f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns "
            f"from {blob_path}"
        )
        
        # Log sample info
        self.logger.info(f"  Columns: {list(df.columns)[:5]}...")
        self.logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def list_available_datasets(self, stage: str = 'validated') -> List[str]:
        """
        List all available datasets in GCS
        
        Args:
            stage: 'processed' or 'validated'
            
        Returns:
            List of dataset names (without suffix)
            
        Example:
            >>> loader.list_available_datasets('processed')
            ['orders', 'customers', 'products']
        """
        prefix = f"data/{stage}/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        
        datasets = set()
        for blob in blobs:
            if blob.name.endswith('.csv'):
                filename = Path(blob.name).stem
                # Remove stage suffix: 'orders_processed' -> 'orders'
                dataset_name = filename.replace(f'_{stage}', '')
                datasets.add(dataset_name)
        
        dataset_list = sorted(list(datasets))
        self.logger.info(f"Found {len(dataset_list)} datasets in {prefix}")
        
        return dataset_list
    
    def get_latest_dataset(self, stage: str = 'validated') -> str:
        """
        Get the most recently uploaded dataset name from GCS
        
        Args:
            stage: 'processed' or 'validated'
            
        Returns:
            Dataset name (e.g., 'orders')
            
        Example:
            >>> latest = loader.get_latest_dataset('processed')
            >>> print(f"Latest dataset: {latest}")
        """
        prefix = f"data/{stage}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            raise ValueError(
                f"No datasets found in gs://{self.bucket_name}/{prefix}"
            )
        
        # Filter CSV files only
        csv_blobs = [b for b in blobs if b.name.endswith('.csv')]
        
        if not csv_blobs:
            raise ValueError(
                f"No CSV files found in gs://{self.bucket_name}/{prefix}"
            )
        
        # Get most recent by update time
        latest = max(csv_blobs, key=lambda b: b.updated)
        
        # Extract dataset name: 'data/processed/orders_processed.csv' -> 'orders'
        filename = Path(latest.name).stem
        dataset_name = filename.replace(f'_{stage}', '')
        
        self.logger.info(
            f"Latest {stage} dataset: {dataset_name} "
            f"(updated: {latest.updated.strftime('%Y-%m-%d %H:%M:%S')})"
        )
        
        return dataset_name
    
    # ========================================
    # BIGQUERY OPERATIONS
    # ========================================
    
    def load_to_bigquery(
        self, 
        df: pd.DataFrame, 
        dataset_name: str,
        table_suffix: Optional[str] = "_processed"
    ) -> str:
        """
        Load DataFrame to BigQuery table
        
        Args:
            df: DataFrame to load
            dataset_name: Base name for the table (e.g., 'orders')
            table_suffix: Optional suffix (e.g., '_processed', '_features')
            
        Returns:
            Full table ID (project.dataset.table)
            
        Example:
            >>> table_id = loader.load_to_bigquery(df, 'orders', '_processed')
            >>> print(f"Loaded to: {table_id}")
        """
        table_name = f"{dataset_name}{table_suffix or ''}"
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        self.logger.info(f"Loading to BigQuery: {table_id}")
        
        # Configure load job
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        )
        
        # Load data
        load_job = self.bq_client.load_table_from_dataframe(
            df, 
            table_id, 
            job_config=job_config
        )
        
        # Wait for job to complete
        load_job.result()
        
        # Get table info
        table = self.bq_client.get_table(table_id)
        
        self.logger.info(
            f"✓ Loaded {table.num_rows:,} rows to {table_id} "
            f"({table.num_bytes / 1024**2:.2f} MB)"
        )
        
        return table_id
    
    def query_bigquery_table(
        self, 
        table_name: str, 
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query data from BigQuery table
        
        Args:
            table_name: Name of the table (without dataset prefix)
            limit: Optional row limit
            
        Returns:
            DataFrame with query results
            
        Example:
            >>> df = loader.query_bigquery_table('orders_processed', limit=1000)
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        self.logger.info(f"Querying BigQuery table: {table_name}")
        
        df = self.bq_client.query(query).to_dataframe()
        
        self.logger.info(f"✓ Queried {len(df):,} rows from {table_name}")
        
        return df
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Get information about a BigQuery table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table metadata
            
        Example:
            >>> info = loader.get_table_info('orders_processed')
            >>> print(f"Rows: {info['num_rows']:,}")
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        try:
            table = self.bq_client.get_table(table_id)
            
            return {
                "table_id": table_id,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None,
                "schema": [
                    {"name": field.name, "type": field.field_type}
                    for field in table.schema
                ]
            }
        except NotFound:
            self.logger.warning(f"Table not found: {table_id}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting table info: {str(e)}")
            return {}
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a BigQuery table exists
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists, False otherwise
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        try:
            self.bq_client.get_table(table_id)
            return True
        except NotFound:
            return False
    
    # ========================================
    # METADATA LOADING FROM GCS
    # ========================================
    
    def load_dataset_profile_from_gcs(self, dataset_name: str) -> dict:
        """
        Load schema profile for the dataset from GCS
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing schema profile
            
        Example:
            >>> schema = loader.load_dataset_profile_from_gcs('orders')
            >>> print(f"Columns: {len(schema.get('columns', []))}")
        """
        blob_path = f"reports/schema/{dataset_name}_schema_report.json"
        blob = self.bucket.blob(blob_path)
        
        if not blob.exists():
            self.logger.warning(
                f"Schema profile not found at gs://{self.bucket_name}/{blob_path}"
            )
            return {"dataset_name": dataset_name, "columns": []}
        
        profile_json = blob.download_as_text()
        profile = json.loads(profile_json)
        
        self.logger.info(
            f"✓ Loaded schema profile: {len(profile.get('columns', []))} columns"
        )
        
        return profile
    
    def load_validation_report_from_gcs(self, dataset_name: str) -> dict:
        """
        Load validation report from GCS
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Validation report dictionary
        """
        blob_path = f"reports/validation/{dataset_name}_validation_report.json"
        blob = self.bucket.blob(blob_path)
        
        if not blob.exists():
            self.logger.warning(f"Validation report not found at {blob_path}")
            return {}
        
        report_json = blob.download_as_text()
        report = json.loads(report_json)
        
        self.logger.info("✓ Loaded validation report")
        
        return report
    
    def load_bias_report_from_gcs(self, dataset_name: str) -> dict:
        """
        Load bias report from GCS
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Bias report dictionary
        """
        blob_path = f"reports/bias/{dataset_name}_bias_report.json"
        blob = self.bucket.blob(blob_path)
        
        if not blob.exists():
            self.logger.warning(f"Bias report not found at {blob_path}")
            return {}
        
        report_json = blob.download_as_text()
        report = json.loads(report_json)
        
        self.logger.info("✓ Loaded bias report")
        
        return report
    
    def load_cleaning_metrics_from_gcs(self, dataset_name: str) -> dict:
        """
        Load cleaning metrics from GCS
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Cleaning metrics dictionary
        """
        blob_path = f"reports/cleaning/{dataset_name}_cleaning_report.json"
        blob = self.bucket.blob(blob_path)
        
        if not blob.exists():
            self.logger.warning(f"Cleaning metrics not found at {blob_path}")
            return {}
        
        report_json = blob.download_as_text()
        report = json.loads(report_json)
        
        self.logger.info("✓ Loaded cleaning metrics")
        
        return report
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def get_gcs_file_info(self, blob_path: str) -> Dict:
        """
        Get information about a file in GCS
        
        Args:
            blob_path: Path to blob in GCS
            
        Returns:
            Dictionary with file metadata
        """
        blob = self.bucket.blob(blob_path)
        
        if not blob.exists():
            return {}
        
        blob.reload()
        
        return {
            "name": blob.name,
            "size": blob.size,
            "size_mb": blob.size / 1024**2,
            "updated": blob.updated.isoformat() if blob.updated else None,
            "content_type": blob.content_type,
            "public_url": blob.public_url
        }
    
    def download_from_gcs(
        self, 
        blob_path: str, 
        local_path: str
    ) -> str:
        """
        Download a file from GCS to local path
        
        Args:
            blob_path: Path to blob in GCS
            local_path: Local destination path
            
        Returns:
            Local file path
        """
        blob = self.bucket.blob(blob_path)
        
        if not blob.exists():
            raise FileNotFoundError(
                f"File not found: gs://{self.bucket_name}/{blob_path}"
            )
        
        # Ensure directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        blob.download_to_filename(local_path)
        
        self.logger.info(f"✓ Downloaded to {local_path}")
        
        return local_path
    
    def get_pipeline_summary(self, dataset_name: str) -> Dict:
        """
        Get comprehensive summary of dataset and pipeline artifacts
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with pipeline summary
        """
        summary = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "gcs": {
                "bucket": self.bucket_name,
                "processed_file": None,
                "validated_file": None
            },
            "bigquery": {
                "project": self.project_id,
                "dataset": self.dataset_id,
                "table": None,
                "table_info": None
            },
            "reports": {
                "schema": None,
                "validation": None,
                "bias": None,
                "cleaning": None
            }
        }
        
        # Check GCS files
        for stage in ['processed', 'validated']:
            blob_path = f"data/{stage}/{dataset_name}_{stage}.csv"
            if self.bucket.blob(blob_path).exists():
                summary["gcs"][f"{stage}_file"] = f"gs://{self.bucket_name}/{blob_path}"
        
        # Check BigQuery table
        table_name = f"{dataset_name}_processed"
        if self.table_exists(table_name):
            summary["bigquery"]["table"] = f"{self.project_id}.{self.dataset_id}.{table_name}"
            summary["bigquery"]["table_info"] = self.get_table_info(table_name)
        
        # Check reports
        summary["reports"]["schema"] = bool(self.load_dataset_profile_from_gcs(dataset_name).get("columns"))
        summary["reports"]["validation"] = bool(self.load_validation_report_from_gcs(dataset_name))
        summary["reports"]["bias"] = bool(self.load_bias_report_from_gcs(dataset_name))
        summary["reports"]["cleaning"] = bool(self.load_cleaning_metrics_from_gcs(dataset_name))
        
        return summary


# ========================================
# STANDALONE USAGE
# ========================================

if __name__ == "__main__":
    """
    Standalone test script
    Usage: python model_data_loader.py
    """
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    PROJECT_ID = "datacraft-data-pipeline"
    BUCKET_NAME = "isha-retail-data"
    DATASET_ID = "datacraft_ml"
    DATASET_NAME = "orders"
    
    try:
        # Initialize loader
        print("\n" + "=" * 60)
        print("TESTING MODEL DATA LOADER")
        print("=" * 60 + "\n")
        
        loader = ModelDataLoader(
            bucket_name=BUCKET_NAME,
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID
        )
        
        # Test 1: List datasets
        print("\n1. Available datasets:")
        datasets = loader.list_available_datasets('processed')
        for ds in datasets:
            print(f"   - {ds}")
        
        # Test 2: Load from GCS
        print(f"\n2. Loading dataset '{DATASET_NAME}' from GCS...")
        df = loader.load_processed_data_from_gcs(DATASET_NAME, stage='validated')
        print(f"   Preview:\n{df.head()}")
        
        # Test 3: Load to BigQuery
        print(f"\n3. Loading to BigQuery...")
        table_id = loader.load_to_bigquery(df, DATASET_NAME, table_suffix="_test")
        
        # Test 4: Load metadata
        print(f"\n4. Loading metadata...")
        schema = loader.load_dataset_profile_from_gcs(DATASET_NAME)
        validation = loader.load_validation_report_from_gcs(DATASET_NAME)
        bias = loader.load_bias_report_from_gcs(DATASET_NAME)
        
        # Test 5: Summary
        print(f"\n5. Pipeline summary:")
        summary = loader.get_pipeline_summary(DATASET_NAME)
        print(json.dumps(summary, indent=2))
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)