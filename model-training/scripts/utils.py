"""
GCP Utilities for DataCraft ML Pipeline
Automatically handles credentials for local development and Airflow
"""

import os
import logging
from pathlib import Path
from typing import Optional
from google.cloud import storage, bigquery


def setup_gcp_credentials(
    service_account_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Automatically setup GCP credentials for local and Airflow environments
    
    This function:
    1. Checks if credentials are already set (Airflow case)
    2. Tries to find service account JSON in common locations (local case)
    3. Falls back to Application Default Credentials
    
    Args:
        service_account_path: Optional explicit path to service account JSON
        logger: Optional logger for output
        
    Returns:
        True if credentials are set/found, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Case 1: Credentials already set (Airflow, CI/CD, or already configured)
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        logger.info(f"✓ Using existing credentials: {cred_path}")
        return True
    
    # Case 2: Explicit path provided
    if service_account_path:
        if Path(service_account_path).exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
            logger.info(f"✓ Set credentials from provided path: {service_account_path}")
            return True
        else:
            logger.warning(f"Provided credentials path not found: {service_account_path}")
    
    # Case 3: Search common local development locations
    possible_paths = [
        # Relative to this file (model-training/utils/gcp_utils.py -> ../../gcp/)
        Path(__file__).parent.parent.parent / "gcp" / "service-account.json",
        # Relative to current working directory
        Path.cwd() / "gcp" / "service-account.json",
        Path.cwd().parent / "gcp" / "service-account.json",
        # Home directory
        Path.home() / ".config" / "gcloud" / "service-account.json",
        Path.home() / "gcp" / "service-account.json",
        # Current directory
        Path("service-account.json"),
    ]
    
    for cred_path in possible_paths:
        if cred_path.exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(cred_path)
            logger.info(f"✓ Found and set credentials: {cred_path}")
            return True
    
    # Case 4: Try Application Default Credentials (gcloud auth application-default login)
    logger.info("No service account file found. Attempting Application Default Credentials...")
    logger.info("If this fails, run: gcloud auth application-default login")
    return False


def get_gcp_clients(
    project_id: str,
    service_account_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Get initialized GCP clients with automatic credential handling
    
    Args:
        project_id: GCP project ID
        service_account_path: Optional path to service account JSON
        logger: Optional logger
        
    Returns:
        Tuple of (storage_client, bigquery_client)
        
    Example:
        >>> storage_client, bq_client = get_gcp_clients('my-project')
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Setup credentials
    setup_gcp_credentials(service_account_path, logger)
    
    try:
        # Initialize clients
        storage_client = storage.Client(project=project_id)
        bq_client = bigquery.Client(project=project_id)
        
        logger.info(f"✓ GCP clients initialized for project: {project_id}")
        return storage_client, bq_client
        
    except Exception as e:
        logger.error(f"Failed to initialize GCP clients: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check if GOOGLE_APPLICATION_CREDENTIALS is set correctly")
        logger.error("2. Run: gcloud auth application-default login")
        logger.error("3. Ensure service-account.json exists in gcp/ folder")
        raise


def validate_gcp_setup(
    project_id: str,
    bucket_name: str,
    service_account_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate GCP setup by testing access to project and bucket
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name to test
        service_account_path: Optional path to service account JSON
        logger: Optional logger
        
    Returns:
        True if setup is valid, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Get clients
        storage_client, bq_client = get_gcp_clients(
            project_id, 
            service_account_path, 
            logger
        )
        
        # Test storage access
        bucket = storage_client.bucket(bucket_name)
        if bucket.exists():
            logger.info(f"✓ Bucket accessible: gs://{bucket_name}")
        else:
            logger.error(f"✗ Bucket not found: gs://{bucket_name}")
            return False
        
        # Test BigQuery access
        datasets = list(bq_client.list_datasets(max_results=1))
        logger.info(f"✓ BigQuery access confirmed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ GCP validation failed: {str(e)}")
        return False


def is_running_in_airflow() -> bool:
    """
    Detect if code is running in Airflow environment
    
    Returns:
        True if running in Airflow, False otherwise
    """
    # Check for Airflow-specific environment variables
    airflow_indicators = [
        'AIRFLOW_HOME',
        'AIRFLOW__CORE__DAGS_FOLDER',
        'AIRFLOW_CONFIG'
    ]
    
    return any(os.getenv(var) for var in airflow_indicators)


def get_temp_dir() -> Path:
    """
    Get appropriate temporary directory for current environment
    
    Returns:
        Path object for temp directory
    """
    if is_running_in_airflow():
        # Airflow typically uses /tmp
        temp_dir = Path("/tmp/datacraft_ml")
    elif os.name == 'nt':  # Windows
        temp_dir = Path("C:/temp/datacraft_ml")
    else:  # Linux/Mac
        temp_dir = Path("/tmp/datacraft_ml")
    
    temp_dir.mkdir(exist_ok=True, parents=True)
    return temp_dir


# Convenience function for quick setup
def quick_setup(
    project_id: str,
    bucket_name: str,
    service_account_path: Optional[str] = None,
    validate: bool = True
):
    """
    Quick setup function for scripts
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name
        service_account_path: Optional path to service account
        validate: Whether to validate setup
        
    Returns:
        Tuple of (storage_client, bq_client) if successful
        
    Example:
        >>> from utils.gcp_utils import quick_setup
        >>> storage, bq = quick_setup('my-project', 'my-bucket')
    """
    logger = logging.getLogger(__name__)
    
    if validate:
        if not validate_gcp_setup(project_id, bucket_name, service_account_path, logger):
            raise RuntimeError("GCP setup validation failed")
    
    return get_gcp_clients(project_id, service_account_path, logger)


if __name__ == "__main__":
    """Test the utilities"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 60)
    print("TESTING GCP UTILITIES")
    print("=" * 60 + "\n")
    
    # Test credential detection
    print("1. Testing credential detection...")
    setup_gcp_credentials()
    
    # Test environment detection
    print(f"\n2. Running in Airflow: {is_running_in_airflow()}")
    print(f"   Temp directory: {get_temp_dir()}")
    
    # Test with actual project (update these values)
    PROJECT_ID = "datacraft-data-pipeline"
    BUCKET_NAME = "isha-retail-data"
    
    print(f"\n3. Validating GCP setup...")
    if validate_gcp_setup(PROJECT_ID, BUCKET_NAME):
        print("✓ GCP setup is valid!")
    else:
        print("✗ GCP setup validation failed")
    
    print("\n" + "=" * 60)