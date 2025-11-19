"""
Unified Utilities for DataCraft ML Pipeline
Works for BOTH data-pipeline and model-training
Combines all functions from both original utils.py files
"""

import os
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional
import chardet
import sys

# For GCP operations
from google.cloud import storage, bigquery


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(log_name):
    """
    Setup logging configuration for pipeline components
    Works in both Docker and local environments
    
    Args:
        log_name (str): Name of the logger (used for log file naming)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Determine log directory based on environment
    if os.environ.get('AIRFLOW_HOME'):
        # Running in Docker/Airflow - use shared logs folder
        log_dir = Path("/opt/airflow/logs")
    else:
        # Running locally - determine project root
        current_file = Path(__file__).resolve()
        
        if current_file.parent.name == 'scripts':
            project_root = current_file.parent.parent
        else:
            project_root = current_file.parent
        
        log_dir = project_root / "logs"
    
    # Create logs directory (with error handling for permissions)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fallback to /tmp if we can't write to the main logs folder
        log_dir = Path("/tmp/airflow-logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Warning: Using fallback log directory: {log_dir}")
    
    # Create unique log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{log_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Detailed formatter for file logs
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simple formatter for console
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler - save everything to file
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        logger.debug(f"Log file created: {log_file}")
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")
    
    # Console handler - only important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def load_config(config_path=None):
    """
    Load pipeline configuration from YAML file
    Automatically resolves correct path inside Docker or locally
    """
    if config_path is None:
        # Detect project root (two levels up from this file)
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config" / "pipeline_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {str(e)}")


def validate_config(config):
    """
    Validate pipeline configuration
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (is_valid, errors)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ['data', 'gcp', 'validation', 'bias', 'cleaning']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration section: {key}")
    
    # Validate data paths
    if 'data' in config:
        required_data_keys = ['raw_path', 'processed_path', 'validated_path']
        for key in required_data_keys:
            if key not in config['data']:
                errors.append(f"Missing required data configuration: {key}")
    
    # Validate GCP config
    if 'gcp' in config:
        required_gcp_keys = ['bucket_name', 'project_id']
        for key in required_gcp_keys:
            if key not in config['gcp']:
                errors.append(f"Missing required GCP configuration: {key}")
    
    return len(errors) == 0, errors


# =============================================================================
# FILE & DIRECTORY UTILITIES
# =============================================================================

def ensure_dir(directory):
    """
    Create directory if it doesn't exist (including parent directories)
    
    Args:
        directory (str or Path): Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def detect_encoding(file_path):
    """
    Detect file encoding using chardet
    
    Args:
        file_path: Path to file
        
    Returns:
        tuple: (encoding, confidence)
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(100000)  # Read first 100KB
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        return encoding, confidence


def get_dataset_path(dataset_name, stage='raw'):
    """
    Get standardized path for dataset at different pipeline stages
    
    Args:
        dataset_name (str): Name of the dataset
        stage (str): Pipeline stage ('raw', 'processed', 'validated')
    
    Returns:
        Path: Path to dataset file
    """
    config = load_config()
    
    stage_paths = {
        'raw': config['data']['raw_path'],
        'processed': config['data']['processed_path'],
        'validated': config['data']['validated_path']
    }
    
    if stage not in stage_paths:
        raise ValueError(f"Invalid stage: {stage}. Must be one of {list(stage_paths.keys())}")
    
    base_path = Path(stage_paths[stage])
    
    # For raw data, just the dataset name with .csv
    if stage == 'raw':
        return base_path / f"{dataset_name}.csv"
    # For processed/validated, add stage suffix
    elif stage == 'processed':
        return base_path / f"{dataset_name}_validated.csv"
    else:  # validated
        return base_path / f"{dataset_name}_cleaned.csv"


def get_file_info(file_path):
    """
    Get metadata about a file
    
    Args:
        file_path (str or Path): Path to file
    
    Returns:
        dict: File metadata (size, modified time, etc.)
    """
    path = Path(file_path)
    
    if not path.exists():
        return None
    
    stat = path.stat()
    
    return {
        "name": path.name,
        "size": stat.st_size,
        "size_formatted": format_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "extension": path.suffix,
        "absolute_path": str(path.absolute())
    }


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_size(size_bytes):
    """
    Format bytes to human-readable size
    
    Args:
        size_bytes (int): Size in bytes
    
    Returns:
        str: Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_duration(seconds):
    """
    Format seconds to human-readable duration
    
    Args:
        seconds (float): Duration in seconds
    
    Returns:
        str: Formatted duration string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# =============================================================================
# GCP UTILITIES (From model-training utils.py)
# =============================================================================

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
    
    # Case 1: Credentials already set
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
    
    # Case 3: Search common locations
    possible_paths = [
        Path(__file__).parent.parent.parent / "gcp" / "service-account.json",
        Path.cwd() / "gcp" / "service-account.json",
        Path.cwd().parent / "gcp" / "service-account.json",
        Path.home() / ".config" / "gcloud" / "service-account.json",
        Path.home() / "gcp" / "service-account.json",
        Path("service-account.json"),
    ]
    
    for cred_path in possible_paths:
        if cred_path.exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(cred_path)
            logger.info(f"✓ Found and set credentials: {cred_path}")
            return True
    
    # Case 4: Try Application Default Credentials
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
        temp_dir = Path("/tmp/datacraft_ml")
    elif os.name == 'nt':  # Windows
        temp_dir = Path("C:/temp/datacraft_ml")
    else:  # Linux/Mac
        temp_dir = Path("/tmp/datacraft_ml")
    
    temp_dir.mkdir(exist_ok=True, parents=True)
    return temp_dir


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
    """
    logger = logging.getLogger(__name__)
    
    if validate:
        if not validate_gcp_setup(project_id, bucket_name, service_account_path, logger):
            raise RuntimeError("GCP setup validation failed")
    
    return get_gcp_clients(project_id, service_account_path, logger)


def validate_gcp_credentials():
    """
    Validate that GCP credentials are properly configured
    
    Returns:
        bool: True if credentials are valid
    
    Raises:
        EnvironmentError: If credentials are not configured
    """
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not credentials_path:
        raise EnvironmentError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable not set. "
            "Please set it to point to your GCP service account key file."
        )
    
    if not Path(credentials_path).exists():
        raise FileNotFoundError(
            f"GCP credentials file not found at: {credentials_path}"
        )
    
    return True


# =============================================================================
# DATA UTILITIES
# =============================================================================

def summarize_dataframe(df, name="dataset"):
    """
    Generate a summary of a pandas DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
        name (str): Name for the dataset
    
    Returns:
        dict: Summary statistics
    """
    return {
        "name": name,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "column_types": df.dtypes.value_counts().to_dict(),
        "null_count": int(df.isnull().sum().sum()),
        "null_percentage": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        "duplicate_count": int(df.duplicated().sum()),
        "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100)
    }


# =============================================================================
# CONSOLE OUTPUT UTILITIES
# =============================================================================

def print_header(text, char="=", width=80):
    """Print a formatted header for console output"""
    print()
    print(char * width)
    print(text.center(width))
    print(char * width)
    print()


def print_success(message):
    """Print a success message with checkmark"""
    print(f"✓ {message}")


def print_warning(message):
    """Print a warning message"""
    print(f"⚠ {message}")


def print_error(message):
    """Print an error message"""
    print(f"✗ {message}")


def print_info(message):
    """Print an info message"""
    print(f"ℹ {message}")


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """Simple progress tracker for pipeline operations"""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step_name=None):
        """Update progress"""
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        message = f"[{self.current_step}/{self.total_steps}] ({percentage:.0f}%) {self.description}"
        if step_name:
            message += f": {step_name}"
        
        print(message)
    
    def complete(self):
        """Mark as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n✓ Completed in {format_duration(elapsed)}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the utilities"""
    print_header("Testing Unified Utils Module")
    
    # Test logging
    logger = setup_logging("test")
    logger.info("Testing logger setup")
    
    # Test config loading (will fail if no config file, that's OK)
    try:
        config = load_config()
        print_success("Configuration loaded successfully")
    except Exception as e:
        print_warning(f"Config not found (OK for testing): {str(e)}")
    
    # Test directory creation
    ensure_dir("test_dir/nested/deep")
    print_success("Directory creation test passed")
    
    # Test file size formatting
    print(f"Size formatting: {format_size(1234567890)}")
    
    # Test duration formatting
    print(f"Duration formatting: {format_duration(3725)}")
    
    # Test environment detection
    print(f"Running in Airflow: {is_running_in_airflow()}")
    print(f"Temp directory: {get_temp_dir()}")
    
    # Test progress tracker
    tracker = ProgressTracker(5, "Testing")
    for i in range(5):
        tracker.update(f"Step {i+1}")
    tracker.complete()
    
    print_success("All utility tests passed!")