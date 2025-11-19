import pandas as pd
import shutil
import chardet
from pathlib import Path
from utils import setup_logging, load_config, ensure_dir
from schema_detector import SchemaDetector

def detect_encoding(file_path):
    """Detect file encoding using chardet"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(100000)  # Read first 100KB
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        return encoding, confidence

def detect_file_format(file_path):
    """Detect file format from extension"""
    extension = Path(file_path).suffix.lower()
    format_map = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json',
        '.parquet': 'parquet',
        '.txt': 'text'
    }
    return format_map.get(extension, 'unknown')

def read_data_file(file_path):
    """Read data file based on format with automatic encoding detection"""
    file_format = detect_file_format(file_path)
    logger = setup_logging("data_acquisition")
    
    if file_format == 'csv':
        # Detect encoding for CSV files
        encoding, confidence = detect_encoding(file_path)
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
        
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            logger.warning(f"Failed with detected encoding {encoding}, trying common encodings...")
            # Fallback to common encodings
            for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
                try:
                    logger.info(f"Trying encoding: {enc}")
                    return pd.read_csv(file_path, encoding=enc)
                except:
                    continue
            raise ValueError(f"Could not read CSV file with any common encoding: {str(e)}")
    
    elif file_format == 'excel':
        return pd.read_excel(file_path)
    
    elif file_format == 'json':
        return pd.read_json(file_path)
    
    elif file_format == 'parquet':
        return pd.read_parquet(file_path)
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

def get_project_root():
    """Determine project root based on environment (Docker vs local)"""
    # Check if running in Docker
    docker_root = Path("/opt/airflow/data-pipeline")
    if docker_root.exists():
        return docker_root
    
    # Fallback to local path (two levels up from scripts/)
    return Path(__file__).parent.parent

def find_data_files(config):
    """Search for data files in multiple locations (Docker-aware)"""
    logger = setup_logging("data_acquisition")
    supported_formats = config['data']['supported_formats']
    project_root = get_project_root()
    
    # Define search paths (Docker-compatible)
    search_locations = [
        ('Docker data/raw/', project_root / 'data' / 'raw'),
        ('Local data/raw/', Path('data/raw')),
        ('Project root', project_root),
    ]
    
    for location_name, location_path in search_locations:
        if not location_path.exists():
            logger.debug(f"Path does not exist: {location_path}")
            continue
            
        logger.info(f"Searching in {location_name} ({location_path})...")
        
        for fmt in supported_formats:
            potential_files = list(location_path.glob(f'*.{fmt}'))
            if potential_files:
                found_file = potential_files[0]
                logger.info(f"✓ Found: {found_file.name} in {location_name}")
                return str(found_file)
    
    return None

def acquire_data(source_file=None):
    """
    Acquire data from any source and prepare for processing
    Supports multiple file formats and automatic encoding detection
    
    Args:
        source_file (str, optional): Path to source file. If None, auto-detects.
    
    Returns:
        dict: Acquisition metadata
    """
    logger = setup_logging("data_acquisition")
    config = load_config()
    project_root = get_project_root()
    
    try:
        logger.info("Starting data acquisition...")
        logger.info("=" * 60)
        logger.info(f"Project root: {project_root}")
        
        # Determine raw data path (Docker-aware)
        raw_path = project_root / 'data' / 'raw'
        processed_path = project_root / 'data' / 'processed'
        
        # Ensure directories exist
        ensure_dir(raw_path)
        ensure_dir(processed_path)
        
        # If no source file specified, search for it
        if source_file is None:
            logger.info("No source file specified, searching for data files...")
            source_file = find_data_files(config)
        
        if source_file is None:
            # Provide helpful error message with both Docker and local paths
            error_msg = (
                "No data file found!\n\n"
                "📋 Please either:\n"
                "  1. Place your data file (CSV, Excel, JSON, Parquet) in:\n"
                f"     - Docker: /opt/airflow/data-pipeline/data/raw/\n"
                f"     - Local: data/raw/ directory\n\n"
                "  2. Specify the file path in DAG run config:\n"
                "     {'source_file': 'path/to/your/file.csv'}\n\n"
                f"  Supported formats: {', '.join(config['data']['supported_formats'])}\n\n"
                f"  Current search path: {raw_path}"
            )
            raise FileNotFoundError(error_msg)
        
        # Check if source file exists
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Get file info
        file_size = source_path.stat().st_size / 1024 / 1024  # MB
        logger.info(f"Source file: {source_path.name}")
        logger.info(f"File size: {file_size:.2f} MB")
        
        # Detect format and dataset name
        file_format = detect_file_format(source_file)
        dataset_name = source_path.stem
        
        logger.info(f"Detected format: {file_format}")
        logger.info(f"Dataset name: {dataset_name}")
        logger.info("=" * 60)
        
        # Read data with automatic encoding detection
        logger.info("Reading data file...")
        df = read_data_file(source_file)
        logger.info(f"✓ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Save to raw directory as CSV (standardized format)
        destination = raw_path / f"{dataset_name}.csv"
        
        # Only copy if source is not already in the exact destination
        if source_path.resolve() != destination.resolve():
            logger.info(f"Saving to: {destination}")
            df.to_csv(destination, index=False, encoding='utf-8')
            logger.info(f"✓ Data saved to {destination}")
        else:
            logger.info(f"✓ Data already in destination: {destination}")
        
        # Generate schema profile
        logger.info("Generating schema profile...")
        detector = SchemaDetector()
        schema_profile = detector.generate_schema_profile(df, dataset_name)
        
        # Save schema profile to config directory
        config_dir = project_root / 'config' / 'dataset_profiles'
        ensure_dir(config_dir)
        
        logger.info("=" * 60)
        logger.info("DATA ACQUISITION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Rows: {df.shape[0]:,}")
        logger.info(f"Columns: {df.shape[1]}")
        logger.info(f"Column types detected:")
        for col_type, cols in schema_profile['column_types'].items():
            if cols:
                logger.info(f"  - {col_type}: {len(cols)} columns")
        logger.info(f"Protected attributes: {len(schema_profile['protected_attributes'])}")
        logger.info(f"Schema profile: {config_dir}/{dataset_name}_profile.json")
        logger.info("=" * 60)
        logger.info("✓ Data acquisition completed successfully!")
        
        return {
            "dataset_name": dataset_name,
            "file_path": str(destination),
            "rows": len(df),
            "columns": len(df.columns),
            "schema_profile": schema_profile
        }
        
    except Exception as e:
        logger.error(f"Error in data acquisition: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    source_file = sys.argv[1] if len(sys.argv) > 1 else None
    acquire_data(source_file)