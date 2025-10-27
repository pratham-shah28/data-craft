import os
import json
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
import pandas as pd

from utils import (
    setup_logging,
    load_config,
    validate_gcp_credentials,
    get_file_info,
    print_success,
    print_error,
    print_info,
    format_size,
    ensure_dir
)

logger = setup_logging("gcp_upload")

def initialize_gcs_client():
    """Initialize Google Cloud Storage client"""
    try:
        validate_gcp_credentials()
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        
        config = load_config()
        project_id = config['gcp']['project_id']
        
        client = storage.Client(credentials=credentials, project=project_id)
        logger.info("GCS client initialized successfully")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {str(e)}")
        raise

def upload_file_to_gcs(client, bucket_name, source_file, destination_blob_name):
    """Upload a file to Google Cloud Storage"""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Get file info before upload
        file_info = get_file_info(source_file)
        
        if not file_info:
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        logger.info(f"Uploading {source_file} to gs://{bucket_name}/{destination_blob_name}")
        
        # Upload with metadata
        blob.metadata = {
            "uploaded_at": datetime.now().isoformat(),
            "original_filename": Path(source_file).name,
            "file_size": str(file_info['size'])
        }
        
        blob.upload_from_filename(source_file)
        
        logger.info(f"Successfully uploaded {file_info['size_formatted']}")
        
        return {
            "bucket": bucket_name,
            "blob_name": destination_blob_name,
            "size": file_info['size'],
            "public_url": blob.public_url,
            "uploaded_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to upload {source_file}: {str(e)}")
        raise

def upload_dataset_metadata(client, bucket_name, dataset_name, metadata):
    """Upload dataset metadata to GCS"""
    try:
        metadata_blob_name = f"metadata/{dataset_name}_metadata.json"
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(metadata_blob_name)
        
        blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type='application/json'
        )
        
        logger.info(f"Metadata uploaded to gs://{bucket_name}/{metadata_blob_name}")
        
    except Exception as e:
        logger.error(f"Failed to upload metadata: {str(e)}")
        raise

def generate_dataset_metadata(dataset_name, config):
    """Generate comprehensive metadata for the dataset"""
    metadata = {
        "dataset_name": dataset_name,
        "upload_timestamp": datetime.now().isoformat(),
        "stages": {}
    }
    
    # Define stage paths
    stage_paths = {
        'raw': Path(config['data']['raw_path']) / f"{dataset_name}.csv",
        'processed': Path(config['data']['processed_path']) / f"{dataset_name}_validated.csv",
        'validated': Path(config['data']['validated_path']) / f"{dataset_name}_cleaned.csv"
    }
    
    # Collect metadata for each stage
    for stage, file_path in stage_paths.items():
        try:
            if file_path.exists():
                file_info = get_file_info(file_path)
                
                # Load dataframe for additional stats
                df = pd.read_csv(file_path)
                
                metadata['stages'][stage] = {
                    "file_info": file_info,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
                }
                
        except Exception as e:
            logger.warning(f"Could not collect metadata for {stage}: {str(e)}")
    
    # Load schema profile if available
    schema_path = Path("config/dataset_profiles") / f"{dataset_name}_profile.json"
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            metadata['schema_profile'] = json.load(f)
    
    # Load validation report if available
    validation_path = Path(config['data']['processed_path']) / f"{dataset_name}_validation_report.json"
    if validation_path.exists():
        with open(validation_path, 'r') as f:
            metadata['validation_report'] = json.load(f)
    
    # Load bias report if available
    bias_path = Path(config['data']['processed_path']) / f"{dataset_name}_bias_report.json"
    if bias_path.exists():
        with open(bias_path, 'r') as f:
            metadata['bias_report'] = json.load(f)
    
    # Load cleaning metrics if available
    cleaning_path = Path(config['data']['validated_path']) / f"{dataset_name}_cleaning_metrics.json"
    if cleaning_path.exists():
        with open(cleaning_path, 'r') as f:
            metadata['cleaning_metrics'] = json.load(f)
    
    return metadata

def upload_to_gcs(dataset_name, include_raw=False, include_reports=True):
    """
    Upload dataset files to Google Cloud Storage
    
    Args:
        dataset_name (str): Name of the dataset
        include_raw (bool): Whether to upload raw data
        include_reports (bool): Whether to upload validation/bias reports
    
    Returns:
        dict: Upload summary
    """
    config = load_config()
    bucket_name = config['gcp']['bucket_name']
    
    print_info(f"Starting GCS upload for dataset: {dataset_name}")
    
    try:
        # Initialize GCS client
        client = initialize_gcs_client()
        
        upload_results = []
        
        # 1. Upload data files
        stage_paths = {
            'raw': Path(config['data']['raw_path']) / f"{dataset_name}.csv",
            'processed': Path(config['data']['processed_path']) / f"{dataset_name}_validated.csv",
            'validated': Path(config['data']['validated_path']) / f"{dataset_name}_cleaned.csv"
        }
        
        # Upload stages
        stages_to_upload = ['processed', 'validated']
        if include_raw:
            stages_to_upload.insert(0, 'raw')
        
        for stage in stages_to_upload:
            source_file = stage_paths[stage]
            
            if source_file.exists():
                destination = f"data/{stage}/{dataset_name}_{stage}.csv"
                
                try:
                    result = upload_file_to_gcs(
                        client,
                        bucket_name,
                        str(source_file),
                        destination
                    )
                    
                    upload_results.append(result)
                    print_success(f"Uploaded {stage} data")
                except Exception as e:
                    logger.error(f"Failed to upload {stage} data: {str(e)}")
            else:
                logger.warning(f"File not found: {source_file}")
        
        # 2. Upload reports if requested
        if include_reports:
            report_files = [
                (Path(config['data']['processed_path']) / f"{dataset_name}_validation_report.json", "validation"),
                (Path(config['data']['processed_path']) / f"{dataset_name}_bias_report.json", "bias"),
                (Path(config['data']['validated_path']) / f"{dataset_name}_cleaning_metrics.json", "cleaning"),
                (Path("config/dataset_profiles") / f"{dataset_name}_profile.json", "schema")
            ]
            
            for report_file, report_type in report_files:
                if report_file.exists():
                    destination = f"reports/{report_type}/{dataset_name}_{report_type}_report.json"
                    
                    try:
                        result = upload_file_to_gcs(
                            client,
                            bucket_name,
                            str(report_file),
                            destination
                        )
                        
                        upload_results.append(result)
                        print_success(f"Uploaded {report_type} report")
                    except Exception as e:
                        logger.error(f"Failed to upload {report_type} report: {str(e)}")
        
        # 3. Generate and upload comprehensive metadata
        metadata = generate_dataset_metadata(dataset_name, config)
        metadata['gcs_files'] = upload_results
        
        upload_dataset_metadata(client, bucket_name, dataset_name, metadata)
        print_success("Uploaded dataset metadata")
        
        # Summary
        total_size = sum(r['size'] for r in upload_results)
        
        summary = {
            "dataset_name": dataset_name,
            "files_uploaded": len(upload_results),
            "total_size": total_size,
            "total_size_formatted": format_size(total_size),
            "bucket": bucket_name,
            "upload_timestamp": datetime.now().isoformat(),
            "files": upload_results
        }
        
        print_success(f"✓ Upload complete! {len(upload_results)} files ({format_size(total_size)})")
        
        return summary
        
    except Exception as e:
        print_error(f"GCS upload failed: {str(e)}")
        logger.error(f"GCS upload failed: {str(e)}", exc_info=True)
        raise

def download_from_gcs(dataset_name, stage='processed', destination_path=None):
    """Download dataset from GCS"""
    config = load_config()
    bucket_name = config['gcp']['bucket_name']
    
    try:
        client = initialize_gcs_client()
        bucket = client.bucket(bucket_name)
        
        # Source blob in GCS
        source_blob_name = f"data/{stage}/{dataset_name}_{stage}.csv"
        blob = bucket.blob(source_blob_name)
        
        # Destination path
        if destination_path is None:
            if stage == 'raw':
                destination_path = Path(config['data']['raw_path']) / f"{dataset_name}.csv"
            elif stage == 'processed':
                destination_path = Path(config['data']['processed_path']) / f"{dataset_name}_validated.csv"
            else:  # validated
                destination_path = Path(config['data']['validated_path']) / f"{dataset_name}_cleaned.csv"
        
        # Ensure directory exists
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading from gs://{bucket_name}/{source_blob_name}")
        
        blob.download_to_filename(str(destination_path))
        
        file_info = get_file_info(destination_path)
        print_success(f"✓ Downloaded {file_info['size_formatted']} to {destination_path}")
        
        return str(destination_path)
        
    except Exception as e:
        print_error(f"Download failed: {str(e)}")
        logger.error(f"Download failed: {str(e)}", exc_info=True)
        raise

def list_gcs_datasets(bucket_name=None):
    """List all datasets in GCS bucket"""
    if bucket_name is None:
        config = load_config()
        bucket_name = config['gcp']['bucket_name']
    
    try:
        client = initialize_gcs_client()
        bucket = client.bucket(bucket_name)
        
        # List blobs in data/processed
        blobs = bucket.list_blobs(prefix='data/processed/')
        
        datasets = set()
        for blob in blobs:
            # Extract dataset name from path
            filename = Path(blob.name).stem
            dataset_name = filename.replace('_processed', '')
            datasets.add(dataset_name)
        
        return sorted(list(datasets))
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload/Download data to/from GCS")
    parser.add_argument("action", choices=['upload', 'download', 'list'],
                       help="Action to perform")
    parser.add_argument("--dataset", type=str,
                       help="Dataset name")
    parser.add_argument("--stage", type=str, default='processed',
                       choices=['raw', 'processed', 'validated'],
                       help="Dataset stage")
    parser.add_argument("--include-raw", action='store_true',
                       help="Include raw data in upload")
    parser.add_argument("--include-reports", action='store_true', default=True,
                       help="Include reports in upload")
    
    args = parser.parse_args()
    
    if args.action == 'upload':
        if not args.dataset:
            print_error("Dataset name required for upload")
            exit(1)
        
        upload_to_gcs(
            dataset_name=args.dataset,
            include_raw=args.include_raw,
            include_reports=args.include_reports
        )
    
    elif args.action == 'download':
        if not args.dataset:
            print_error("Dataset name required for download")
            exit(1)
        
        download_from_gcs(
            dataset_name=args.dataset,
            stage=args.stage
        )
    
    elif args.action == 'list':
        datasets = list_gcs_datasets()
        print_info(f"Found {len(datasets)} datasets:")
        for dataset in datasets:
            print(f"  • {dataset}")