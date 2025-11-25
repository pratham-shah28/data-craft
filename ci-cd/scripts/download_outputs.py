#!/usr/bin/env python3
"""
Download Latest Model Outputs from GCS
Downloads the most recent best model responses from GCS
"""

import sys
import os
import yaml
from pathlib import Path
from google.cloud import storage
from typing import Optional

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "ci_cd_config.yaml"
outputs_dir = project_root / "outputs"

def load_config():
    """Load CI/CD configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_latest_model_path(bucket, base_path: str) -> Optional[str]:
    """Find the most recent model path in GCS"""
    blobs = list(bucket.list_blobs(prefix=f"{base_path}/"))
    
    if not blobs:
        return None
    
    # Group by model run (timestamp_model pattern)
    model_paths = set()
    for blob in blobs:
        # Extract path like "best_model_responses/20250115_120000_gemini-2.5-flash/..."
        parts = blob.name.split('/')
        if len(parts) >= 2:
            model_paths.add(f"{parts[0]}/{parts[1]}")
    
    if not model_paths:
        return None
    
    # Sort by path name (timestamp format sorts chronologically)
    latest_path = sorted(model_paths, reverse=True)[0]
    return latest_path

def download_outputs(bucket, model_path: str, local_dir: Path):
    """Download all outputs from model path"""
    print(f"Downloading from: {model_path}")
    
    downloaded_files = []
    
    # Download all files in the model path
    blobs = list(bucket.list_blobs(prefix=f"{model_path}/"))
    
    for blob in blobs:
        # Skip if it's a directory marker
        if blob.name.endswith('/'):
            continue
        
        # Create local file path, preserving structure
        relative_path = blob.name.replace(f"{model_path}/", "")
        # Map to best-model-responses directory structure
        local_file = local_dir / "best-model-responses" / relative_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        try:
            blob.download_to_filename(str(local_file))
            downloaded_files.append(relative_path)
            print(f"  ✓ {relative_path}")
        except Exception as e:
            print(f"  ✗ Failed to download {relative_path}: {e}")
    
    return downloaded_files

def main():
    """Main function"""
    print("=" * 70)
    print("DOWNLOAD MODEL OUTPUTS FROM GCS")
    print("=" * 70)
    
    try:
        config = load_config()
        gcp_config = config['gcp']
        project_id = gcp_config['project_id']
        bucket_name = gcp_config['bucket_name']
        
        # DAG saves to best_model_responses/ path
        base_path = "best_model_responses"
        
        print(f"Project: {project_id}")
        print(f"Bucket: {bucket_name}")
        print(f"Base Path: {base_path}")
        
        # Initialize GCS client
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        
        # Find latest model path
        print("\nFinding latest model outputs...")
        latest_path = find_latest_model_path(bucket, base_path)
        
        if not latest_path:
            print("✗ No model outputs found in GCS")
            return 1
        
        print(f"✓ Found latest model: {latest_path}")
        
        # Download outputs
        print("\nDownloading outputs...")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = download_outputs(bucket, latest_path, outputs_dir)
        
        print(f"\n✓ Downloaded {len(downloaded)} files")
        print(f"  Local directory: {outputs_dir}/best-model-responses")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ DOWNLOAD ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
