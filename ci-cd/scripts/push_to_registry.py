#!/usr/bin/env python3
"""
Push Model to Production Registry
Uploads validated model artifacts to GCS with commit SHA tagging
"""

import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict
from google.cloud import storage

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "ci_cd_config.yaml"
outputs_dir = project_root / "outputs"

def load_config():
    """Load CI/CD configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_best_model_metadata() -> Dict:
    """Get best model metadata from selection report"""
    best_model_dir = outputs_dir / "best-model-responses"
    reports = list(best_model_dir.rglob("model_selection_report.json"))
    
    if not reports:
        return {}
    
    latest = max(reports, key=lambda p: p.stat().st_mtime)
    with open(latest, 'r') as f:
        data = json.load(f)
    
    best_model = data.get('best_model', {})
    return {
        'selected_model': best_model.get('name', 'unknown'),
        'composite_score': best_model.get('composite_score', 0),
        'performance_score': best_model.get('performance_score', 0),
        'bias_score': best_model.get('bias_score', 0),
        'success_rate': best_model.get('success_rate', 0),
        'response_time': best_model.get('response_time', 0),
        'selection_date': data.get('selection_timestamp', datetime.now().isoformat())
    }

def upload_artifacts_to_gcs(
    project_id: str,
    bucket_name: str,
    base_path: str,
    commit_sha: str
) -> str:
    """
    Upload model artifacts to GCS production registry
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name
        base_path: Base path within bucket (models/)
        commit_sha: Git commit SHA
        
    Returns:
        GCS URI to uploaded artifacts
    """
    print(f"\nUploading artifacts to GCS: gs://{bucket_name}")
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Create GCS path with timestamp and commit SHA
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gcs_base_path = f"{base_path}/{timestamp}_{commit_sha[:7]}"
    
    uploaded_files = []
    
    # Upload best model responses and metadata
    best_model_dir = outputs_dir / "best-model-responses"
    if best_model_dir.exists():
        # Find the latest model run directory
        model_dirs = [d for d in best_model_dir.iterdir() if d.is_dir()]
        if model_dirs:
            latest_model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
            
            # Upload all files from this directory
            for file_path in latest_model_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(best_model_dir)
                    gcs_path = f"{gcs_base_path}/best-model-responses/{relative_path}"
                    
                    try:
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(str(file_path))
                        uploaded_files.append(gcs_path)
                        print(f"  ✓ Uploaded: {relative_path.name}")
                    except Exception as e:
                        print(f"  ✗ Failed to upload {relative_path.name}: {e}")
    
    # Upload validation reports
    validation_dir = outputs_dir / "validation"
    if validation_dir.exists():
        for file_path in validation_dir.glob("*.json"):
            relative_path = file_path.name
            gcs_path = f"{gcs_base_path}/validation/{relative_path}"
            
            try:
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(file_path))
                uploaded_files.append(gcs_path)
                print(f"  ✓ Uploaded: {relative_path}")
            except Exception as e:
                print(f"  ✗ Failed to upload {relative_path}: {e}")
    
    # Upload best_model_metadata.json at root for easy access
    best_model_dir = outputs_dir / "best-model-responses"
    metadata_files = list(best_model_dir.rglob("best_model_metadata.json"))
    if metadata_files:
        latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
        gcs_metadata_path = f"{gcs_base_path}/best_model_metadata.json"
        try:
            blob = bucket.blob(gcs_metadata_path)
            blob.upload_from_filename(str(latest_metadata))
            uploaded_files.append(gcs_metadata_path)
            print(f"  ✓ Uploaded: best_model_metadata.json")
        except Exception as e:
            print(f"  ✗ Failed to upload metadata: {e}")
    
    gcs_uri = f"gs://{bucket_name}/{gcs_base_path}"
    print(f"  ✓ Uploaded {len(uploaded_files)} files to {gcs_uri}")
    
    return gcs_uri

def main():
    """Main push function"""
    parser = argparse.ArgumentParser(description='Push model to registry')
    parser.add_argument('--commit-sha', required=True, help='Git commit SHA')
    args = parser.parse_args()
    
    print("=" * 70)
    print("PUSH MODEL TO PRODUCTION REGISTRY")
    print("=" * 70)
    
    try:
        config = load_config()
        print("✓ Loaded configuration")
        
        gcp_config = config['gcp']
        project_id = gcp_config['project_id']
        bucket_name = gcp_config['bucket_name']
        base_path = config['model_registry']['base_path']
        
        # Get model metadata
        metadata = get_best_model_metadata()
        if not metadata:
            print("⚠ No model selection report found, using defaults")
            metadata = {
                'selected_model': 'unknown',
                'composite_score': 0,
                'performance_score': 0,
                'bias_score': 0,
                'success_rate': 0,
                'response_time': 0
            }
        
        # Upload artifacts to GCS
        artifacts_gcs_uri = upload_artifacts_to_gcs(
            project_id=project_id,
            bucket_name=bucket_name,
            base_path=base_path,
            commit_sha=args.commit_sha
        )
        
        # Save push report
        push_report = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "commit_sha": args.commit_sha,
            "artifacts_gcs_uri": artifacts_gcs_uri,
            "model_name": metadata.get('selected_model', 'unknown'),
            "model_score": metadata.get('composite_score', 0)
        }
        
        report_file = outputs_dir / "validation" / "push_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(push_report, f, indent=2)
        
        print("\n" + "=" * 70)
        print("PUSH COMPLETE")
        print("=" * 70)
        print(f"  Model: {metadata.get('selected_model', 'unknown')}")
        print(f"  Score: {metadata.get('composite_score', 0):.2f}")
        print(f"  Artifacts: {artifacts_gcs_uri}")
        print(f"\nPush report saved: {report_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ PUSH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
