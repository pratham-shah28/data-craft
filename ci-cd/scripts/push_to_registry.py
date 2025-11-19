#!/usr/bin/env python3
"""
Push Model to GCP Model Registry
Pushes model metadata and artifacts to GCS and Vertex AI Model Registry

This implementation uses:
- GCS for storing model artifacts (files, reports, etc.)
- Vertex AI Model Registry for model versioning and metadata management

Usage:
    python push_to_registry.py --commit-sha <commit_sha>
"""

import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from google.cloud import storage
from google.cloud import aiplatform

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
    selection_dir = outputs_dir / "model-selection"
    reports = list(selection_dir.glob("model_selection_*.json"))
    
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
    Upload model artifacts to GCS
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name
        base_path: Base path within bucket
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
    
    # Upload all relevant artifacts
    artifacts_to_upload = [
        ("model-selection", "model_selection_*.json"),
        ("evaluation", "model_comparison_*.json"),
        ("bias", "bias_comparison_*.json"),
        ("best-model-responses", "**/*.json"),
        ("validation", "*.json"),
    ]
    
    for dir_name, pattern in artifacts_to_upload:
        source_dir = outputs_dir / dir_name
        if source_dir.exists():
            for file_path in source_dir.rglob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(outputs_dir)
                    gcs_path = f"{gcs_base_path}/{relative_path}"
                    
                    try:
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(str(file_path))
                        uploaded_files.append(gcs_path)
                        print(f"  ✓ Uploaded: {relative_path.name}")
                    except Exception as e:
                        print(f"  ✗ Failed to upload {relative_path.name}: {e}")
    
    gcs_uri = f"gs://{bucket_name}/{gcs_base_path}"
    print(f"  ✓ Uploaded {len(uploaded_files)} files to {gcs_uri}")
    
    return gcs_uri

def push_to_vertex_ai_registry(
    project_id: str,
    location: str,
    artifacts_gcs_uri: str,
    metadata: Dict,
    commit_sha: str
) -> Optional[str]:
    """
    Push model to Vertex AI Model Registry
    
    Vertex AI Model Registry is specifically designed for ML models and provides:
    - Model versioning
    - Metadata management
    - Integration with other Vertex AI services
    
    Args:
        project_id: GCP project ID
        location: GCP region
        artifacts_gcs_uri: GCS URI to model artifacts
        metadata: Model metadata dictionary
        commit_sha: Git commit SHA
        
    Returns:
        Model resource name if successful, None otherwise
    """
    print(f"\nPushing to Vertex AI Model Registry...")
    
    try:
        # Initialize AI Platform
        aiplatform.init(project=project_id, location=location)
        
        model_name = metadata.get('selected_model', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create model display name
        display_name = f"gemini-query-model-{timestamp}"
        
        # Prepare model metadata for Vertex AI
        model_metadata = {
            'composite_score': float(metadata.get('composite_score', 0)),
            'performance_score': float(metadata.get('performance_score', 0)),
            'bias_score': float(metadata.get('bias_score', 0)),
            'success_rate': float(metadata.get('success_rate', 0)),
            'response_time': float(metadata.get('response_time', 0)),
            'model_name': str(model_name),
            'commit_sha': str(commit_sha),
            'selection_date': str(metadata.get('selection_date', datetime.now().isoformat())),
            'deployment_ready': True
        }
        
        # Create description
        description = (
            f"Best model: {model_name}. "
            f"Composite Score: {model_metadata['composite_score']:.2f}, "
            f"Performance: {model_metadata['performance_score']:.2f}, "
            f"Bias Score: {model_metadata['bias_score']:.2f}"
        )
        
        # Upload model to Vertex AI Model Registry
        # For LLM models like Gemini, we store metadata and reference the model name
        # The actual model is hosted by Google, we register our configuration and artifacts
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifacts_gcs_uri,
            serving_container_image_uri=None,  # Not needed for LLM models
            description=description,
            metadata=model_metadata
        )
        
        print(f"  ✓ Model uploaded to Vertex AI Registry")
        print(f"  Model Resource Name: {model.resource_name}")
        print(f"  Display Name: {display_name}")
        print(f"  Model ID: {model.name}")
        
        return model.resource_name
        
    except Exception as e:
        print(f"  ⚠ Failed to upload to Vertex AI Registry: {e}")
        print(f"  Continuing with GCS storage only...")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main push function"""
    parser = argparse.ArgumentParser(description='Push model to registry')
    parser.add_argument('--commit-sha', required=True, help='Git commit SHA')
    args = parser.parse_args()
    
    print("=" * 70)
    print("PUSH MODEL TO REGISTRY")
    print("=" * 70)
    
    try:
        config = load_config()
        print("✓ Loaded configuration")
        
        gcp_config = config['gcp']
        project_id = gcp_config['project_id']
        location = gcp_config['region']
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
        
        # Push to Vertex AI Model Registry (if enabled)
        model_resource_name = None
        vertex_ai_enabled = config.get('gcp', {}).get('vertex_ai', {}).get('enabled', False)
        
        if vertex_ai_enabled:
            model_resource_name = push_to_vertex_ai_registry(
                project_id=project_id,
                location=location,
                artifacts_gcs_uri=artifacts_gcs_uri,
                metadata=metadata,
                commit_sha=args.commit_sha
            )
        else:
            print("\n✓ Using GCS-only model registry (Vertex AI disabled)")
            print("  This provides version control and reproducibility as required")
        
        # Save push report
        push_report = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "commit_sha": args.commit_sha,
            "artifacts_gcs_uri": artifacts_gcs_uri,
            "model_name": metadata.get('selected_model', 'unknown'),
            "model_score": metadata.get('composite_score', 0),
            "vertex_ai_model": model_resource_name if model_resource_name else None
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
        if model_resource_name:
            print(f"  Vertex AI Model: {model_resource_name}")
        else:
            print(f"  Vertex AI: Not registered (check logs above)")
        print(f"\nPush report saved: {report_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ PUSH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
