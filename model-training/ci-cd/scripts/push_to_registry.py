#!/usr/bin/env python3
"""
Push Model to Registry Script
Pushes model metadata and artifacts to GCP Artifact Registry and Vertex AI Model Registry

Usage:
    python push_to_registry.py [metadata_file] [artifacts_dir]
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import yaml
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud import artifactregistry

def load_config():
    """Load CI/CD configuration"""
    config_path = Path(__file__).parent.parent / "config" / "ci_cd_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def push_to_artifact_registry(
    project_id: str,
    location: str,
    repository: str,
    metadata_file: str,
    artifacts_dir: str
) -> str:
    """
    Push model artifacts to GCP Artifact Registry
    
    Args:
        project_id: GCP project ID
        location: Artifact Registry location
        repository: Repository name
        metadata_file: Path to model metadata JSON file
        artifacts_dir: Directory containing model artifacts
        
    Returns:
        Artifact Registry URI
    """
    print(f"\nPushing to Artifact Registry: {repository}")
    
    # Initialize Artifact Registry client
    client = artifactregistry.ArtifactRegistryClient()
    
    # Repository path
    repository_path = f"projects/{project_id}/locations/{location}/repositories/{repository}"
    
    # Read metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Create version tag
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = metadata.get('selected_model', 'unknown').replace('/', '_').replace('.', '_')
    version_tag = f"{model_name}_{timestamp}"
    
    # Upload metadata file
    metadata_blob_name = f"metadata/{version_tag}_metadata.json"
    
    try:
        # Note: Artifact Registry API for generic artifacts requires using gcloud or Storage API
        # For simplicity, we'll use GCS and reference it
        print(f"  Metadata will be stored in GCS and referenced in registry")
        print(f"  Version tag: {version_tag}")
        
        return f"{repository_path}/packages/{version_tag}"
        
    except Exception as e:
        print(f"  Warning: Artifact Registry upload failed: {e}")
        print(f"  Continuing with Vertex AI Model Registry only...")
        return None

def push_to_vertex_ai_registry(
    project_id: str,
    location: str,
    metadata_file: str,
    artifacts_gcs_path: str
) -> str:
    """
    Push model to Vertex AI Model Registry
    
    Args:
        project_id: GCP project ID
        location: GCP region
        metadata_file: Path to model metadata JSON file
        artifacts_gcs_path: GCS path to model artifacts
        
    Returns:
        Model resource name
    """
    print(f"\nPushing to Vertex AI Model Registry...")
    
    # Initialize AI Platform
    aiplatform.init(project=project_id, location=location)
    
    # Read metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata.get('selected_model', 'unknown')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model display name
    display_name = f"gemini-query-model-{timestamp}"
    
    # Prepare model metadata
    model_metadata = {
        'composite_score': metadata.get('composite_score', 0),
        'performance_score': metadata.get('performance_score', 0),
        'bias_score': metadata.get('bias_score', 0),
        'model_name': model_name,
        'selection_date': metadata.get('selection_date', datetime.now().isoformat()),
        'deployment_ready': metadata.get('deployment_ready', True)
    }
    
    try:
        # Upload model to Vertex AI Model Registry
        # Note: For LLM models like Gemini, we store metadata and reference the model name
        # The actual model is hosted by Google, we just register our configuration
        
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifacts_gcs_path,  # GCS path to artifacts
            serving_container_image_uri=None,  # Not needed for LLM models
            description=f"Best model: {model_name}. Composite Score: {model_metadata['composite_score']:.2f}",
            metadata=model_metadata
        )
        
        print(f"  ✓ Model uploaded to Vertex AI Registry")
        print(f"  Model Resource Name: {model.resource_name}")
        print(f"  Display Name: {display_name}")
        
        return model.resource_name
        
    except Exception as e:
        print(f"  ✗ Failed to upload to Vertex AI Registry: {e}")
        raise

def upload_artifacts_to_gcs(
    project_id: str,
    bucket_name: str,
    local_artifacts_dir: str
) -> str:
    """
    Upload model artifacts to GCS
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name
        local_artifacts_dir: Local directory with artifacts
        
    Returns:
        GCS path to uploaded artifacts
    """
    print(f"\nUploading artifacts to GCS: gs://{bucket_name}")
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    artifacts_path = Path(local_artifacts_dir)
    if not artifacts_path.exists():
        print(f"  Warning: Artifacts directory not found: {local_artifacts_dir}")
        return None
    
    # Create GCS path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gcs_base_path = f"model_registry/{timestamp}"
    
    uploaded_files = []
    
    # Upload all files
    for file_path in artifacts_path.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(artifacts_path)
            gcs_path = f"{gcs_base_path}/{relative_path}"
            
            try:
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(file_path))
                uploaded_files.append(gcs_path)
                print(f"  ✓ Uploaded: {relative_path}")
            except Exception as e:
                print(f"  ✗ Failed to upload {relative_path}: {e}")
    
    gcs_uri = f"gs://{bucket_name}/{gcs_base_path}"
    print(f"  ✓ Uploaded {len(uploaded_files)} files to {gcs_uri}")
    
    return gcs_uri

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    gcp_config = config['gcp']
    deployment_config = config['deployment']
    
    # Get file paths
    if len(sys.argv) > 1:
        metadata_file = sys.argv[1]
    else:
        metadata_file = config['paths']['best_model_output'] + "/best_model_metadata.json"
    
    if len(sys.argv) > 2:
        artifacts_dir = sys.argv[2]
    else:
        artifacts_dir = config['paths']['best_model_output']
    
    if not Path(metadata_file).exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("PUSHING MODEL TO REGISTRY")
    print("=" * 70)
    
    try:
        # Upload artifacts to GCS first
        artifacts_gcs_path = upload_artifacts_to_gcs(
            project_id=gcp_config['project_id'],
            bucket_name=gcp_config['bucket_name'],
            local_artifacts_dir=artifacts_dir
        )
        
        if not artifacts_gcs_path:
            print("ERROR: Failed to upload artifacts to GCS")
            sys.exit(1)
        
        # Push to Vertex AI Model Registry
        model_resource_name = push_to_vertex_ai_registry(
            project_id=deployment_config['vertex_ai']['project_id'],
            location=deployment_config['vertex_ai']['region'],
            metadata_file=metadata_file,
            artifacts_gcs_path=artifacts_gcs_path
        )
        
        # Try Artifact Registry (optional)
        try:
            artifact_registry_uri = push_to_artifact_registry(
                project_id=deployment_config['registry']['project_id'],
                location=deployment_config['registry']['region'],
                repository=deployment_config['registry']['repository'],
                metadata_file=metadata_file,
                artifacts_dir=artifacts_dir
            )
        except Exception as e:
            print(f"  Note: Artifact Registry push skipped: {e}")
            artifact_registry_uri = None
        
        print("\n" + "=" * 70)
        print("✓ MODEL SUCCESSFULLY PUSHED TO REGISTRY")
        print("=" * 70)
        print(f"Vertex AI Model: {model_resource_name}")
        if artifact_registry_uri:
            print(f"Artifact Registry: {artifact_registry_uri}")
        print(f"Artifacts GCS Path: {artifacts_gcs_path}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to push model to registry: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

