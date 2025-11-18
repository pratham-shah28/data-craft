#!/usr/bin/env python3
"""
Model Deployment Script
Deploys model to Vertex AI endpoint or other production environment

Usage:
    python deploy_model.py [model_resource_name]
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import yaml
from google.cloud import aiplatform

def load_config():
    """Load CI/CD configuration"""
    config_path = Path(__file__).parent.parent / "config" / "ci_cd_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def deploy_to_vertex_ai_endpoint(
    project_id: str,
    location: str,
    model_resource_name: str,
    endpoint_name: str
) -> str:
    """
    Deploy model to Vertex AI endpoint
    
    Args:
        project_id: GCP project ID
        location: GCP region
        model_resource_name: Vertex AI model resource name
        endpoint_name: Endpoint display name
        
    Returns:
        Endpoint resource name
    """
    print(f"\nDeploying model to Vertex AI endpoint: {endpoint_name}")
    
    # Initialize AI Platform
    aiplatform.init(project=project_id, location=location)
    
    # Get model
    try:
        model = aiplatform.Model(model_resource_name)
        print(f"  Model: {model.display_name}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        raise
    
    # Create or get endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    
    if endpoints:
        endpoint = endpoints[0]
        print(f"  Using existing endpoint: {endpoint.display_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        print(f"  Created new endpoint: {endpoint.display_name}")
    
    # Deploy model
    deployed_model_display_name = f"{endpoint_name}-{datetime.now().strftime('%Y%m%d')}"
    
    try:
        endpoint.deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            traffic_percentage=100  # Route 100% traffic to new model
        )
        
        print(f"  ✓ Model deployed successfully")
        print(f"  Endpoint Resource Name: {endpoint.resource_name}")
        print(f"  Deployed Model Name: {deployed_model_display_name}")
        
        return endpoint.resource_name
        
    except Exception as e:
        print(f"  ✗ Deployment failed: {e}")
        raise

def get_model_resource_name_from_metadata(metadata_file: str) -> str:
    """
    Extract model resource name from metadata file
    For now, we'll need to get it from the push_to_registry output
    """
    # In a real scenario, you'd store the model resource name in metadata
    # For now, we'll list the most recent model
    print("  Note: Model resource name should be passed as argument or stored in metadata")
    return None

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    deployment_config = config['deployment']['vertex_ai']
    
    # Get model resource name
    if len(sys.argv) > 1:
        model_resource_name = sys.argv[1]
    else:
        # Try to get from metadata or list recent models
        print("ERROR: Model resource name required as argument")
        print("Usage: python deploy_model.py <model_resource_name>")
        print("\nTo get model resource name, check the output of push_to_registry.py")
        sys.exit(1)
    
    # Check if auto-deploy is enabled
    if not deployment_config.get('auto_deploy', False):
        print("WARNING: Auto-deploy is disabled in configuration")
        print("Set 'deployment.vertex_ai.auto_deploy: true' in ci_cd_config.yaml to enable")
        response = input("Do you want to proceed with deployment? (yes/no): ")
        if response.lower() != 'yes':
            print("Deployment cancelled")
            sys.exit(0)
    
    print("\n" + "=" * 70)
    print("DEPLOYING MODEL TO PRODUCTION")
    print("=" * 70)
    
    try:
        endpoint_resource_name = deploy_to_vertex_ai_endpoint(
            project_id=deployment_config['project_id'],
            location=deployment_config['region'],
            model_resource_name=model_resource_name,
            endpoint_name=deployment_config['endpoint_name']
        )
        
        print("\n" + "=" * 70)
        print("✓ MODEL SUCCESSFULLY DEPLOYED")
        print("=" * 70)
        print(f"Endpoint: {endpoint_resource_name}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

