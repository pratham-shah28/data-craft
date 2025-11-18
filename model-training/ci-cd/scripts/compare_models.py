#!/usr/bin/env python3
"""
Model Comparison Script
Compares new model with previous model
Implements rollback logic if new model performs worse

Usage:
    python compare_models.py [current_metrics_file]
"""

import json
import sys
from pathlib import Path
from google.cloud import storage
from datetime import datetime
import yaml

def load_config():
    """Load CI/CD configuration"""
    config_path = Path(__file__).parent.parent / "config" / "ci_cd_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_previous_model_metrics(bucket_name: str, project_id: str) -> dict:
    """
    Retrieve previous model metrics from GCS
    
    Args:
        bucket_name: GCS bucket name
        project_id: GCP project ID
        
    Returns:
        Previous model metrics dictionary or None if not found
    """
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        # List all model metadata files
        prefix = "best_model_responses/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        # Find all metadata files
        metadata_files = [
            b for b in blobs 
            if b.name.endswith("best_model_metadata.json") or 
               b.name.endswith("model_selection_report.json")
        ]
        
        if not metadata_files:
            print("No previous model metadata found in GCS")
            return None
        
        # Get most recent (by time created)
        latest = sorted(metadata_files, key=lambda x: x.time_created, reverse=True)[0]
        
        # Download and parse
        content = latest.download_as_text()
        metadata = json.loads(content)
        
        print(f"Found previous model metadata: {latest.name}")
        print(f"  Created: {latest.time_created}")
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not retrieve previous model metrics: {e}")
        return None

def extract_score_from_metadata(metadata: dict) -> float:
    """Extract composite score from metadata (handles different formats)"""
    # Try different possible locations
    if 'composite_score' in metadata:
        return metadata['composite_score']
    elif 'best_model' in metadata and 'composite_score' in metadata['best_model']:
        return metadata['best_model']['composite_score']
    elif 'selection_report' in metadata:
        return metadata['selection_report'].get('best_model', {}).get('composite_score', 0)
    else:
        return 0.0

def compare_models(current_metrics: dict, previous_metrics: dict, config: dict) -> tuple[bool, dict]:
    """
    Compare models, return True if new model is better
    
    Args:
        current_metrics: Current model metrics
        previous_metrics: Previous model metrics (or None)
        config: CI/CD configuration
        
    Returns:
        Tuple of (is_better: bool, comparison_details: dict)
    """
    if not previous_metrics:
        print("No previous model found - accepting new model")
        return True, {
            'previous_score': None,
            'current_score': current_metrics.get('best_model', {}).get('composite_score', 0),
            'improvement': None,
            'decision': 'ACCEPT (no previous model)'
        }
    
    # Extract scores
    current_score = current_metrics.get('best_model', {}).get('composite_score', 0)
    previous_score = extract_score_from_previous_metadata(previous_metrics)
    
    improvement = current_score - previous_score
    improvement_pct = (improvement / previous_score * 100) if previous_score > 0 else 0
    
    # Get rollback configuration
    rollback_config = config['rollback']
    degradation_threshold = rollback_config['performance_degradation_threshold']
    min_improvement = rollback_config.get('min_improvement_required', 0.0)
    
    # Calculate threshold (allow small regression)
    threshold = previous_score * (1 - degradation_threshold)
    
    # Decision logic
    is_better = current_score >= threshold
    
    # Additional check for minimum improvement if required
    if min_improvement > 0 and improvement < min_improvement:
        is_better = False
    
    comparison_details = {
        'previous_score': previous_score,
        'current_score': current_score,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'threshold': threshold,
        'is_better': is_better,
        'decision': 'ACCEPT' if is_better else 'REJECT (rollback recommended)'
    }
    
    return is_better, comparison_details

def extract_score_from_previous_metadata(metadata: dict) -> float:
    """Extract composite score from previous metadata (handles various formats)"""
    # Try different possible locations
    if 'composite_score' in metadata:
        return metadata['composite_score']
    elif 'best_model' in metadata:
        if isinstance(metadata['best_model'], dict):
            return metadata['best_model'].get('composite_score', 0)
    elif 'selection_report' in metadata:
        return metadata['selection_report'].get('best_model', {}).get('composite_score', 0)
    
    return 0.0

def print_comparison_results(comparison_details: dict):
    """Print comparison results in a readable format"""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    if comparison_details['previous_score'] is None:
        print("No previous model found - accepting new model")
        print(f"Current Model Score: {comparison_details['current_score']:.2f}/100")
    else:
        print(f"Previous Model Score: {comparison_details['previous_score']:.2f}/100")
        print(f"Current Model Score:  {comparison_details['current_score']:.2f}/100")
        print(f"Improvement:          {comparison_details['improvement']:+.2f} ({comparison_details['improvement_pct']:+.2f}%)")
        print(f"Threshold:            {comparison_details['threshold']:.2f}/100")
        print(f"Decision:             {comparison_details['decision']}")
    
    print("=" * 70)
    
    if comparison_details['is_better']:
        print("✓ NEW MODEL ACCEPTED")
    else:
        print("✗ NEW MODEL REJECTED - Performance degradation detected")
        print("  Recommendation: Rollback to previous model version")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    gcp_config = config['gcp']
    
    # Get current metrics file path
    if len(sys.argv) > 1:
        current_metrics_file = sys.argv[1]
    else:
        current_metrics_file = config['paths']['model_training_output'] + "/pipeline_summary.json"
    
    if not Path(current_metrics_file).exists():
        print(f"ERROR: Current metrics file not found: {current_metrics_file}")
        sys.exit(1)
    
    # Load current metrics
    with open(current_metrics_file, 'r') as f:
        current_metrics = json.load(f)
    
    # Get previous model metrics from GCS
    previous_metrics = get_previous_model_metrics(
        bucket_name=gcp_config['bucket_name'],
        project_id=gcp_config['project_id']
    )
    
    # Compare models
    is_better, comparison_details = compare_models(
        current_metrics=current_metrics,
        previous_metrics=previous_metrics,
        config=config
    )
    
    # Print results
    print_comparison_results(comparison_details)
    
    # Exit with appropriate code
    sys.exit(0 if is_better else 1)

