#!/usr/bin/env python3
"""
Rollback Manager
Compares new model with previous model and implements rollback if needed
"""

import sys
import json
import yaml
from pathlib import Path
from google.cloud import storage
from typing import Optional, Dict

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "ci_cd_config.yaml"
outputs_dir = project_root / "outputs"

def load_config():
    """Load CI/CD configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_current_model_metrics() -> Optional[Dict]:
    """Get metrics from current model selection report"""
    selection_dir = outputs_dir / "model-selection"
    if not selection_dir.exists():
        return None
    
    reports = list(selection_dir.glob("model_selection_*.json"))
    if not reports:
        return None
    
    latest_report = max(reports, key=lambda p: p.stat().st_mtime)
    with open(latest_report, 'r') as f:
        data = json.load(f)
    
    return data.get('best_model', {})

def get_previous_model_metrics(config: Dict) -> Optional[Dict]:
    """Get metrics from previous model in GCS
    
    Looks for metadata files in this order:
    1. best_model_metadata.json (from response_saver)
    2. model_selection_*.json (from model_selector)
    
    Returns the second most recent model's metrics.
    """
    try:
        bucket_name = config['gcp']['bucket_name']
        base_path = config['model_registry']['base_path']
        
        storage_client = storage.Client(project=config['gcp']['project_id'])
        bucket = storage_client.bucket(bucket_name)
        
        # Look for previous model metadata - search recursively under base_path
        blobs = list(bucket.list_blobs(prefix=f"{base_path}/"))
        
        # Find metadata files (best_model_metadata.json or model_selection_*.json)
        # Priority: best_model_metadata.json > model_selection_*.json
        metadata_blobs = []
        selection_blobs = []
        
        for blob in blobs:
            blob_name = blob.name
            # Look for best_model_metadata.json (preferred)
            if blob_name.endswith('best_model_metadata.json') or blob_name.endswith('/best_model_metadata.json'):
                metadata_blobs.append(blob)
            # Fallback: model_selection_*.json
            elif 'model-selection' in blob_name and blob_name.endswith('.json') and 'model_selection_' in blob_name:
                selection_blobs.append(blob)
        
        # Use metadata files if available, otherwise use selection reports
        target_blobs = metadata_blobs if metadata_blobs else selection_blobs
        
        if not target_blobs:
            print("⚠ No previous model metadata found in GCS")
            return None
        
        # Sort by time, get second most recent (previous model)
        target_blobs.sort(key=lambda b: b.updated, reverse=True)
        
        if len(target_blobs) < 2:
            # No previous model (this is the first deployment)
            print("⚠ Only one model found in registry (first deployment)")
            return None
        
        previous_blob = target_blobs[1]  # Second most recent
        print(f"  Found previous model metadata: {previous_blob.name}")
        
        metadata_json = previous_blob.download_as_text()
        metadata = json.loads(metadata_json)
        
        # Normalize metadata structure (handle both formats)
        # If it's a selection report, extract best_model
        if 'best_model' in metadata:
            best_model = metadata['best_model']
            return {
                'selected_model': best_model.get('name', 'unknown'),
                'composite_score': best_model.get('composite_score', 0),
                'performance_score': best_model.get('performance_score', 0),
                'bias_score': best_model.get('bias_score', 0),
                'success_rate': best_model.get('success_rate', 0),
                'response_time': best_model.get('response_time', 0),
                'selection_date': metadata.get('selection_timestamp', metadata.get('selection_date'))
            }
        else:
            # Already in the correct format (from best_model_metadata.json)
            return metadata
        
    except Exception as e:
        print(f"⚠ Could not load previous model: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(current: Dict, previous: Optional[Dict], config: Dict) -> Dict:
    """Compare current model with previous model"""
    rollback_config = config.get('rollback', {})
    enabled = rollback_config.get('enabled', True)
    min_improvement = rollback_config.get('min_improvement_threshold', 0.0)
    
    if not previous:
        print("✓ No previous model found - deploying new model")
        return {
            "should_deploy": True,
            "reason": "No previous model to compare",
            "current_score": current.get('composite_score', 0),
            "previous_score": None
        }
    
    current_score = current.get('composite_score', 0)
    previous_score = previous.get('composite_score', 0)
    
    improvement = current_score - previous_score
    
    if not enabled:
        print("⚠ Rollback disabled - deploying regardless of comparison")
        return {
            "should_deploy": True,
            "reason": "Rollback disabled",
            "current_score": current_score,
            "previous_score": previous_score,
            "improvement": improvement
        }
    
    if improvement >= min_improvement:
        print(f"✓ New model is better (improvement: {improvement:.2f})")
        return {
            "should_deploy": True,
            "reason": f"Model improved by {improvement:.2f} points",
            "current_score": current_score,
            "previous_score": previous_score,
            "improvement": improvement
        }
    else:
        print(f"✗ New model is worse (degradation: {improvement:.2f})")
        return {
            "should_deploy": False,
            "reason": f"Model degraded by {abs(improvement):.2f} points",
            "current_score": current_score,
            "previous_score": previous_score,
            "improvement": improvement
        }

def main():
    """Main rollback check function"""
    print("=" * 70)
    print("MODEL COMPARISON & ROLLBACK CHECK")
    print("=" * 70)
    
    try:
        config = load_config()
        print("✓ Loaded CI/CD configuration")
        
        current_metrics = get_current_model_metrics()
        if not current_metrics:
            print("✗ No current model metrics found")
            return 1
        
        print(f"✓ Current model: {current_metrics.get('name')}")
        print(f"  Composite Score: {current_metrics.get('composite_score', 0):.2f}")
        
        previous_metrics = get_previous_model_metrics(config)
        if previous_metrics:
            print(f"✓ Previous model: {previous_metrics.get('selected_model', 'unknown')}")
            print(f"  Composite Score: {previous_metrics.get('composite_score', 0):.2f}")
        else:
            print("⚠ No previous model found")
        
        comparison = compare_models(current_metrics, previous_metrics, config)
        
        # Save comparison report
        comparison_report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "comparison": comparison,
            "current_model": current_metrics,
            "previous_model": previous_metrics
        }
        
        comparison_dir = outputs_dir / "validation"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_file = comparison_dir / "model_comparison_report.json"
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        print("\n" + "=" * 70)
        if comparison['should_deploy']:
            print("DEPLOYMENT APPROVED")
            print("=" * 70)
            print(f"  Reason: {comparison['reason']}")
            print(f"\nComparison report saved: {comparison_file}")
            return 0
        else:
            print("DEPLOYMENT BLOCKED - ROLLBACK RECOMMENDED")
            print("=" * 70)
            print(f"  Reason: {comparison['reason']}")
            print(f"  Current Score: {comparison['current_score']:.2f}")
            print(f"  Previous Score: {comparison['previous_score']:.2f}")
            print(f"\nComparison report saved: {comparison_file}")
            return 1
            
    except Exception as e:
        print(f"\n✗ COMPARISON ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        # On error, allow deployment (fail open)
        return 0

if __name__ == "__main__":
    sys.exit(main())

