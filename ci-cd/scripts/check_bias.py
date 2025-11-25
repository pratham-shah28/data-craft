#!/usr/bin/env python3
"""
Check Bias Detection Results
Validates that bias is within acceptable thresholds using best model responses
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Optional

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "validation_thresholds.yaml"
outputs_dir = project_root / "outputs"

def load_thresholds():
    """Load bias thresholds"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['validation_thresholds']['bias']

def find_latest_selection_report() -> Optional[Path]:
    """Find the latest model selection report in best-model-responses"""
    best_model_dir = outputs_dir / "best-model-responses"
    if not best_model_dir.exists():
        return None
    
    # Look for model_selection_report.json
    reports = list(best_model_dir.rglob("model_selection_report.json"))
    if reports:
        return max(reports, key=lambda p: p.stat().st_mtime)
    
    return None

def main():
    """Check bias detection results"""
    print("=" * 70)
    print("BIAS DETECTION CHECK")
    print("=" * 70)
    
    try:
        thresholds = load_thresholds()
        print("✓ Loaded bias thresholds")
        
        selection_report_path = find_latest_selection_report()
        if not selection_report_path:
            print("✗ No model selection report found")
            return 1
        
        print(f"✓ Found selection report: {selection_report_path.name}")
        
        with open(selection_report_path, 'r') as f:
            selection_data = json.load(f)
        
        # Get best model info
        best_model = selection_data.get('best_model', {})
        if not best_model:
            print("✗ No best model found in selection report")
            return 1
        
        model_name = best_model.get('name', 'unknown')
        bias_score = best_model.get('bias_score', 100)
        
        # Determine severity based on bias score (matching bias_detector.py logic)
        # From bias_detector.py: _get_severity_rating method
        if bias_score >= 60:
            severity = 'HIGH'
        elif bias_score >= 30:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        print(f"\nBest Model: {model_name}")
        print(f"  Bias Score: {bias_score:.2f}/100")
        print(f"  Severity: {severity}")
        
        errors = []
        
        # Check bias score threshold
        if bias_score > thresholds['max_bias_score']:
            errors.append(
                f"Bias score {bias_score:.2f} exceeds threshold "
                f"{thresholds['max_bias_score']}"
            )
        
        # Check severity threshold
        severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        max_severity_level = severity_levels.get(thresholds['max_severity'], 2)
        model_severity_level = severity_levels.get(severity, 3)
        
        if model_severity_level > max_severity_level:
            errors.append(
                f"Bias severity '{severity}' exceeds threshold "
                f"'{thresholds['max_severity']}'"
            )
        
        # Save check report
        check_report = {
            "status": "passed" if not errors else "failed",
            "bias_score": bias_score,
            "severity": severity,
            "errors": errors,
            "model": model_name
        }
        
        check_dir = outputs_dir / "validation"
        check_dir.mkdir(parents=True, exist_ok=True)
        check_file = check_dir / "bias_check_report.json"
        
        with open(check_file, 'w') as f:
            json.dump(check_report, f, indent=2)
        
        print("\n" + "=" * 70)
        if errors:
            print("BIAS CHECK FAILED")
            print("=" * 70)
            for error in errors:
                print(f"  ✗ {error}")
            print(f"\nReport saved: {check_file}")
            return 1
        else:
            print("BIAS CHECK PASSED")
            print("=" * 70)
            print(f"  ✓ Bias score within acceptable range")
            print(f"  ✓ Severity level acceptable")
            print(f"\nReport saved: {check_file}")
            return 0
            
    except Exception as e:
        print(f"\n✗ BIAS CHECK ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
